use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
    MemoryLocation,
};

use super::{allocator::GpuAllocator, commands::Commands, device::Device};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(24),
        ]
    }
}

/// Camera matrices (set 0, binding 0).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUbo {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}

/// Per-object model matrix via push constants (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConstants {
    pub model: [[f32; 4]; 4],
}

/// Vulkan buffer backed by gpu-allocator.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    pub fn new(
        device: &Device,
        allocator: &GpuAllocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Self {
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = device.device.create_buffer(&info, None).unwrap();
            let requirements = device.device.get_buffer_memory_requirements(buffer);

            let allocation = allocator.allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });

            device
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();

            Self {
                buffer,
                allocation: Some(allocation),
                size,
            }
        }
    }

    /// Write data to a host-visible buffer via its mapped pointer.
    pub fn write<T: bytemuck::NoUninit>(&self, data: &T) {
        let alloc = self.allocation.as_ref().expect("Buffer already destroyed");
        let ptr = alloc.mapped_ptr().expect("Buffer not mapped");
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytemuck::bytes_of(data).as_ptr(),
                ptr.as_ptr() as *mut u8,
                std::mem::size_of::<T>(),
            );
        }
    }

    /// Upload data to device-local memory via staging buffer.
    pub fn device_local(
        device: &Device,
        allocator: &GpuAllocator,
        commands: &Commands,
        data: &[u8],
        usage: vk::BufferUsageFlags,
        name: &str,
    ) -> Self {
        let size = data.len() as vk::DeviceSize;

        let staging = Self::new(
            device,
            allocator,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            &format!("{name}_staging"),
        );

        if let Some(alloc) = &staging.allocation {
            if let Some(ptr) = alloc.mapped_ptr() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        ptr.as_ptr() as *mut u8,
                        data.len(),
                    );
                }
            }
        }

        let gpu_buf = Self::new(
            device,
            allocator,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            name,
        );

        commands.run_one_time(device, |dev, cmd| {
            let region = vk::BufferCopy::default().size(size);
            unsafe { dev.cmd_copy_buffer(cmd, staging.buffer, gpu_buf.buffer, &[region]) };
        });

        staging.destroy(device, allocator);
        gpu_buf
    }

    pub fn destroy(mut self, device: &Device, allocator: &GpuAllocator) {
        unsafe { device.device.destroy_buffer(self.buffer, None) };
        if let Some(alloc) = self.allocation.take() {
            allocator.free(alloc);
        }
    }
}

/// Uploaded vertex + index buffers ready for indexed draw calls.
pub struct GpuMesh {
    pub vertex_buffer: GpuBuffer,
    pub index_buffer: GpuBuffer,
    pub index_count: u32,
}

impl GpuMesh {
    pub fn upload(
        device: &Device,
        allocator: &GpuAllocator,
        commands: &Commands,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Self {
        let vb = GpuBuffer::device_local(
            device,
            allocator,
            commands,
            bytemuck::cast_slice(vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "mesh_vertices",
        );

        let ib = GpuBuffer::device_local(
            device,
            allocator,
            commands,
            bytemuck::cast_slice(indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
            "mesh_indices",
        );

        Self {
            vertex_buffer: vb,
            index_buffer: ib,
            index_count: indices.len() as u32,
        }
    }

    pub fn destroy(self, device: &Device, allocator: &GpuAllocator) {
        self.vertex_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
    }
}
