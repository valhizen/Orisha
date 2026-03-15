use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
    MemoryLocation,
};

use super::{allocator::GpuAllocator, commands::Commands, device::Device};

/// CPU-side vertex format sent to the GPU.
///
/// This must match the vertex shader input layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub color:    [f32; 3],
    pub uv:       [f32; 2],  // world-space tiled UVs (location 3)
}

impl Vertex {
    /// Describes one vertex binding slot for Vulkan.
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    /// Describes how each field in `Vertex` maps to shader locations.
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
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
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(3)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(36),
        ]
    }
}

/// Camera uniform data sent once per frame.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUbo {
    pub view:    [[f32; 4]; 4],
    pub proj:    [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
}

/// Per-draw push constants for the main world shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConstants {
    pub model:     [[f32; 4]; 4],
    pub tex_blend: f32,
    pub time:      f32,
}

/// Per-draw push constants for the sky shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyPushConstants {
    pub sun_dir:     [f32; 4],
    pub time_of_day: f32,
    pub _pad0:       f32,
    pub _pad1:       f32,
    pub _pad2:       f32,
}

/// Generic Vulkan buffer plus its memory allocation.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    /// Creates a Vulkan buffer and allocates memory for it.
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

    /// Writes CPU data directly into a mapped buffer.
    ///
    /// This is mainly used for uniform buffers.
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

    /// Creates a GPU-only buffer and uploads data into it through a staging buffer.
    ///
    /// This is the usual path for vertex and index buffers.
    pub fn device_local(
        device: &Device,
        allocator: &GpuAllocator,
        commands: &Commands,
        data: &[u8],
        usage: vk::BufferUsageFlags,
        name: &str,
    ) -> Self {
        let size = data.len() as vk::DeviceSize;

        // First create a CPU-visible staging buffer.
        let staging = Self::new(
            device,
            allocator,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            &format!("{name}_staging"),
        );

        // Copy raw bytes into the staging buffer.
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

        // Then create the final GPU-only buffer.
        let gpu_buf = Self::new(
            device,
            allocator,
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            name,
        );

        // Copy data from staging into the final GPU buffer.
        commands.run_one_time(device, |dev, cmd| {
            let region = vk::BufferCopy::default().size(size);
            unsafe { dev.cmd_copy_buffer(cmd, staging.buffer, gpu_buf.buffer, &[region]) };
        });

        staging.destroy(device, allocator);
        gpu_buf
    }

    /// Destroys the Vulkan buffer and frees its memory.
    pub fn destroy(mut self, device: &Device, allocator: &GpuAllocator) {
        unsafe { device.device.destroy_buffer(self.buffer, None) };
        if let Some(alloc) = self.allocation.take() {
            allocator.free(alloc);
        }
    }
}

/// Simple mesh wrapper containing a vertex buffer and an index buffer.
pub struct GpuMesh {
    pub vertex_buffer: GpuBuffer,
    pub index_buffer: GpuBuffer,
    pub index_count: u32,
}

impl GpuMesh {
    /// Uploads CPU-side mesh data into GPU vertex and index buffers.
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

    /// Destroys both GPU buffers owned by the mesh.
    pub fn destroy(self, device: &Device, allocator: &GpuAllocator) {
        self.vertex_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
    }
}
