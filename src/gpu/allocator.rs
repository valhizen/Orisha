use std::sync::Mutex;

use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};

use super::device::Device;

/// Small wrapper around the `gpu-allocator` Vulkan allocator.
///
/// This keeps memory allocation code in one place instead of spreading it
/// across buffers, images, and textures.
pub struct GpuAllocator {
    /// The allocator is wrapped in a mutex because allocation/free need
    /// mutable access, while many systems may share this wrapper.
    inner: Mutex<Allocator>,
}

impl GpuAllocator {
    /// Create the Vulkan memory allocator using the already-created device.
    pub fn new(device: &Device) -> Self {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: device.instance.clone(),
            device: device.device.clone(),
            physical_device: device.pdevice,
            debug_settings: Default::default(),
            // Enabled because some Vulkan features may need buffer addresses.
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .expect("Failed to create GPU memory allocator");

        Self {
            inner: Mutex::new(allocator),
        }
    }

    /// Allocate one Vulkan memory block that matches the given requirements.
    pub fn allocate(&self, desc: &AllocationCreateDesc<'_>) -> Allocation {
        self.inner
            .lock()
            .unwrap()
            .allocate(desc)
            .expect("GPU memory allocation failed")
    }

    /// Free a previously allocated Vulkan memory block.
    pub fn free(&self, allocation: Allocation) {
        self.inner
            .lock()
            .unwrap()
            .free(allocation)
            .expect("GPU memory free failed");
    }
}
