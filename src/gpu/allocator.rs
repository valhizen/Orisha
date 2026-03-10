use std::sync::Mutex;

use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};

use super::device::Device;

/// Thread-safe GPU memory sub-allocator.
pub struct GpuAllocator {
    inner: Mutex<Allocator>,
}

impl GpuAllocator {
    pub fn new(device: &Device) -> Self {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: device.instance.clone(),
            device: device.device.clone(),
            physical_device: device.pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .expect("Failed to create GPU memory allocator");

        Self {
            inner: Mutex::new(allocator),
        }
    }

    pub fn allocate(&self, desc: &AllocationCreateDesc<'_>) -> Allocation {
        self.inner
            .lock()
            .unwrap()
            .allocate(desc)
            .expect("GPU memory allocation failed")
    }

    pub fn free(&self, allocation: Allocation) {
        self.inner
            .lock()
            .unwrap()
            .free(allocation)
            .expect("GPU memory free failed");
    }
}
