use ash::vk;
use std::error::Error;

use super::{commands::MAX_FRAMES_IN_FLIGHT, device::Device};

pub struct FrameSync {
    pub image_available: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    /// Indexed by swapchain image (not frame slot) to avoid presentation conflicts
    pub render_finished: Vec<vk::Semaphore>,
    pub fences: [vk::Fence; MAX_FRAMES_IN_FLIGHT],
}

impl FrameSync {
    pub fn new(device: &Device, swapchain_image_count: usize) -> Result<Self, Box<dyn Error>> {
        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        unsafe {
            let image_available = std::array::from_fn(|_| {
                device.device.create_semaphore(&sem_info, None).unwrap()
            });

            let render_finished = (0..swapchain_image_count)
                .map(|_| device.device.create_semaphore(&sem_info, None).unwrap())
                .collect();

            let fences = std::array::from_fn(|_| {
                device.device.create_fence(&fence_info, None).unwrap()
            });

            Ok(Self {
                image_available,
                render_finished,
                fences,
            })
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            for &s in &self.image_available {
                device.device.destroy_semaphore(s, None);
            }
            for &s in &self.render_finished {
                device.device.destroy_semaphore(s, None);
            }
            for &f in &self.fences {
                device.device.destroy_fence(f, None);
            }
        }
    }
}
