use ash::vk;
use std::error::Error;

use super::{commands::MAX_FRAMES_IN_FLIGHT, device::Device};

/// Holds the synchronization objects used by the frame loop.
///
/// In simple terms:
/// - `image_available` means "a swapchain image is ready to render into"
/// - `render_finished` means "rendering is done, this image can be presented"
/// - `fences` let the CPU wait until a frame's GPU work is finished
pub struct FrameSync {
    pub image_available: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    pub render_finished: Vec<vk::Semaphore>,
    pub fences: [vk::Fence; MAX_FRAMES_IN_FLIGHT],
}

impl FrameSync {
    /// Create the semaphores and fences needed for rendering frames safely.
    pub fn new(device: &Device, swapchain_image_count: usize) -> Result<Self, Box<dyn Error>> {
        let sem_info = vk::SemaphoreCreateInfo::default();

        // Start fences in the signaled state so the first frame does not block forever.
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        unsafe {
            // One "image available" semaphore per frame in flight.
            let image_available = std::array::from_fn(|_| {
                device.device.create_semaphore(&sem_info, None).unwrap()
            });

            // One "render finished" semaphore per swapchain image.
            let render_finished = (0..swapchain_image_count)
                .map(|_| device.device.create_semaphore(&sem_info, None).unwrap())
                .collect();

            // One fence per frame in flight so the CPU can wait for old frame work.
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

    /// Destroy all synchronization objects owned by this struct.
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
