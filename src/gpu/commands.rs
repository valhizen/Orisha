use ash::vk;
use std::error::Error;

use super::device::Device;

/// Number of frames the renderer can work on at the same time.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Owns the Vulkan command pool and the command buffers allocated from it.
///
/// This project uses:
/// - one setup buffer for short upload/copy jobs
/// - one draw buffer per frame in flight
pub struct Commands {
    pub pool: vk::CommandPool,
    pub draw_buffers: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    pub setup_buffer: vk::CommandBuffer,
}

impl Commands {
    /// Creates the command pool and all command buffers needed by the renderer.
    pub fn new(device: &Device) -> Result<Self, Box<dyn Error>> {
        unsafe {
            // The pool is tied to the graphics queue family.
            // RESET_COMMAND_BUFFER lets us reset command buffers one by one.
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(device.queue_family_index);

            let pool = device.device.create_command_pool(&pool_info, None)?;

            // Allocate one setup buffer plus one draw buffer for each frame in flight.
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1 + MAX_FRAMES_IN_FLIGHT as u32)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let buffers = device.device.allocate_command_buffers(&alloc_info)?;

            let setup_buffer = buffers[0];
            let draw_buffers = buffers[1..][..MAX_FRAMES_IN_FLIGHT].try_into().unwrap();

            Ok(Self {
                pool,
                draw_buffers,
                setup_buffer,
            })
        }
    }

    /// Records and submits a short command buffer for one-time work.
    ///
    /// This is mainly used for setup tasks such as:
    /// - buffer copies
    /// - texture uploads
    /// - image layout transitions during resource creation
    ///
    /// The function waits for completion before returning, so it is simple
    /// to understand, but it is also fully synchronous.
    pub fn run_one_time<F>(&self, device: &Device, f: F)
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        unsafe {
            let cmd = self.setup_buffer;

            // Start from a clean command buffer each time.
            device
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .unwrap();

            // Tell Vulkan this command buffer will be submitted only once.
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.device.begin_command_buffer(cmd, &begin).unwrap();

            // Let the caller record whatever commands are needed.
            f(&device.device, cmd);

            device.device.end_command_buffer(cmd).unwrap();

            // Use a temporary fence so we can wait until the GPU finishes this job.
            let fence = device
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();

            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device
                .device
                .queue_submit(device.present_queue, &[submit], fence)
                .unwrap();

            // Block until the one-time work is complete, then clean up the fence.
            device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();
            device.device.destroy_fence(fence, None);
        }
    }

    /// Destroys the command pool.
    ///
    /// Destroying the pool also frees the command buffers allocated from it.
    pub fn destroy(&self, device: &Device) {
        unsafe { device.device.destroy_command_pool(self.pool, None) };
    }
}
