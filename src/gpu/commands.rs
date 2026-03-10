use ash::vk;
use std::error::Error;

use super::device::Device;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Commands {
    pub pool: vk::CommandPool,
    pub draw_buffers: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    pub setup_buffer: vk::CommandBuffer,
}

impl Commands {
    pub fn new(device: &Device) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(device.queue_family_index);

            let pool = device.device.create_command_pool(&pool_info, None)?;

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

    /// Record and submit a one-shot command, blocking until complete.
    pub fn run_one_time<F>(&self, device: &Device, f: F)
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        unsafe {
            let cmd = self.setup_buffer;

            device
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .unwrap();

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.device.begin_command_buffer(cmd, &begin).unwrap();

            f(&device.device, cmd);

            device.device.end_command_buffer(cmd).unwrap();

            let fence = device
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();

            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            device
                .device
                .queue_submit(device.present_queue, &[submit], fence)
                .unwrap();

            device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();
            device.device.destroy_fence(fence, None);
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.device.destroy_command_pool(self.pool, None) };
    }
}
