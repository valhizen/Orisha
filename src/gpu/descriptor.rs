use ash::vk;
use std::error::Error;

use super::buffer::GpuBuffer;
use super::commands::MAX_FRAMES_IN_FLIGHT;
use super::device::Device;

// Descriptor pool, layout, and per-frame sets for camera UBO.
pub struct Descriptors {
    pub pool: vk::DescriptorPool,
    pub camera_layout: vk::DescriptorSetLayout,
    pub camera_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
}

impl Descriptors {
    pub fn new(device: &Device) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(std::slice::from_ref(&binding));

            let camera_layout = device
                .device
                .create_descriptor_set_layout(&layout_info, None)?;

            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
                .pool_sizes(std::slice::from_ref(&pool_size));

            let pool = device.device.create_descriptor_pool(&pool_info, None)?;

            let layouts = [camera_layout; MAX_FRAMES_IN_FLIGHT];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            let sets_vec = device.device.allocate_descriptor_sets(&alloc_info)?;
            let camera_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT] =
                sets_vec.try_into().unwrap();

            Ok(Self {
                pool,
                camera_layout,
                camera_sets,
            })
        }
    }

    pub fn write_camera_sets(
        &self,
        device: &Device,
        camera_ubos: &[GpuBuffer],
        ubo_size: vk::DeviceSize,
    ) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(camera_ubos[i].buffer)
                .offset(0)
                .range(ubo_size);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.camera_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));

            unsafe {
                device.device.update_descriptor_sets(&[write], &[]);
            }
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.device.destroy_descriptor_pool(self.pool, None);
            device
                .device
                .destroy_descriptor_set_layout(self.camera_layout, None);
        }
    }
}
