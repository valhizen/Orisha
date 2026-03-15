use ash::vk;
use std::error::Error;

use super::buffer::GpuBuffer;
use super::commands::MAX_FRAMES_IN_FLIGHT;
use super::device::Device;
use super::texture::GpuTexture;

/// Owns the descriptor pool, descriptor set layout, and per-frame descriptor sets.
///
/// In this project, one descriptor set contains:
/// - the camera uniform buffer
/// - several sampled textures used by the main fragment shader
pub struct Descriptors {
    pub pool:           vk::DescriptorPool,
    pub camera_layout:  vk::DescriptorSetLayout,
    pub camera_sets:    [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
}

impl Descriptors {
    /// Creates the descriptor set layout, descriptor pool, and one descriptor set
    /// for each frame in flight.
    pub fn new(device: &Device) -> Result<Self, Box<dyn Error>> {
        unsafe {

            // These bindings must match what the shaders expect.
            let bindings = [
                // Binding 0 = camera UBO, used by both vertex and fragment shaders.
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    ),
                // Bindings 1..5 = textures used by the terrain/world shader.
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];

            let layout_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

            // Create the descriptor set layout shared by all per-frame sets.
            let camera_layout = device
                .device
                .create_descriptor_set_layout(&layout_info, None)?;


            // The pool must have enough space for all descriptors across all sets.
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(5 * MAX_FRAMES_IN_FLIGHT as u32),
            ];

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
                .pool_sizes(&pool_sizes);

            let pool = device.device.create_descriptor_pool(&pool_info, None)?;


            // Allocate one descriptor set per frame in flight.
            let layouts    = [camera_layout; MAX_FRAMES_IN_FLIGHT];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            let sets_vec   = device.device.allocate_descriptor_sets(&alloc_info)?;
            let camera_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT] =
                sets_vec.try_into().unwrap();

            Ok(Self {
                pool,
                camera_layout,
                camera_sets,
            })
        }
    }

    /// Writes the camera uniform buffer into binding 0 for each frame's descriptor set.
    pub fn write_camera_sets(
        &self,
        device:    &Device,
        camera_ubos: &[GpuBuffer],
        ubo_size:  vk::DeviceSize,
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

    /// Writes all terrain/world textures into bindings 1..5 for each frame's set.
    pub fn write_texture_sets(
        &self,
        device:    &Device,
        diffuse:   &GpuTexture,
        normal:    &GpuTexture,
        spec:      &GpuTexture,
        roughness: &GpuTexture,
        displace:  &GpuTexture,
    ) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let diff_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(diffuse.view)
                .sampler(diffuse.sampler);

            let norm_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(normal.view)
                .sampler(normal.sampler);

            let spec_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(spec.view)
                .sampler(spec.sampler);

            let rough_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(roughness.view)
                .sampler(roughness.sampler);

            let disp_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(displace.view)
                .sampler(displace.sampler);

            // These writes connect each texture to the binding expected by the shader.
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.camera_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&diff_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.camera_sets[i])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&norm_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.camera_sets[i])
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&spec_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.camera_sets[i])
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&rough_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.camera_sets[i])
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&disp_info)),
            ];

            unsafe {
                device.device.update_descriptor_sets(&writes, &[]);
            }
        }
    }

    /// Destroys the descriptor pool and descriptor set layout.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.device.destroy_descriptor_pool(self.pool, None);
            device
                .device
                .destroy_descriptor_set_layout(self.camera_layout, None);
        }
    }
}
