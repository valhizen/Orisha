use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
    MemoryLocation,
};

use super::{allocator::GpuAllocator, buffer::GpuBuffer, commands::Commands, device::Device};


pub struct GpuTexture {
    pub image:   vk::Image,
    pub view:    vk::ImageView,
    pub sampler: vk::Sampler,
    allocation:  Option<Allocation>,
}

impl GpuTexture {

    pub fn load_rgba8(
        device:   &Device,
        allocator: &GpuAllocator,
        commands: &Commands,
        path:     &str,
        srgb:      bool,
    ) -> Self {
        let img  = image::open(path)
            .unwrap_or_else(|e| panic!("Failed to load texture `{path}`: {e}"));
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let format = if srgb {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        Self::from_rgba8_pixels(device, allocator, commands, rgba.as_raw(), w, h, format)
    }

    pub fn load_exr_as_rgba8(
        device:   &Device,
        allocator: &GpuAllocator,
        commands: &Commands,
        path:     &str,
    ) -> Self {
        let img = match image::open(path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!(
                    "Warning: could not decode EXR `{path}`: {e}\n\
                     Falling back to a 1×1 flat normal map."
                );
                return Self::from_rgba8_pixels(
                    device, allocator, commands,
                    &[128u8, 128, 255, 255],
                    1, 1,
                    vk::Format::R8G8B8A8_UNORM,
                );
            }
        };

        let rgba_f32 = img.to_rgba32f();
        let (w, h)   = rgba_f32.dimensions();

        let pixels: Vec<u8> = rgba_f32
            .iter()
            .map(|&f| (f.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();

        Self::from_rgba8_pixels(
            device, allocator, commands,
            &pixels, w, h,
            vk::Format::R8G8B8A8_UNORM,
        )
    }

    pub fn white_1x1(device: &Device, allocator: &GpuAllocator, commands: &Commands) -> Self {
        Self::from_rgba8_pixels(
            device, allocator, commands,
            &[255u8, 255, 255, 255],
            1, 1,
            vk::Format::R8G8B8A8_UNORM,
        )
    }


    pub fn cleanup(&mut self, device: &Device, allocator: &GpuAllocator) {
        unsafe {
            device.device.destroy_sampler(self.sampler, None);
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
        }
        if let Some(alloc) = self.allocation.take() {
            allocator.free(alloc);
        }
    }

    pub fn destroy(mut self, device: &Device, allocator: &GpuAllocator) {
        self.cleanup(device, allocator);
    }


    fn from_rgba8_pixels(
        device:   &Device,
        allocator: &GpuAllocator,
        commands: &Commands,
        pixels:   &[u8],
        width:    u32,
        height:   u32,
        format:   vk::Format,
    ) -> Self {
        unsafe {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let image = device.device.create_image(&image_info, None).unwrap();
            let requirements = device.device.get_image_memory_requirements(image);

            let allocation = allocator.allocate(&AllocationCreateDesc {
                name:              "texture_image",
                requirements,
                location:          MemoryLocation::GpuOnly,
                linear:            false,
                allocation_scheme: AllocationScheme::DedicatedImage(image),
            });

            device
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();

            let staging = GpuBuffer::new(
                device,
                allocator,
                pixels.len() as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "texture_staging",
            );

            if let Some(alloc) = &staging.allocation {
                if let Some(ptr) = alloc.mapped_ptr() {
                    std::ptr::copy_nonoverlapping(
                        pixels.as_ptr(),
                        ptr.as_ptr() as *mut u8,
                        pixels.len(),
                    );
                }
            }

            let subresource = Self::full_subresource();

            commands.run_one_time(device, |dev, cmd| {
                let to_transfer = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(image)
                    .subresource_range(subresource);

                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[], &[], &[to_transfer],
                );

                // Buffer → Image copy
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask:      vk::ImageAspectFlags::COLOR,
                        mip_level:        0,
                        base_array_layer: 0,
                        layer_count:      1,
                    })
                    .image_offset(vk::Offset3D::default())
                    .image_extent(vk::Extent3D { width, height, depth: 1 });

                dev.cmd_copy_buffer_to_image(
                    cmd,
                    staging.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );

                let to_shader = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(image)
                    .subresource_range(subresource);

                dev.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &[], &[to_shader],
                );
            });

            staging.destroy(device, allocator);


            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(subresource);

            let view = device.device.create_image_view(&view_info, None).unwrap();


            let max_aniso = device
                .instance
                .get_physical_device_properties(device.pdevice)
                .limits
                .max_sampler_anisotropy;

            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(max_aniso)
                .min_lod(0.0)
                .max_lod(0.25);

            let sampler = device.device.create_sampler(&sampler_info, None).unwrap();

            Self {
                image,
                view,
                sampler,
                allocation: Some(allocation),
            }
        }
    }

    fn full_subresource() -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask:      vk::ImageAspectFlags::COLOR,
            base_mip_level:   0,
            level_count:      1,
            base_array_layer: 0,
            layer_count:      1,
        }
    }
}
