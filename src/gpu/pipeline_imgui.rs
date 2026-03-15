use ash::vk;
use gpu_allocator::MemoryLocation;
use std::error::Error;

use super::{
    allocator::GpuAllocator,
    buffer::GpuBuffer,
    commands::{Commands, MAX_FRAMES_IN_FLIGHT},
    device::Device,
    swapchain::SwapChain,
};

// Push constants used by the ImGui vertex shader.
// They convert pixel-space UI coordinates into Vulkan clip space.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ImGuiPush {
    scale:     [f32; 2],
    translate: [f32; 2],
}

// Per-frame CPU-visible buffers that store ImGui-generated vertices and indices.
// ImGui rebuilds its mesh every frame, so these buffers are updated often.
struct FrameBuffers {
    vertex: Option<GpuBuffer>,
    index:  Option<GpuBuffer>,
}

impl FrameBuffers {
    fn new() -> Self {
        Self {
            vertex: None,
            index:  None,
        }
    }

    // Free any per-frame ImGui buffers that were allocated.
    fn destroy(&mut self, device: &Device, allocator: &GpuAllocator) {
        if let Some(buf) = self.vertex.take() {
            buf.destroy(device, allocator);
        }
        if let Some(buf) = self.index.take() {
            buf.destroy(device, allocator);
        }
    }
}

// Owns all Vulkan objects needed to render Dear ImGui:
// - pipeline and layout
// - font texture
// - descriptor set for the font atlas
// - dynamic vertex/index buffers for each frame in flight
pub struct ImGuiPipeline {
    pipeline:        vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    desc_layout:     vk::DescriptorSetLayout,
    desc_pool:       vk::DescriptorPool,
    desc_set:        vk::DescriptorSet,
    font_image:      vk::Image,
    font_view:       vk::ImageView,
    font_sampler:    vk::Sampler,
    font_alloc:      Option<gpu_allocator::vulkan::Allocation>,
    frame_buffers:   [FrameBuffers; MAX_FRAMES_IN_FLIGHT],
}

impl ImGuiPipeline {
    // Create all Vulkan resources needed for UI rendering.
    pub fn new(
        device:    &Device,
        allocator: &GpuAllocator,
        commands:  &Commands,
        swapchain: &SwapChain,
        imgui_ctx: &mut imgui::Context,
    ) -> Result<Self, Box<dyn Error>> {
        // ImGui only needs one sampled image in its descriptor set:
        // the font atlas texture.
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let desc_layout = unsafe {
            device.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(std::slice::from_ref(&binding)),
                None,
            )?
        };

        // The vertex shader uses push constants to convert from pixel coordinates
        // to clip-space coordinates.
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<ImGuiPush>() as u32);

        let pipeline_layout = unsafe {
            device.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&desc_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )?
        };

        let pipeline = Self::create_pipeline(device, swapchain, pipeline_layout)?;

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);

        let desc_pool = unsafe {
            device.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(std::slice::from_ref(&pool_size)),
                None,
            )?
        };

        let desc_set = unsafe {
            device.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(std::slice::from_ref(&desc_layout)),
            )?[0]
        };

        // Ask ImGui to build its font atlas as RGBA8 pixels on the CPU.
        let font_atlas = imgui_ctx.fonts();
        let atlas_texture = font_atlas.build_rgba32_texture();
        let width  = atlas_texture.width;
        let height = atlas_texture.height;
        let pixels = atlas_texture.data;

        // Upload the font atlas to a Vulkan image so the fragment shader can sample it.
        let (font_image, font_view, font_sampler, font_alloc) =
            Self::upload_font_atlas(device, allocator, commands, pixels, width, height);

        // TextureId 0 will refer to the uploaded font atlas.
        font_atlas.tex_id = imgui::TextureId::from(0);

        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(font_view)
            .sampler(font_sampler);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info));

        unsafe { device.device.update_descriptor_sets(&[write], &[]) };

        Ok(Self {
            pipeline,
            pipeline_layout,
            desc_layout,
            desc_pool,
            desc_set,
            font_image,
            font_view,
            font_sampler,
            font_alloc: Some(font_alloc),
            frame_buffers: std::array::from_fn(|_| FrameBuffers::new()),
        })
    }

    // Record draw commands for one ImGui frame.
    pub fn record_commands(
        &mut self,
        device:      &Device,
        allocator:   &GpuAllocator,
        cmd:         vk::CommandBuffer,
        draw_data:   &imgui::DrawData,
        fb_width:    f32,
        fb_height:   f32,
        frame_index: usize,
    ) {
        let total_vtx = draw_data.total_vtx_count as usize;
        let total_idx = draw_data.total_idx_count as usize;

        // Nothing to draw this frame.
        if total_vtx == 0 || total_idx == 0 {
            return;
        }

        let frame = frame_index % MAX_FRAMES_IN_FLIGHT;

        // Compute how much buffer space this frame's UI mesh needs.
        let vtx_size = (total_vtx * std::mem::size_of::<imgui::DrawVert>()) as vk::DeviceSize;
        let idx_size = (total_idx * std::mem::size_of::<imgui::DrawIdx>()) as vk::DeviceSize;

        let fb = &mut self.frame_buffers[frame];

        // Grow the CPU-visible vertex buffer if needed.
        if fb.vertex.as_ref().is_none_or(|b| b.size < vtx_size) {
            fb.vertex.take().map(|b| b.destroy(device, allocator));
            fb.vertex = Some(GpuBuffer::new(
                device, allocator, vtx_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
                "imgui_vtx",
            ));
        }

        // Grow the CPU-visible index buffer if needed.
        if fb.index.as_ref().is_none_or(|b| b.size < idx_size) {
            fb.index.take().map(|b| b.destroy(device, allocator));
            fb.index = Some(GpuBuffer::new(
                device, allocator, idx_size,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
                "imgui_idx",
            ));
        }

        let vb = fb.vertex.as_ref().unwrap();
        let ib = fb.index.as_ref().unwrap();

        // These buffers are CPU-mapped, so we can copy ImGui data directly into them.
        let vb_ptr = vb.allocation.as_ref().unwrap().mapped_ptr().unwrap().as_ptr() as *mut u8;
        let ib_ptr = ib.allocation.as_ref().unwrap().mapped_ptr().unwrap().as_ptr() as *mut u8;

        let mut vtx_offset_bytes: usize = 0;
        let mut idx_offset_bytes: usize = 0;

        // Copy each ImGui draw list into the big per-frame GPU buffers.
        for draw_list in draw_data.draw_lists() {
            let vtx_buf = draw_list.vtx_buffer();
            let idx_buf = draw_list.idx_buffer();

            let vb_bytes = vtx_buf.len() * std::mem::size_of::<imgui::DrawVert>();
            let ib_bytes = idx_buf.len() * std::mem::size_of::<imgui::DrawIdx>();

            unsafe {
                std::ptr::copy_nonoverlapping(
                    vtx_buf.as_ptr() as *const u8,
                    vb_ptr.add(vtx_offset_bytes),
                    vb_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    idx_buf.as_ptr() as *const u8,
                    ib_ptr.add(idx_offset_bytes),
                    ib_bytes,
                );
            }

            vtx_offset_bytes += vb_bytes;
            idx_offset_bytes += ib_bytes;
        }

        unsafe {
            device.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            device.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.desc_set],
                &[],
            );

            // Convert pixel coordinates to clip space in the vertex shader.
            let push = ImGuiPush {
                scale:     [2.0 / fb_width, 2.0 / fb_height],
                translate: [-1.0, -1.0],
            };
            device.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push),
            );

            device.device.cmd_set_viewport(cmd, 0, &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width:  fb_width,
                height: fb_height,
                min_depth: 0.0,
                max_depth: 1.0,
            }]);
        }


        let mut global_vtx_offset: i32 = 0;
        let mut global_idx_offset: u32 = 0;

        // Replay ImGui's draw commands list by list.
        for draw_list in draw_data.draw_lists() {
            let vtx_buf = draw_list.vtx_buffer();
            let idx_buf = draw_list.idx_buffer();

            unsafe {
                device.device.cmd_bind_vertex_buffers(
                    cmd, 0,
                    &[vb.buffer],
                    &[(global_vtx_offset as usize * std::mem::size_of::<imgui::DrawVert>()) as u64],
                );

                device.device.cmd_bind_index_buffer(
                    cmd,
                    ib.buffer,
                    (global_idx_offset as usize * std::mem::size_of::<imgui::DrawIdx>()) as u64,
                    vk::IndexType::UINT16,
                );

                for cmd_params in draw_list.commands() {
                    match cmd_params {
                        imgui::DrawCmd::Elements { count, cmd_params } => {
                            let clip = cmd_params.clip_rect;
                            let scissor_x = (clip[0].max(0.0)) as i32;
                            let scissor_y = (clip[1].max(0.0)) as i32;
                            let scissor_w = ((clip[2] - clip[0]).max(1.0)) as u32;
                            let scissor_h = ((clip[3] - clip[1]).max(1.0)) as u32;

                            // ImGui provides a clip rectangle for each batch.
                            // This becomes a Vulkan scissor rectangle.
                            device.device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                                offset: vk::Offset2D { x: scissor_x, y: scissor_y },
                                extent: vk::Extent2D { width: scissor_w, height: scissor_h },
                            }]);

                            device.device.cmd_draw_indexed(
                                cmd,
                                count as u32,
                                1,
                                cmd_params.idx_offset as u32,
                                cmd_params.vtx_offset as i32,
                                0,
                            );
                        }
                        imgui::DrawCmd::ResetRenderState => {}
                        imgui::DrawCmd::RawCallback { .. } => {}
                    }
                }
            }

            global_vtx_offset += vtx_buf.len() as i32;
            global_idx_offset += idx_buf.len() as u32;
        }
    }


    // Create the graphics pipeline used only for ImGui.
    fn create_pipeline(
        device:    &Device,
        swapchain: &SwapChain,
        layout:    vk::PipelineLayout,
    ) -> Result<vk::Pipeline, Box<dyn Error>> {
        let vert = Self::load_shader(device, "shaders/compiled/imgui_vert.spv")?;
        let frag = Self::load_shader(device, "shaders/compiled/imgui_frag.spv")?;

        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(entry),
        ];

        let binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<imgui::DrawVert>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);

        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(16),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding))
            .vertex_attribute_descriptions(&attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);

        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            );

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));

        let color_formats = [swapchain.surface_format.format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering_info)
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout);

        let pipeline = unsafe {
            device
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| e.1)?[0]
        };

        unsafe {
            device.device.destroy_shader_module(vert, None);
            device.device.destroy_shader_module(frag, None);
        }

        Ok(pipeline)
    }

    // Load a compiled SPIR-V shader module from disk.
    fn load_shader(device: &Device, path: &str) -> Result<vk::ShaderModule, Box<dyn Error>> {
        let bytes = std::fs::read(path)?;
        let (prefix, code, suffix) = unsafe { bytes.align_to::<u32>() };
        assert!(
            prefix.is_empty() && suffix.is_empty(),
            "SPIR-V data is misaligned"
        );
        let info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe { Ok(device.device.create_shader_module(&info, None)?) }
    }

    // Upload ImGui's font atlas pixels into a GPU image and create a view + sampler.
    fn upload_font_atlas(
        device:    &Device,
        allocator: &GpuAllocator,
        commands:  &Commands,
        pixels:    &[u8],
        width:     u32,
        height:    u32,
    ) -> (vk::Image, vk::ImageView, vk::Sampler, gpu_allocator::vulkan::Allocation) {
        use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};

        let format = vk::Format::R8G8B8A8_UNORM;

        unsafe {
            // Create the final GPU image that the fragment shader will sample.
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let image = device.device.create_image(&image_info, None).unwrap();
            let requirements = device.device.get_image_memory_requirements(image);

            let allocation = allocator.allocate(&AllocationCreateDesc {
                name:              "imgui_font_atlas",
                requirements,
                location:          MemoryLocation::GpuOnly,
                linear:            false,
                allocation_scheme: AllocationScheme::DedicatedImage(image),
            });

            device
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();

            // Create a staging buffer so we can copy CPU font pixels into the GPU image.
            let staging = GpuBuffer::new(
                device, allocator,
                pixels.len() as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "imgui_font_staging",
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

            let subresource = vk::ImageSubresourceRange {
                aspect_mask:      vk::ImageAspectFlags::COLOR,
                base_mip_level:   0,
                level_count:      1,
                base_array_layer: 0,
                layer_count:      1,
            };

            commands.run_one_time(device, |dev, cmd| {
                // Transition the image so it can receive copied pixel data.
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

                // Copy staged CPU pixel data into the GPU image.
                dev.cmd_copy_buffer_to_image(
                    cmd, staging.buffer, image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );

                // Transition the image into the layout used by shader sampling.
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

            // The font atlas uses simple linear sampling and clamps at the edges.
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .min_lod(0.0)
                .max_lod(0.25);

            let sampler = device.device.create_sampler(&sampler_info, None).unwrap();

            (image, view, sampler, allocation)
        }
    }

    // Destroy all Vulkan objects owned by the ImGui renderer.
    pub fn destroy(&mut self, device: &Device, allocator: &GpuAllocator) {
        for fb in &mut self.frame_buffers {
            fb.destroy(device, allocator);
        }

        unsafe {
            device.device.destroy_sampler(self.font_sampler, None);
            device.device.destroy_image_view(self.font_view, None);
            device.device.destroy_image(self.font_image, None);
        }
        if let Some(alloc) = self.font_alloc.take() {
            allocator.free(alloc);
        }

        unsafe {
            device.device.destroy_pipeline(self.pipeline, None);
            device.device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.device.destroy_descriptor_pool(self.desc_pool, None);
            device.device.destroy_descriptor_set_layout(self.desc_layout, None);
        }
    }
}
