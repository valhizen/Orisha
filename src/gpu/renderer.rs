use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
    MemoryLocation,
};
use std::error::Error;

use super::{
    allocator::GpuAllocator,
    buffer::{CameraUbo, GpuBuffer},
    commands::{Commands, MAX_FRAMES_IN_FLIGHT},
    descriptor::Descriptors,
    device::Device,
    pipeline::Pipeline,
    pipeline_sky::SkyPipeline,
    swapchain::SwapChain,
    sync::FrameSync,
    texture::GpuTexture,
};

const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

pub struct Renderer {
    pub allocator:  GpuAllocator,
    pub device:     Device,
    pub swapchain:  SwapChain,
    pub commands:   Commands,
    pub sync:       FrameSync,
    pub descriptors: Descriptors,
    pub pipeline:   Pipeline,
    pub sky_pipeline: SkyPipeline,
    camera_ubos:    Vec<GpuBuffer>,
    depth_image:    vk::Image,
    depth_view:     vk::ImageView,
    depth_alloc:    Option<Allocation>,
    terrain_diffuse: GpuTexture,
    terrain_normal:  GpuTexture,
    terrain_spec:    GpuTexture,
    terrain_rough:   GpuTexture,
    terrain_disp:    GpuTexture,
    pub frame_index: usize,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();

        let device    = Device::new(window)?;
        let allocator = GpuAllocator::new(&device);
        let swapchain = SwapChain::new(&device, size.width, size.height)?;
        let commands  = Commands::new(&device)?;
        let sync      = FrameSync::new(&device, swapchain.present_images.len())?;
        let descriptors = Descriptors::new(&device)?;

        let camera_ubos: Vec<GpuBuffer> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                GpuBuffer::new(
                    &device,
                    &allocator,
                    std::mem::size_of::<CameraUbo>() as vk::DeviceSize,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    &format!("camera_ubo_{i}"),
                )
            })
            .collect();

        descriptors.write_camera_sets(
            &device,
            &camera_ubos,
            std::mem::size_of::<CameraUbo>() as vk::DeviceSize,
        );

        // ── Terrain textures ─────────────────────────────────────────────────
        // Load order matters: each upload blocks until the transfer queue is
        // idle so we serialise them naturally here at startup.

        println!("Loading terrain textures (this may take a moment for 4 K assets)…");

        let terrain_diffuse = GpuTexture::load_rgba8(
            &device, &allocator, &commands,
            "Textures/rocky_terrain_02_diff_4k.jpg",
            true,   // sRGB — Vulkan gamma-decodes on sample → linear in shader
        );
        println!("  diffuse  ✓");

        let terrain_normal = GpuTexture::load_rgba8(
            &device, &allocator, &commands,
            "Textures/rocky_terrain_02_nor_gl_4k.png",
            false,
        );
        println!("  normal   ✓");

        let terrain_spec = GpuTexture::load_rgba8(
            &device, &allocator, &commands,
            "Textures/rocky_terrain_02_spec_4k.png",
            false,
        );
        println!("  specular ✓");

        let terrain_rough = GpuTexture::load_rgba8(
            &device, &allocator, &commands,
            "Textures/rocky_terrain_02_rough_4k.png",
            false,
        );
        println!("  roughness ✓");

        let terrain_disp = GpuTexture::load_rgba8(
            &device, &allocator, &commands,
            "Textures/rocky_terrain_02_disp_4k.png",
            false,
        );
        println!("  displacement ✓");

        descriptors.write_texture_sets(
            &device,
            &terrain_diffuse,
            &terrain_normal,
            &terrain_spec,
            &terrain_rough,
            &terrain_disp,
        );

        let (depth_image, depth_view, depth_alloc) =
            Self::create_depth(&device, &allocator, size.width, size.height);

        let pipeline =
            Pipeline::new(&device, &swapchain, descriptors.camera_layout, DEPTH_FORMAT)?;

        let sky_pipeline =
            SkyPipeline::new(&device, &swapchain, descriptors.camera_layout, DEPTH_FORMAT)?;

        Ok(Self {
            allocator,
            device,
            swapchain,
            commands,
            sync,
            descriptors,
            pipeline,
            sky_pipeline,
            camera_ubos,
            depth_image,
            depth_view,
            depth_alloc: Some(depth_alloc),
            terrain_diffuse,
            terrain_normal,
            terrain_spec,
            terrain_rough,
            terrain_disp,
            frame_index: 0,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.swapchain
            .recreate(&self.device, width, height)
            .expect("Swapchain recreate failed");

        unsafe {
            self.device.device.destroy_image_view(self.depth_view, None);
            self.device.device.destroy_image(self.depth_image, None);
        }
        if let Some(alloc) = self.depth_alloc.take() {
            self.allocator.free(alloc);
        }
        let (img, view, alloc) = Self::create_depth(
            &self.device,
            &self.allocator,
            self.swapchain.surface_resolution.width,
            self.swapchain.surface_resolution.height,
        );
        self.depth_image = img;
        self.depth_view = view;
        self.depth_alloc = Some(alloc);
    }

    pub fn draw_frame<F>(&mut self, camera: &CameraUbo, record_fn: F)
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer, vk::PipelineLayout),
    {
        unsafe {
            let frame = self.frame_index % MAX_FRAMES_IN_FLIGHT;

            self.device
                .device
                .wait_for_fences(&[self.sync.fences[frame]], true, u64::MAX)
                .unwrap();
            self.device
                .device
                .reset_fences(&[self.sync.fences[frame]])
                .unwrap();

            let acquire = self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.sync.image_available[frame],
                vk::Fence::null(),
            );

            let image_index = match acquire {
                Ok((idx, _)) => idx as usize,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return,
                Err(e) => panic!("acquire_next_image failed: {e}"),
            };

            self.camera_ubos[frame].write(camera);

            let cmd = self.commands.draw_buffers[frame];
            let extent = self.swapchain.surface_resolution;
            let color_image = self.swapchain.present_images[image_index];
            let color_view = self.swapchain.present_image_views[image_index];

            self.device
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .unwrap();

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.device.begin_command_buffer(cmd, &begin).unwrap();

            let color_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(color_image)
                .subresource_range(Self::color_subresource());

            let depth_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(self.depth_image)
                .subresource_range(Self::depth_subresource());

            let barriers = [color_barrier, depth_barrier];
            self.device.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );

            let color_attach = vk::RenderingAttachmentInfo::default()
                .image_view(color_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.02, 0.02, 0.03, 1.0],
                    },
                });

            let depth_attach = vk::RenderingAttachmentInfo::default()
                .image_view(self.depth_view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });

            let rendering = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attach))
                .depth_attachment(&depth_attach);

            self.device.device.cmd_begin_rendering(cmd, &rendering);

            self.device.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );

            self.device.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout,
                0,
                &[self.descriptors.camera_sets[frame]],
                &[],
            );

            self.device.device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );

            self.device.device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );

            record_fn(&self.device.device, cmd, self.pipeline.pipeline_layout);

            self.device.device.cmd_end_rendering(cmd);

            let to_present = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(color_image)
                .subresource_range(Self::color_subresource());

            self.device.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_present)),
            );

            self.device.device.end_command_buffer(cmd).unwrap();

            let wait_sems = [self.sync.image_available[frame]];
            let signal_sems = [self.sync.render_finished[image_index]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(std::slice::from_ref(&cmd))
                .signal_semaphores(&signal_sems);

            self.device
                .device
                .queue_submit(
                    self.device.present_queue,
                    &[submit],
                    self.sync.fences[frame],
                )
                .expect("Queue submit failed");

            let swapchains = [self.swapchain.swapchain];
            let indices = [image_index as u32];

            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_sems)
                .swapchains(&swapchains)
                .image_indices(&indices);

            match self
                .swapchain
                .swapchain_loader
                .queue_present(self.device.present_queue, &present)
            {
                Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {}
                Err(e) => panic!("queue_present failed: {e}"),
            }

            self.frame_index += 1;
        }
    }

    fn create_depth(
        device: &Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> (vk::Image, vk::ImageView, Allocation) {
        unsafe {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(DEPTH_FORMAT)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let image = device.device.create_image(&info, None).unwrap();
            let requirements = device.device.get_image_memory_requirements(image);

            let alloc = allocator.allocate(&AllocationCreateDesc {
                name: "depth_image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::DedicatedImage(image),
            });

            device
                .device
                .bind_image_memory(image, alloc.memory(), alloc.offset())
                .unwrap();

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(DEPTH_FORMAT)
                .subresource_range(Self::depth_subresource());

            let view = device.device.create_image_view(&view_info, None).unwrap();

            (image, view, alloc)
        }
    }

    fn color_subresource() -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }

    fn depth_subresource() -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe { self.device.device.device_wait_idle().unwrap() };

        self.pipeline.destroy(&self.device);
        self.sky_pipeline.destroy(&self.device);
        self.sync.destroy(&self.device);
        self.descriptors.destroy(&self.device);

        self.terrain_diffuse.cleanup(&self.device, &self.allocator);
        self.terrain_normal .cleanup(&self.device, &self.allocator);
        self.terrain_spec   .cleanup(&self.device, &self.allocator);
        self.terrain_rough  .cleanup(&self.device, &self.allocator);
        self.terrain_disp   .cleanup(&self.device, &self.allocator);

        unsafe {
            self.device.device.destroy_image_view(self.depth_view, None);
            self.device.device.destroy_image(self.depth_image, None);
        }
        if let Some(alloc) = self.depth_alloc.take() {
            self.allocator.free(alloc);
        }

        for ubo in self.camera_ubos.drain(..) {
            ubo.destroy(&self.device, &self.allocator);
        }

        self.commands.destroy(&self.device);
        self.swapchain.destroy(&self.device);
    }
}
