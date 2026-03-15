use ash::vk;
use std::error::Error;

// Main graphics pipeline for terrain, player mesh, and other regular 3D objects.
use super::buffer::{PushConstants, Vertex};
use super::device::Device;
use super::swapchain::SwapChain;
use super::utils::load_shader;

/// Holds the Vulkan pipeline object and its layout.
///
/// The layout describes which descriptor sets and push constants shaders expect.
pub struct Pipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl Pipeline {
    /// Creates the full graphics pipeline in two steps:
    /// 1. create the pipeline layout
    /// 2. create the actual graphics pipeline
    pub fn new(
        device: &Device,
        swapchain: &SwapChain,
        camera_layout: vk::DescriptorSetLayout,
        depth_format: vk::Format,
    ) -> Result<Self, Box<dyn Error>> {
        let layout = Self::create_layout(device, camera_layout)?;
        let pipeline = Self::create_pipeline(device, swapchain, layout, depth_format)?;
        Ok(Self {
            pipeline_layout: layout,
            pipeline,
        })
    }

    /// Creates the pipeline layout.
    ///
    /// This tells Vulkan which descriptor set layouts are used and how large
    /// the push constant block is.
    fn create_layout(
        device: &Device,
        camera_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout, Box<dyn Error>> {
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);

        // The main pipeline currently uses one descriptor set layout:
        // the camera/material set created elsewhere.
        let set_layouts = [camera_layout];

        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));

        unsafe { Ok(device.device.create_pipeline_layout(&info, None)?) }
    }

    /// Creates the actual graphics pipeline state object.
    ///
    /// This bundles shaders, vertex layout, rasterization, depth testing,
    /// blending, and dynamic rendering settings into one Vulkan pipeline.
    fn create_pipeline(
        device: &Device,
        swapchain: &SwapChain,
        layout: vk::PipelineLayout,
        depth_format: vk::Format,
    ) -> Result<vk::Pipeline, Box<dyn Error>> {
        // Load the compiled SPIR-V shaders for the main 3D pass.
        let vert = load_shader(device, "shaders/compiled/basic_vert.spv")?;
        let frag = load_shader(device, "shaders/compiled/basic_frag.spv")?;

        // Both shader modules use the standard GLSL entry point: `main`.
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

        // Describe how one vertex is laid out in memory.
        let binding = Vertex::binding_description();
        let attributes = Vertex::attribute_descriptions();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding))
            .vertex_attribute_descriptions(&attributes);

        // Input triangles are provided as independent triangle lists.
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // Viewport and scissor are dynamic, so they are set while recording commands.
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Fill polygons, cull back faces, and treat counter-clockwise triangles as front faces.
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        // No MSAA yet: one sample per pixel.
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Standard depth testing for 3D geometry.
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        // Color blending is disabled for the main opaque pass.
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            );

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));

        // This project uses dynamic rendering instead of a classic render pass.
        let color_formats = [swapchain.surface_format.format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);

        // Combine all the small pipeline state structs into one pipeline description.
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

        // Shader modules are only needed during pipeline creation,
        // so they can be destroyed afterwards.
        unsafe {
            device.device.destroy_shader_module(vert, None);
            device.device.destroy_shader_module(frag, None);
        }

        Ok(pipeline)
    }

    /// Destroys the Vulkan pipeline and its layout.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.device.destroy_pipeline(self.pipeline, None);
            device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
