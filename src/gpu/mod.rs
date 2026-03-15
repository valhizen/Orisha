/// GPU memory allocation helpers.
pub mod allocator;
/// GPU buffer types, mesh upload helpers, and shader data structs.
pub mod buffer;
/// Command pool and command buffer helpers.
pub mod commands;
/// Descriptor set layouts, pools, and descriptor writes.
pub mod descriptor;
/// Vulkan instance, physical device, logical device, and queue setup.
pub mod device;
/// Main graphics pipeline used for terrain, player, and world objects.
pub mod pipeline;
/// Dedicated pipeline used to render the ImGui interface.
pub mod pipeline_imgui;
/// Dedicated pipeline used to render the sky dome.
pub mod pipeline_sky;
/// High-level renderer that ties together Vulkan subsystems and frame drawing.
pub mod renderer;
/// Swapchain creation, recreation, and image view management.
pub mod swapchain;
/// Per-frame synchronization objects such as semaphores and fences.
pub mod sync;
/// Texture loading, GPU image upload, and sampler creation.
pub mod texture;
/// Small shared GPU helper utilities.
pub mod utils;
