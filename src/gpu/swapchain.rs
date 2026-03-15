use ash::{khr::swapchain, vk};
use std::error::Error;

// The swapchain owns the images we render into before presenting them on screen.
// You can think of it as the queue of screen images managed by Vulkan.

use super::device::Device;

/// Holds the Vulkan swapchain and the image views created from its images.
pub struct SwapChain {
    pub swapchain_loader: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,
}

impl SwapChain {
    /// Creates the first swapchain when the renderer starts.
    pub fn new(device: &Device, width: u32, height: u32) -> Result<Self, Box<dyn Error>> {
        let (loader, swapchain, format, resolution) =
            Self::create_swapchain(device, width, height, vk::SwapchainKHR::null())?;

        // After creating the swapchain, fetch its images and create views for them.
        let images = unsafe { loader.get_swapchain_images(swapchain)? };
        let views = Self::create_image_views(device, &images, format.format);

        Ok(Self {
            swapchain_loader: loader,
            swapchain,
            surface_format: format,
            surface_resolution: resolution,
            present_images: images,
            present_image_views: views,
        })
    }

    /// Recreates the swapchain, usually after a window resize.
    pub fn recreate(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn Error>> {
        // Wait until the GPU is idle before replacing swapchain resources.
        unsafe { device.device.device_wait_idle()? };

        let old = self.swapchain;
        let (loader, swapchain, format, resolution) =
            Self::create_swapchain(device, width, height, old)?;

        let images = unsafe { loader.get_swapchain_images(swapchain)? };
        let views = Self::create_image_views(device, &images, format.format);

        // Destroy the old image views before replacing them.
        for &view in &self.present_image_views {
            unsafe { device.device.destroy_image_view(view, None) };
        }

        // Destroy the old swapchain after the new one has been created.
        unsafe { loader.destroy_swapchain(old, None) };

        self.swapchain_loader = loader;
        self.swapchain = swapchain;
        self.surface_format = format;
        self.surface_resolution = resolution;
        self.present_images = images;
        self.present_image_views = views;

        Ok(())
    }

    /// Creates a Vulkan swapchain and returns the basic data needed by the renderer.
    fn create_swapchain(
        device: &Device,
        width: u32,
        height: u32,
        old: vk::SwapchainKHR,
    ) -> Result<
        (
            swapchain::Device,
            vk::SwapchainKHR,
            vk::SurfaceFormatKHR,
            vk::Extent2D,
        ),
        Box<dyn Error>,
    > {
        unsafe {
            // Query which surface formats this GPU + window surface supports.
            let formats = device
                .surface_loader
                .get_physical_device_surface_formats(device.pdevice, device.surface)?;

            // Prefer SRGB if available, otherwise fall back to the first supported format.
            let format = formats
                .iter()
                .copied()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_SRGB
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or(formats[0]);

            // Query surface limits such as image count and surface size rules.
            let caps = device
            .surface_loader
            .get_physical_device_surface_capabilities(device.pdevice, device.surface)?;

            // Usually we ask for one more image than the minimum so rendering can overlap better.
            let mut image_count = caps.min_image_count + 1;
            if caps.max_image_count > 0 && image_count > caps.max_image_count {
                image_count = caps.max_image_count;
            }

            // Some platforms force the extent. Others let us choose it.
            let extent = match caps.current_extent.width {
                u32::MAX => vk::Extent2D { width, height },
                _ => caps.current_extent,
            };

            // Prefer identity transform if supported so the image is not rotated by the surface.
            let transform =
                if caps
                    .supported_transforms
                    .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
                {
                    vk::SurfaceTransformFlagsKHR::IDENTITY
                } else {
                    caps.current_transform
                };

            // Prefer MAILBOX for lower latency if available, otherwise use FIFO.
            let present_modes = device
                .surface_loader
                .get_physical_device_surface_present_modes(device.pdevice, device.surface)?;
            let present_mode = present_modes
                .iter()
                .copied()
                .find(|&m| m == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let loader = swapchain::Device::new(&device.instance, &device.device);

            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(device.surface)
                .min_image_count(image_count)
                .image_color_space(format.color_space)
                .image_format(format.format)
                .image_extent(extent)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1)
                .old_swapchain(old);

            let swapchain = loader.create_swapchain(&info, None)?;

            Ok((loader, swapchain, format, extent))
        }
    }

    /// Creates one image view for each swapchain image.
    ///
    /// Vulkan rendering usually works with image views rather than raw images.
    fn create_image_views(
        device: &Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Vec<vk::ImageView> {
        images
            .iter()
            .map(|&image| {
                let info = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                unsafe { device.device.create_image_view(&info, None).unwrap() }
            })
            .collect()
    }

    /// Destroys the image views and the swapchain itself.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            for &view in &self.present_image_views {
                device.device.destroy_image_view(view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
