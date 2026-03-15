use std::{borrow::Cow, error::Error, ffi, os::raw::c_char};

use ash::{ext::debug_utils, khr::surface, vk, Device as AshDevice, Entry, Instance};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Validation/debug callback used by Vulkan when validation layers report
/// warnings or errors.
unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    unsafe {
        let data = *p_data;
        let id = data.message_id_number;
        let name = if data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(data.p_message_id_name).to_string_lossy()
        };
        let msg = if data.p_message.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(data.p_message).to_string_lossy()
        };
        println!("{severity:?}:\n{msg_type:?} [{name} ({id})] : {msg}\n");
        vk::FALSE
    }
}

/// Gives each physical device a score so the app can choose the best GPU.
///
/// A device is rejected with `-1` if it does not support the Vulkan version,
/// features, queue support, or swapchain support that this renderer needs.
fn score_device(
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> i32 {
    unsafe {
        let props = instance.get_physical_device_properties(pdevice);

        // This renderer expects Vulkan 1.3.
        if props.api_version < vk::API_VERSION_1_3 {
            return -1;
        }

        // Query Vulkan 1.3 feature support from the GPU.
        let mut features_13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features_13);
        instance.get_physical_device_features2(pdevice, &mut features2);

        // Reject devices missing features used by the renderer.
        if features_13.dynamic_rendering == vk::FALSE
            || features_13.synchronization2 == vk::FALSE
        {
            return -1;
        }

        // Check whether this GPU has at least one queue family that can both
        // render graphics and present images to the window surface.
        let has_queue = instance
            .get_physical_device_queue_family_properties(pdevice)
            .iter()
            .enumerate()
            .any(|(i, info)| {
                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && surface_loader
                        .get_physical_device_surface_support(pdevice, i as u32, surface)
                        .unwrap_or(false)
            });
        // Get All the SwapChain and Surface features Supported By Vulkan
        // uncomment the code to see it

        // let surface_support = surface_loader.get_physical_device_surface_formats(pdevice, surface);
        // println!("Vulkan SwapChain Support :{ :?}", surface_support);

        if !has_queue { return -1; }

        // Swapchain rendering also needs supported formats and present modes.
        let formats = surface_loader
            .get_physical_device_surface_formats(pdevice, surface)
            .unwrap_or_default();
        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(pdevice, surface)
            .unwrap_or_default();

        if formats.is_empty() || present_modes.is_empty() {
            return -1;
        }

        // Prefer faster GPU classes when multiple valid devices exist.
        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            _ => 1,
        }
    }
}

/// Owns the main Vulkan objects needed for rendering.
///
/// This includes:
/// - Vulkan entry/instance
/// - debug messenger
/// - window surface
/// - logical device
/// - selected physical device
/// - graphics/present queue information
pub struct Device {
    pub entry: Entry,
    pub instance: Instance,
    pub device: AshDevice,
    pub surface_loader: surface::Instance,
    pub debug_utils_loader: debug_utils::Instance,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,
    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,
    pub surface: vk::SurfaceKHR,
}

impl Device {
    /// Creates the Vulkan instance, picks a GPU, and creates the logical device.
    pub fn new(window: &winit::window::Window) -> Result<Self, Box<dyn Error>> {
        unsafe {
            // Entry is the starting point for loading Vulkan function pointers.
            let entry = Entry::linked();
            let app_name = c"Orisha";

            // Validation layers help catch Vulkan mistakes while developing.
            let layer_names = [c"VK_LAYER_KHRONOS_validation"];
            let layers_raw: Vec<*const c_char> =
                layer_names.iter().map(|n| n.as_ptr()).collect();

            // Ask the windowing helper which instance extensions are required
            // for this platform, then add debug utils for validation messages.
            let mut extension_names =
                ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())
                    .unwrap()
                    .to_vec();
            extension_names.push(debug_utils::NAME.as_ptr());



                        #[cfg(any(target_os = "macos", target_os = "ios"))]
                        {
                            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
                            extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
                        }

            // Application/engine info is mostly metadata, but it also declares
            // the Vulkan API version this app wants to use.
            let app_info = vk::ApplicationInfo::default()
                .application_name(app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(c"Orisha Engine")
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_3);

            let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
                       vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
                   } else {
                       vk::InstanceCreateFlags::default()
                   };

            // Create the Vulkan instance.
            let instance_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers_raw)
                .enabled_extension_names(&extension_names)
                .flags(create_flags);

            let instance = entry.create_instance(&instance_info, None).expect("Create Instance Error!");

            // Set up the debug messenger so validation messages are printed.
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None).unwrap();

            let surface_loader = surface::Instance::new(&entry, &instance);

            // Create the OS/window surface so Vulkan can present images to it.
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?;

            // List all physical devices (GPUs) visible to Vulkan.
            let pdevices = instance.enumerate_physical_devices()?;

            println!("── Available GPUs ──");
            for &pd in &pdevices {
                let props = instance.get_physical_device_properties(pd);
                let name = ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
                let score = score_device(&instance, pd, &surface_loader, surface);
                println!("  [{score:4}] {name} ({:?})", props.device_type);
            }

            // Pick the highest-scoring usable GPU.
            let pdevice = pdevices
                .iter()
                .copied()
                .max_by_key(|&pd| score_device(&instance, pd, &surface_loader, surface))
                .filter(|&pd| score_device(&instance, pd, &surface_loader, surface) > 0)
                .expect("No suitable GPU found");

            let sel = instance.get_physical_device_properties(pdevice);
            let name = ffi::CStr::from_ptr(sel.device_name.as_ptr()).to_string_lossy();
            println!("Selected GPU: {name}\n");

            // Find one queue family that supports both graphics commands and
            // presenting to the window surface.
            let queue_family_index = instance
                .get_physical_device_queue_family_properties(pdevice)
                .iter()
                .enumerate()
                .find_map(|(i, info)| {
                    let gfx = info.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                    let present = surface_loader
                        .get_physical_device_surface_support(pdevice, i as u32, surface)
                        .unwrap_or(false);
                    (gfx && present).then_some(i as u32)
                })
                .expect("No graphics+present queue family");

            // Create one logical queue from the chosen family.
            let priorities = [1.0f32];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            // Device extensions add optional Vulkan features.
            let device_extensions = [ash::khr::swapchain::NAME.as_ptr(),
                ash::khr::ray_tracing_pipeline::NAME.as_ptr()
            ];

            // Enable Vulkan 1.2/1.3 feature structs needed by the renderer.
            let mut features_12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true);

            let mut features_13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);

            // Enable core device features.
            let features = vk::PhysicalDeviceFeatures::default()
                .sampler_anisotropy(true);

            // Create the logical device from the selected physical device.
            let device_info = vk::DeviceCreateInfo::default()
                .push_next(&mut features_12)
                .push_next(&mut features_13)
                .enabled_features(&features)
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extensions);

            let device = instance.create_device(pdevice, &device_info, None)?;
            let present_queue = device.get_device_queue(queue_family_index, 0);

            // Store memory properties so other systems can inspect GPU memory types.
            let device_memory_properties =
                instance.get_physical_device_memory_properties(pdevice);

            Ok(Self {
                entry,
                instance,
                device,
                surface_loader,
                debug_utils_loader,
                debug_call_back,
                pdevice,
                device_memory_properties,
                queue_family_index,
                present_queue,
                surface,
            })
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // Wait until the GPU is idle before destroying Vulkan objects.
            self.device.device_wait_idle().unwrap();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}
