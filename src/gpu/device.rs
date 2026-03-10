use std::{borrow::Cow, error::Error, ffi, os::raw::c_char};

use ash::{ext::debug_utils, khr::surface, vk, Device as AshDevice, Entry, Instance};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

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

/// Score a physical device for selection. Returns -1 if unsuitable.
fn score_device(
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> i32 {
    unsafe {
        let props = instance.get_physical_device_properties(pdevice);

        if props.api_version < vk::API_VERSION_1_3 {
            return -1;
        }

        let mut features_13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features_13);
        instance.get_physical_device_features2(pdevice, &mut features2);

        if features_13.dynamic_rendering == vk::FALSE
            || features_13.synchronization2 == vk::FALSE
        {
            return -1;
        }

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

        if !has_queue { return -1; }

        let formats = surface_loader
            .get_physical_device_surface_formats(pdevice, surface)
            .unwrap_or_default();
        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(pdevice, surface)
            .unwrap_or_default();

        if formats.is_empty() || present_modes.is_empty() {
            return -1;
        }

        // On hybrid GPU laptops, only prefer discrete if explicitly requested
        let prefer_discrete = std::env::var("DRI_PRIME").is_ok_and(|v| v == "1")
            || std::env::var("ORISHA_GPU").is_ok_and(|v| v == "discrete");

        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => {
                if prefer_discrete { 1000 } else { 50 }
            }
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            _ => 1,
        }
    }
}

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
    pub fn new(window: &winit::window::Window) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = Entry::linked();
            let app_name = c"Orisha";

            let layer_names = [c"VK_LAYER_KHRONOS_validation"];
            let layers_raw: Vec<*const c_char> =
                layer_names.iter().map(|n| n.as_ptr()).collect();

            let mut extension_names =
                ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())
                    .unwrap()
                    .to_vec();
            extension_names.push(debug_utils::NAME.as_ptr());

            let app_info = vk::ApplicationInfo::default()
                .application_name(app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(c"Orisha Engine")
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_3);

            let instance_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers_raw)
                .enabled_extension_names(&extension_names);

            let instance = entry.create_instance(&instance_info, None)?;

            // Debug messenger (errors + warnings only)
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
                .create_debug_utils_messenger(&debug_info, None)?;

            // Surface
            let surface_loader = surface::Instance::new(&entry, &instance);
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?;

            // Pick best GPU
            let pdevices = instance.enumerate_physical_devices()?;

            println!("── Available GPUs ──");
            for &pd in &pdevices {
                let props = instance.get_physical_device_properties(pd);
                let name = ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
                let score = score_device(&instance, pd, &surface_loader, surface);
                println!("  [{score:4}] {name} ({:?})", props.device_type);
            }

            let pdevice = pdevices
                .iter()
                .copied()
                .max_by_key(|&pd| score_device(&instance, pd, &surface_loader, surface))
                .filter(|&pd| score_device(&instance, pd, &surface_loader, surface) > 0)
                .expect("No suitable GPU found");

            let sel = instance.get_physical_device_properties(pdevice);
            let name = ffi::CStr::from_ptr(sel.device_name.as_ptr()).to_string_lossy();
            println!("Selected GPU: {name}\n");

            // Queue family (graphics + present)
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

            // Logical device with Vulkan 1.3 features
            let priorities = [1.0f32];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);

            let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

            let mut features_13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);

            let device_info = vk::DeviceCreateInfo::default()
                .push_next(&mut features_13)
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extensions);

            let device = instance.create_device(pdevice, &device_info, None)?;
            let present_queue = device.get_device_queue(queue_family_index, 0);
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
            self.device.device_wait_idle().unwrap();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}
