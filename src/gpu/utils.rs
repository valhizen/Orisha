use ash::vk;
use std::error::Error;

use super::device::Device;

pub fn load_shader(device: &Device, path: &str) -> Result<vk::ShaderModule, Box<dyn Error>> {
        let bytes = std::fs::read(path)?;
        let (prefix, code, suffix) = unsafe { bytes.align_to::<u32>() };
        assert!(
            prefix.is_empty() && suffix.is_empty(),
            "SPIR-V data is misaligned"
        );

        let info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe { Ok(device.device.create_shader_module(&info, None)?) }
    }
