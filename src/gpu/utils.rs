use ash::vk;
use std::error::Error;

use super::device::Device;

/// Loads a compiled SPIR-V shader from disk and creates a Vulkan shader module.
///
/// This is a shared helper so multiple pipeline files do not need to repeat
/// the same file-loading and shader-module creation code.
pub fn load_shader(device: &Device, path: &str) -> Result<vk::ShaderModule, Box<dyn Error>> {
        // Read the compiled `.spv` file into memory.
        let bytes = std::fs::read(path)?;

        // Vulkan expects shader code as `u32` words, not raw bytes.
        let (prefix, code, suffix) = unsafe { bytes.align_to::<u32>() };
        assert!(
            prefix.is_empty() && suffix.is_empty(),
            "SPIR-V data is misaligned"
        );

        // Describe the shader bytecode to Vulkan, then create the shader module.
        let info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe { Ok(device.device.create_shader_module(&info, None)?) }
    }
