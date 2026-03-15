use std::fs;
use std::path::Path;
use std::process::Command;

// This build script runs before the main Rust code is compiled.
// Its job is to compile GLSL shader files into SPIR-V binaries.
fn main() {
    // Source shaders live in `shaders/`, and compiled output goes to `shaders/compiled/`.
    let shader_dir = Path::new("shaders");
    let out_dir = shader_dir.join("compiled");
    fs::create_dir_all(&out_dir).unwrap();

    // Each pair is: (input GLSL file, output SPIR-V file).
    let shaders = [
        ("basic.vert", "basic_vert.spv"),
        ("basic.frag", "basic_frag.spv"),
        ("imgui.vert", "imgui_vert.spv"),
        ("imgui.frag", "imgui_frag.spv"),
        ("sky.vert",   "sky_vert.spv"),
        ("sky.frag",   "sky_frag.spv"),
    ];

    for (src, dst) in &shaders {
        let src_path = shader_dir.join(src);
        let dst_path = out_dir.join(dst);

        // Tell Cargo to rerun this build script if a shader source file changes.
        println!("cargo:rerun-if-changed={}", src_path.display());

        // Run `glslc` to compile the GLSL shader into SPIR-V.
        let output = Command::new("glslc")
            .arg(src_path.to_str().unwrap())
            .arg("-o")
            .arg(dst_path.to_str().unwrap())
            .output()
            .expect("Failed to execute glslc. Is the Vulkan SDK installed?");

        // If compilation fails, stop the build and print the shader compiler error.
        if !output.status.success() {
            panic!(
                "Shader compilation failed for {}:\n{}",
                src,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
