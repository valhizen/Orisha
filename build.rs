use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let shader_dir = Path::new("shaders");
    let out_dir = shader_dir.join("compiled");
    fs::create_dir_all(&out_dir).unwrap();

    let shaders = [
        ("basic.vert", "basic_vert.spv"),
        ("basic.frag", "basic_frag.spv"),
    ];

    for (src, dst) in &shaders {
        let src_path = shader_dir.join(src);
        let dst_path = out_dir.join(dst);

        println!("cargo:rerun-if-changed={}", src_path.display());

        let output = Command::new("glslc")
            .arg(src_path.to_str().unwrap())
            .arg("-o")
            .arg(dst_path.to_str().unwrap())
            .output()
            .expect("Failed to execute glslc. Is the Vulkan SDK installed?");

        if !output.status.success() {
            panic!(
                "Shader compilation failed for {}:\n{}",
                src,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
