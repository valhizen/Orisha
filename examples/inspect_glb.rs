use glam::{Mat4, Quat, Vec3, Vec4};

fn main() {
    // This is a small debugging/example program for looking inside a GLB file.
    // It does not render anything. It just prints useful information to the console.
    let path = "character/characte2.glb";
    let (document, buffers, _images) = gltf::import(path).expect("Failed to load GLB");

    println!("=== GLB Structure: {path} ===\n");

    // Print the scenes and their root nodes.
    // A GLB can contain more than one scene.
    for scene in document.scenes() {
        println!("Scene #{}: {:?}", scene.index(), scene.name());
        for node in scene.nodes() {
            print_node(&node, &buffers, 1);
        }
    }

    // Print every mesh in the file, even if it is not currently used by a scene node.
    println!("\n=== All Meshes ===");
    for mesh in document.meshes() {
        println!("  Mesh #{}: {:?}", mesh.index(), mesh.name());
        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
            let vert_count = reader.read_positions().map(|p| p.count()).unwrap_or(0);
            let idx_count = reader.read_indices().map(|i| i.into_u32().count()).unwrap_or(0);
            let has_normals = prim.get(&gltf::Semantic::Normals).is_some();
            let has_joints = prim.get(&gltf::Semantic::Joints(0)).is_some();
            let has_weights = prim.get(&gltf::Semantic::Weights(0)).is_some();
            let mat = prim.material();
            let base_color = mat.pbr_metallic_roughness().base_color_factor();
            println!("    Primitive: {vert_count} verts, {idx_count} indices, normals={has_normals}, joints={has_joints}, weights={has_weights}");
            println!("      Material: {:?}, base_color={base_color:?}, double_sided={}", mat.name(), mat.double_sided());
        }
    }

    // Print skin data.
    // A skin usually means the model is meant for skeletal animation.
    println!("\n=== Skins ===");
    for skin in document.skins() {
        println!("  Skin #{}: {:?}, {} joints", skin.index(), skin.name(), skin.joints().count());
    }

    // Print all nodes with their transform information.
    // This helps you understand the GLB hierarchy.
    println!("\n=== Node Tree ===");
    for node in document.nodes() {
        let (t, r, s) = node.transform().decomposed();
        let mesh_info = node.mesh().map(|m| format!("Mesh #{}", m.index())).unwrap_or_default();
        let skin_info = node.skin().map(|s| format!("Skin #{}", s.index())).unwrap_or_default();
        let children: Vec<_> = node.children().map(|c| c.index()).collect();
        if t != [0.0, 0.0, 0.0] || r != [0.0, 0.0, 0.0, 1.0] || s != [1.0, 1.0, 1.0] || !mesh_info.is_empty() {
            println!("  Node #{} {:?}: T={t:?} R={r:?} S={s:?} {mesh_info} {skin_info} children={children:?}",
                node.index(), node.name());
        }
    }

    // Print all animations and how many channels each one contains.
    // A channel usually animates translation, rotation, or scale for one node.
    println!("\n=== Animations ===");
    for anim in document.animations() {
        println!("  Animation #{}: {:?}, {} channels", anim.index(), anim.name(), anim.channels().count());
    }
}

fn print_node(node: &gltf::Node, buffers: &[gltf::buffer::Data], depth: usize) {
    // Indentation makes the parent/child hierarchy easier to read in the console.
    let indent = "  ".repeat(depth);
    let (t, r, s) = node.transform().decomposed();
    let mesh_info = node.mesh().map(|m| format!(" → Mesh #{}", m.index())).unwrap_or_default();
    let skin_info = node.skin().map(|s| format!(" → Skin #{}", s.index())).unwrap_or_default();
    println!("{indent}Node #{} {:?}{mesh_info}{skin_info}", node.index(), node.name());

    // Only print transform data if it is not the default identity transform.
    if t != [0.0, 0.0, 0.0] || r != [0.0, 0.0, 0.0, 1.0] || s != [1.0, 1.0, 1.0] {
        println!("{indent}  T={t:?} R={r:?} S={s:?}");
    }

    // Recurse into child nodes so the whole hierarchy is shown.
    for child in node.children() {
        print_node(&child, buffers, depth + 1);
    }
}
