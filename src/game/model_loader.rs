use std::path::Path;

use glam::{Mat4, Vec3, Vec4};

use crate::gpu::buffer::Vertex;

/// Loads a GLB file as a single static mesh.
///
/// This loader walks through all scenes and nodes, applies node transforms,
/// and merges everything into one vertex/index list.
///
/// Returns:
/// - all vertices
/// - all indices
/// - a Y offset that can be used to place the model on the ground
pub fn load_glb(path: impl AsRef<Path>) -> (Vec<Vertex>, Vec<u32>, f32) {
    // Load the glTF document and its binary buffer data.
    let (document, buffers, _images) =
        gltf::import(path).expect("Failed to load GLB file");

    // Final merged mesh data stored on the CPU before GPU upload.
    let mut all_vertices: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    // Track the lowest point of the model so we can compute a ground offset.
    let mut min_y: f32 = f32::MAX;

    // Walk every scene root and recursively collect mesh data.
    for scene in document.scenes() {
        for node in scene.nodes() {
            collect_node(
                &node,
                &buffers,
                Mat4::IDENTITY,
                &mut all_vertices,
                &mut all_indices,
                &mut min_y,
            );
        }
    }

    // If the model's lowest point is below Y=0, this moves it upward so its
    // feet/base can sit on the ground more easily.
    let y_offset = if min_y.is_finite() { -min_y } else { 0.0 };
    (all_vertices, all_indices, y_offset)
}

/// Recursively collects mesh data from one node and its children.
fn collect_node(
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    parent_transform: Mat4,
    all_vertices: &mut Vec<Vertex>,
    all_indices: &mut Vec<u32>,
    min_y: &mut f32,
) {
    // glTF stores a local transform per node.
    // Multiplying by the parent transform gives this node's world transform.
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_transform * local;

    // Some nodes contain a mesh. A mesh may contain multiple primitives.
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

            // Positions are required for rendering.
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .expect("GLB primitive missing positions")
                .collect();

            // If normals are missing, fall back to simple upward normals.
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            // Use the material's base color as a simple vertex color.
            let base_color = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_factor();
            let color = [base_color[0], base_color[1], base_color[2]];

            // Normal vectors must be transformed differently from positions.
            let normal_mat = world.inverse().transpose();

            // Remember where this primitive starts in the merged vertex list.
            let base_vertex = all_vertices.len() as u32;

            for (pos, norm) in positions.iter().zip(normals.iter()) {
                // Transform position into world space.
                let wp = world * Vec4::new(pos[0], pos[1], pos[2], 1.0);

                // Transform normal into world space.
                // w = 0 means "direction", so translation is ignored.
                let wn = normal_mat * Vec4::new(norm[0], norm[1], norm[2], 0.0);
                let wn3 = Vec3::new(wn.x, wn.y, wn.z).normalize_or_zero();

                // Track the lowest point of the model.
                if wp.y < *min_y {
                    *min_y = wp.y;
                }

                all_vertices.push(Vertex {
                    position: [wp.x, wp.y, wp.z],
                    normal: wn3.into(),
                    color,
                    // UVs are not loaded yet in this simple loader.
                    uv: [0.0, 0.0],
                });
            }

            // If the primitive has indices, reuse them.
            // Otherwise build a simple sequential index list.
            if let Some(indices) = reader.read_indices() {
                for idx in indices.into_u32() {
                    all_indices.push(base_vertex + idx);
                }
            } else {
                for i in 0..positions.len() as u32 {
                    all_indices.push(base_vertex + i);
                }
            }
        }
    }

    // Recurse into child nodes so the full node tree is collected.
    for child in node.children() {
        collect_node(&child, buffers, world, all_vertices, all_indices, min_y);
    }
}
