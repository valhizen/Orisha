use std::path::Path;

use glam::{Mat4, Vec3, Vec4};

use crate::gpu::buffer::Vertex;

pub fn load_glb(path: impl AsRef<Path>) -> (Vec<Vertex>, Vec<u32>, f32) {
    let (document, buffers, _images) =
        gltf::import(path).expect("Failed to load GLB file");

    let mut all_vertices: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    let mut min_y: f32 = f32::MAX;

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

    let y_offset = if min_y.is_finite() { -min_y } else { 0.0 };
    (all_vertices, all_indices, y_offset)
}

fn collect_node(
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    parent_transform: Mat4,
    all_vertices: &mut Vec<Vertex>,
    all_indices: &mut Vec<u32>,
    min_y: &mut f32,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_transform * local;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .expect("GLB primitive missing positions")
                .collect();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            let base_color = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_factor();
            let color = [base_color[0], base_color[1], base_color[2]];

            let normal_mat = world.inverse().transpose();

            let base_vertex = all_vertices.len() as u32;

            for (pos, norm) in positions.iter().zip(normals.iter()) {
                let wp = world * Vec4::new(pos[0], pos[1], pos[2], 1.0);
                let wn = normal_mat * Vec4::new(norm[0], norm[1], norm[2], 0.0);
                let wn3 = Vec3::new(wn.x, wn.y, wn.z).normalize_or_zero();

                if wp.y < *min_y {
                    *min_y = wp.y;
                }

                all_vertices.push(Vertex {
                    position: [wp.x, wp.y, wp.z],
                    normal: wn3.into(),
                    color,
                });
            }

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

    for child in node.children() {
        collect_node(&child, buffers, world, all_vertices, all_indices, min_y);
    }
}
