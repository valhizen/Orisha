use std::path::Path;

use glam::{Mat4, Quat, Vec3};

use crate::gpu::buffer::{Vertex, MAX_JOINTS};

// This file contains helper code for skeletal animation.
// Important: in the current project, this module looks like a work-in-progress
// and may not be fully wired into the active renderer yet.

// ── Data structures ──────────────────────────────────────────────────────────

/// Parsed skeleton from a glTF skin.
pub struct Skeleton {
    pub joint_count: usize,
    /// Inverse bind matrix per joint.
    /// This moves a vertex from mesh bind space into joint space.
    pub inverse_bind: Vec<Mat4>,
    /// Parent joint index (None for root joints).
    /// This describes the skeleton hierarchy.
    pub parents: Vec<Option<usize>>,
    /// Bind-pose local T/R/S per joint.
    /// These are the default local transforms before animation is applied.
    pub bind_translation: Vec<Vec3>,
    pub bind_rotation: Vec<Quat>,
    pub bind_scale: Vec<Vec3>,
}

/// A single animation clip (e.g. "run").
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

pub struct Channel {
    /// Index into the skeleton's joint array.
    pub joint: usize,
    /// Which part of the joint transform this channel animates.
    pub property: Property,
    /// Keyframe times in seconds.
    pub timestamps: Vec<f32>,
    /// Keyframe values matching the timestamps above.
    pub values: ChannelValues,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Property {
    Translation,
    Rotation,
    Scale,
}

pub enum ChannelValues {
    Vec3s(Vec<Vec3>),
    Quats(Vec<Quat>),
}

/// Runtime animation playback state.
pub struct AnimationState {
    pub clip_index: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
}

impl AnimationState {
    /// Create default playback state.
    pub fn new() -> Self {
        Self {
            clip_index: 0,
            time: 0.0,
            speed: 1.0,
            looping: true,
        }
    }

    /// Move animation time forward by `dt`.
    pub fn advance(&mut self, dt: f32, clips: &[AnimationClip]) {
        if clips.is_empty() {
            return;
        }
        let clip = &clips[self.clip_index.min(clips.len() - 1)];
        self.time += dt * self.speed;
        if self.looping && clip.duration > 0.0 {
            // Wrap back to the start for looping clips.
            self.time %= clip.duration;
        } else {
            // Clamp at the end for non-looping clips.
            self.time = self.time.min(clip.duration);
        }
    }
}

// ── Sampling ─────────────────────────────────────────────────────────────────

/// Sample animation at `state.time` and return `joint_count` final matrices
/// ready to upload as the bone SSBO.
pub fn sample_animation(
    skeleton: &Skeleton,
    clips: &[AnimationClip],
    state: &AnimationState,
) -> Vec<Mat4> {
    let n = skeleton.joint_count;

    // Start from the bind pose, then replace only the animated parts.
    let mut local_t: Vec<Vec3> = skeleton.bind_translation.clone();
    let mut local_r: Vec<Quat> = skeleton.bind_rotation.clone();
    let mut local_s: Vec<Vec3> = skeleton.bind_scale.clone();

    // Sample each animation channel at the current playback time.
    if !clips.is_empty() {
        let clip = &clips[state.clip_index.min(clips.len() - 1)];
        let t = state.time;
        for channel in &clip.channels {
            let j = channel.joint;
            if j >= n {
                continue;
            }
            match (&channel.property, &channel.values) {
                (Property::Translation, ChannelValues::Vec3s(vals)) => {
                    local_t[j] = sample_vec3(&channel.timestamps, vals, t);
                }
                (Property::Rotation, ChannelValues::Quats(vals)) => {
                    local_r[j] = sample_quat(&channel.timestamps, vals, t);
                }
                (Property::Scale, ChannelValues::Vec3s(vals)) => {
                    local_s[j] = sample_vec3(&channel.timestamps, vals, t);
                }
                _ => {}
            }
        }
    }

    // Convert local joint transforms into world-space joint transforms
    // by walking from parents to children.
    let mut world = vec![Mat4::IDENTITY; n];
    for j in 0..n {
        let local = Mat4::from_scale_rotation_translation(local_s[j], local_r[j], local_t[j]);
        world[j] = match skeleton.parents[j] {
            Some(p) => world[p] * local,
            None => local,
        };
    }

    // Final skinning matrix:
    // current joint world transform * inverse bind matrix.
    let mut result = vec![Mat4::IDENTITY; n];
    for j in 0..n {
        result[j] = world[j] * skeleton.inverse_bind[j];
    }

    result
}

/// Fill a `[[[f32;4];4]; MAX_JOINTS]` array from joint matrices.
pub fn fill_bone_array(joint_matrices: &[Mat4]) -> [[[f32; 4]; 4]; MAX_JOINTS] {
    let id = Mat4::IDENTITY.to_cols_array_2d();
    let mut arr = [id; MAX_JOINTS];
    for (i, m) in joint_matrices.iter().enumerate() {
        if i >= MAX_JOINTS {
            break;
        }
        arr[i] = m.to_cols_array_2d();
    }
    // Any unused slots stay as identity matrices.
    arr
}

// ── Interpolation helpers ────────────────────────────────────────────────────

fn find_keyframe(timestamps: &[f32], t: f32) -> (usize, usize, f32) {
    if timestamps.is_empty() {
        return (0, 0, 0.0);
    }
    if t <= timestamps[0] {
        return (0, 0, 0.0);
    }
    let last = timestamps.len() - 1;
    if t >= timestamps[last] {
        return (last, last, 0.0);
    }
    // Binary search for the two keyframes around time `t`.
    let mut lo = 0;
    let mut hi = last;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if timestamps[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = timestamps[hi] - timestamps[lo];
    let factor = if span > 1e-9 {
        // Interpolation factor between keyframe `lo` and `hi`.
        (t - timestamps[lo]) / span
    } else {
        0.0
    };
    (lo, hi, factor)
}

fn sample_vec3(timestamps: &[f32], values: &[Vec3], t: f32) -> Vec3 {
    if values.len() == 1 {
        return values[0];
    }
    let (a, b, f) = find_keyframe(timestamps, t);
    // Linear interpolation for translation and scale.
    values[a].lerp(values[b], f)
}

fn sample_quat(timestamps: &[f32], values: &[Quat], t: f32) -> Quat {
    if values.len() == 1 {
        return values[0];
    }
    let (a, b, f) = find_keyframe(timestamps, t);
    // Spherical interpolation gives smoother rotation blending.
    values[a].slerp(values[b], f)
}

// ── glTF loading ─────────────────────────────────────────────────────────────

/// Full result of loading a skinned character from a glTF file.
pub struct SkinnedModel {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub skeleton: Skeleton,
    pub clips: Vec<AnimationClip>,
    pub y_offset: f32,
}

/// Load a skinned & animated character from a GLB file.
pub fn load_skinned_glb(path: impl AsRef<Path>) -> SkinnedModel {
    let (document, buffers, _images) =
        gltf::import(path).expect("Failed to load GLB file");

    // Read the first skin from the file.
    // A skin contains the list of joints used for skeletal animation.
    let skin = document
        .skins()
        .next()
        .expect("GLB has no skin");

    let joint_nodes: Vec<usize> = skin.joints().map(|n| n.index()).collect();
    let joint_count = joint_nodes.len();

    // Map glTF node indices to compact joint indices used by this runtime.
    let mut node_to_joint = std::collections::HashMap::new();
    for (ji, &ni) in joint_nodes.iter().enumerate() {
        node_to_joint.insert(ni, ji);
    }

    // Read inverse bind matrices.
    // If the file does not provide them, fall back to identity matrices.
    let reader = skin.reader(|buf| Some(&buffers[buf.index()]));
    let inverse_bind: Vec<Mat4> = reader
        .read_inverse_bind_matrices()
        .map(|iter| iter.map(|m| Mat4::from_cols_array_2d(&m)).collect())
        .unwrap_or_else(|| vec![Mat4::IDENTITY; joint_count]);

    // Build skeleton hierarchy and bind-pose transforms from the glTF node tree.
    let mut parents = vec![None; joint_count];
    let mut bind_t = vec![Vec3::ZERO; joint_count];
    let mut bind_r = vec![Quat::IDENTITY; joint_count];
    let mut bind_s = vec![Vec3::ONE; joint_count];

    for node in document.nodes() {
        if let Some(ji) = node_to_joint.get(&node.index()) {
            let (t, r, s) = node.transform().decomposed();
            bind_t[*ji] = Vec3::from(t);
            bind_r[*ji] = Quat::from_array(r);
            bind_s[*ji] = Vec3::from(s);

            // Find this node's parent joint by scanning the node tree.
            for candidate in document.nodes() {
                for child in candidate.children() {
                    if child.index() == node.index() {
                        if let Some(parent_ji) = node_to_joint.get(&candidate.index()) {
                            parents[*ji] = Some(*parent_ji);
                        }
                        break;
                    }
                }
            }
        }
    }

    let skeleton = Skeleton {
        joint_count,
        inverse_bind,
        parents,
        bind_translation: bind_t,
        bind_rotation: bind_r,
        bind_scale: bind_s,
    };

    // Read every animation clip from the file.
    let mut clips = Vec::new();
    for anim in document.animations() {
        let mut duration: f32 = 0.0;
        let mut channels = Vec::new();

        for chan in anim.channels() {
            let target_node = chan.target().node().index();
            // Skip channels that do not target a joint in this skin.
            let joint = match node_to_joint.get(&target_node) {
                Some(&j) => j,
                None => continue,
            };

            let sampler = chan.sampler();
            let reader = chan.reader(|buf| Some(&buffers[buf.index()]));

            let timestamps: Vec<f32> = reader
                .read_inputs()
                .expect("animation missing timestamps")
                .collect();

            if let Some(&last) = timestamps.last() {
                duration = duration.max(last);
            }

            let property;
            let values;
            match chan.target().property() {
                gltf::animation::Property::Translation => {
                    property = Property::Translation;
                    let v: Vec<Vec3> = reader
                        .read_outputs()
                        .expect("missing outputs")
                        .into_f32()
                        .map(|xyz| Vec3::from(xyz))
                        .collect();
                    values = ChannelValues::Vec3s(v);
                }
                gltf::animation::Property::Rotation => {
                    property = Property::Rotation;
                    let v: Vec<Quat> = reader
                        .read_outputs()
                        .expect("missing outputs")
                        .into_f32()
                        .map(|xyzw| Quat::from_array(xyzw))
                        .collect();
                    values = ChannelValues::Quats(v);
                }
                gltf::animation::Property::Scale => {
                    property = Property::Scale;
                    let v: Vec<Vec3> = reader
                        .read_outputs()
                        .expect("missing outputs")
                        .into_f32()
                        .map(|xyz| Vec3::from(xyz))
                        .collect();
                    values = ChannelValues::Vec3s(v);
                }
                _ => continue, // Skip morph targets for now.
            }

            channels.push(Channel {
                joint,
                property,
                timestamps,
                values,
            });
        }

        clips.push(AnimationClip {
            name: anim.name().unwrap_or("unnamed").to_string(),
            duration,
            channels,
        });
    }

    // Load the mesh data itself, including joint indices and weights.
    let mut all_vertices: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    let mut min_y: f32 = f32::MAX;

    for node in document.nodes() {
        if let (Some(mesh), Some(_skin)) = (node.mesh(), node.skin()) {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .expect("missing positions")
                    .collect();

                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|i| i.collect())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

                // Each vertex can reference up to 4 joints with 4 weights.
                let joints: Vec<[u16; 4]> = reader
                    .read_joints(0)
                    .map(|i| i.into_u16().collect())
                    .unwrap_or_else(|| vec![[0; 4]; positions.len()]);

                let weights: Vec<[f32; 4]> = reader
                    .read_weights(0)
                    .map(|i| i.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0; 4]; positions.len()]);

                let base_color = primitive
                    .material()
                    .pbr_metallic_roughness()
                    .base_color_factor();
                let color = [base_color[0], base_color[1], base_color[2]];

                let base_vertex = all_vertices.len() as u32;

                for i in 0..positions.len() {
                    let pos = positions[i];
                    if pos[1] < min_y {
                        min_y = pos[1];
                    }

                    all_vertices.push(Vertex {
                        position: pos,
                        normal: normals[i],
                        color,
                        uv: [0.0, 0.0],
                        // Store skinning data per vertex so the shader can
                        // blend multiple joint transforms later.
                        joint_indices: [
                            joints[i][0] as u32,
                            joints[i][1] as u32,
                            joints[i][2] as u32,
                            joints[i][3] as u32,
                        ],
                        joint_weights: weights[i],
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
    }

    // Offset the model upward so its lowest point can sit on the ground.
    let y_offset = if min_y.is_finite() { -min_y } else { 0.0 };

    SkinnedModel {
        vertices: all_vertices,
        indices: all_indices,
        skeleton,
        clips,
        y_offset,
    }
}
