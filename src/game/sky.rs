use std::f32::consts::PI;

use glam::Vec3;

use crate::gpu::buffer::Vertex;

const RINGS: u32 = 64;
const SEGMENTS: u32 = 64;
const RADIUS: f32 = 10.0;

pub fn sky_dome_geometry() -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for ring in 0..=RINGS {
        let phi = -PI * 0.5 + (ring as f32 / RINGS as f32) * PI;
        let (sin_phi, cos_phi) = phi.sin_cos();

        for seg in 0..=SEGMENTS {
            let theta = (seg as f32 / SEGMENTS as f32) * 2.0 * PI;
            let (sin_theta, cos_theta) = theta.sin_cos();

            let x = cos_phi * cos_theta;
            let y = sin_phi;
            let z = cos_phi * sin_theta;

            let position = [x * RADIUS, y * RADIUS, z * RADIUS];
            let normal = [x, y, z];

            vertices.push(Vertex {
                position,
                normal,
                color: [1.0, 1.0, 1.0],
                uv:    [0.0, 0.0],
            });
        }
    }

    let stride = SEGMENTS + 1;
    for ring in 0..RINGS {
        for seg in 0..SEGMENTS {
            let tl = ring * stride + seg;
            let tr = tl + 1;
            let bl = tl + stride;
            let br = bl + 1;

            indices.push(tl);
            indices.push(tr);
            indices.push(bl);

            indices.push(tr);
            indices.push(br);
            indices.push(bl);
        }
    }

    (vertices, indices)
}

pub fn sun_direction(time_of_day: f32) -> Vec3 {
    let hour_angle = (time_of_day - 12.0) / 24.0 * 2.0 * PI;

    let elevation = -hour_angle.sin();
    let azimuth = -hour_angle.cos();

    Vec3::new(azimuth, elevation, 0.3).normalize()
}
