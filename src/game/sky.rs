use std::f32::consts::PI;

use crate::gpu::buffer::Vertex;

const RINGS: u32 = 40;    // horizontal segments
const SEGMENTS: u32 = 40; // vertical segments
const RADIUS: f32 = 900.0;

// Inverted hemisphere dome with sky gradient colors.
pub fn sky_dome_geometry() -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for ring in 0..=RINGS {
        let phi = (ring as f32 / RINGS as f32) * (PI * 0.5);
        let (sin_phi, cos_phi) = phi.sin_cos();

        for seg in 0..=SEGMENTS {
            let theta = (seg as f32 / SEGMENTS as f32) * 2.0 * PI;
            let (sin_theta, cos_theta) = theta.sin_cos();

            let x = cos_phi * cos_theta;
            let y = sin_phi;
            let z = cos_phi * sin_theta;

            let position = [x * RADIUS, y * RADIUS, z * RADIUS];

            let normal = [-x, -y, -z];

            let t = sin_phi;
            let color = sky_gradient(t);

            vertices.push(Vertex {
                position,
                normal,
                color,
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

// Horizon-to-zenith color gradient.
fn sky_gradient(t: f32) -> [f32; 3] {
    let horizon = [0.65, 0.78, 0.92];
    let zenith = [0.18, 0.32, 0.72];
    let t = (t * t).clamp(0.0, 1.0);

    [
        horizon[0] + (zenith[0] - horizon[0]) * t,
        horizon[1] + (zenith[1] - horizon[1]) * t,
        horizon[2] + (zenith[2] - horizon[2]) * t,
    ]
}
