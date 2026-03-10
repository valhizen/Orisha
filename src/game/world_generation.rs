use glam::Vec3;
use noise::{NoiseFn, Perlin};

use crate::gpu::buffer::Vertex;

// Procedural terrain using layered noise.
pub struct Terrain {
    pub amplitude: f32,
    continent:  Perlin,  // large-scale: plains vs mountains
    erosion:    Perlin,  // medium: roughness control
    detail:     Perlin,  // small: bumps
    ridge:      Perlin,  // ridge noise for mountain chains
    warp:       Perlin,  // domain warping for organic shapes
    biome:      Perlin,  // biome selector: plains / hills / mountains
}

impl Terrain {

    pub fn new(amplitude: f32, seed: u32) -> Self {
        Self {
            amplitude,
            continent: Perlin::new(seed),
            erosion:   Perlin::new(seed.wrapping_add(1)),
            detail:    Perlin::new(seed.wrapping_add(2)),
            ridge:     Perlin::new(seed.wrapping_add(3)),
            warp:      Perlin::new(seed.wrapping_add(4)),
            biome:     Perlin::new(seed.wrapping_add(5)),
        }
    }

    // Deterministic height at world-space (x, z).
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let xd = x as f64;
        let zd = z as f64;

        let warp_strength = 60.0;
        let wx = xd + fbm(&self.warp, xd, zd, 0.002, 2, 2.0, 0.5) * warp_strength;
        let wz = zd + fbm(&self.warp, xd + 1000.0, zd + 1000.0, 0.002, 2, 2.0, 0.5) * warp_strength;

        let biome_raw = fbm(&self.biome, wx, wz, 0.0008, 2, 2.0, 0.5);
        let biome_t = ((biome_raw * 1.6).clamp(-1.0, 1.0) * 0.5 + 0.5) as f64;

        let continent_raw = fbm(&self.continent, wx, wz, 0.0015, 3, 2.0, 0.5);
        let continent = continent_raw * lerp(0.05, 1.0, biome_t);

        let ridge = ridged_fbm(&self.ridge, wx, wz, 0.003, 3, 2.0, 0.5);
        let ridge_contribution = ridge * biome_t * 0.8;

        let erosion_raw = fbm(&self.erosion, wx, wz, 0.006, 2, 2.0, 0.5);
        let erosion = (erosion_raw * 0.5 + 0.5).clamp(0.0, 1.0);

        let detail_raw = fbm(&self.detail, wx, wz, 0.03, 2, 2.5, 0.45);
        let detail_strength = lerp(0.05, 1.0, biome_t) * (1.0 - erosion * 0.7);
        let detail = detail_raw * detail_strength * 0.25;

        let raw_height = continent + ridge_contribution + detail;
        let plateau_strength = (biome_t * 2.0 - 0.5).clamp(0.0, 1.0);
        let steps = 8.0;
        let quantised = (raw_height * steps).round() / steps;
        let final_height = raw_height + (quantised - raw_height) * plateau_strength * 0.35;

        final_height as f32 * self.amplitude
    }

    // Surface normal via central differences.
    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        let e = 0.2;
        let h_left  = self.height_at(x - e, z);
        let h_right = self.height_at(x + e, z);
        let h_down  = self.height_at(x, z - e);
        let h_up    = self.height_at(x, z + e);
        Vec3::new(h_left - h_right, 2.0 * e, h_down - h_up).normalize_or(Vec3::Y)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// Fractal Brownian motion.
fn fbm(
    noise: &Perlin, x: f64, z: f64,
    base_freq: f64, octaves: u32,
    lacunarity: f64, persistence: f64,
) -> f64 {
    let mut value = 0.0;
    let mut freq = base_freq;
    let mut amp = 1.0;
    let mut max_amp = 0.0;

    for _ in 0..octaves {
        value += noise.get([x * freq, z * freq]) * amp;
        max_amp += amp;
        freq *= lacunarity;
        amp *= persistence;
    }
    value / max_amp
}

// Ridged fBm for sharp mountain peaks.
fn ridged_fbm(
    noise: &Perlin, x: f64, z: f64,
    base_freq: f64, octaves: u32,
    lacunarity: f64, persistence: f64,
) -> f64 {
    let mut value = 0.0;
    let mut freq = base_freq;
    let mut amp = 1.0;
    let mut max_amp = 0.0;

    for _ in 0..octaves {
        let s = noise.get([x * freq, z * freq]);
        value += (1.0 - s.abs()) * amp;
        max_amp += amp;
        freq *= lacunarity;
        amp *= persistence;
    }
    value / max_amp
}

pub const CHUNK_SIZE: u32 = 64;
pub const CHUNK_SCALE: f32 = 2.0;
pub const CHUNK_WORLD_SIZE: f32 = CHUNK_SIZE as f32 * CHUNK_SCALE;

// Integer chunk coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn world_origin(&self) -> (f32, f32) {
        (self.x as f32 * CHUNK_WORLD_SIZE, self.z as f32 * CHUNK_WORLD_SIZE)
    }

    pub fn from_world(wx: f32, wz: f32) -> Self {
        Self {
            x: (wx / CHUNK_WORLD_SIZE).floor() as i32,
            z: (wz / CHUNK_WORLD_SIZE).floor() as i32,
        }
    }
}

// Generate vertex + index data for one terrain chunk.
pub fn build_chunk_mesh(terrain: &Terrain, coord: ChunkCoord) -> (Vec<Vertex>, Vec<u32>) {
    let vps = CHUNK_SIZE + 1; // vertices per side
    let vert_count = (vps * vps) as usize;
    let idx_count = (CHUNK_SIZE * CHUNK_SIZE * 6) as usize;

    let mut vertices = Vec::with_capacity(vert_count);
    let mut indices = Vec::with_capacity(idx_count);

    let (ox, oz) = coord.world_origin();

    for z in 0..vps {
        for x in 0..vps {
            let wx = ox + x as f32 * CHUNK_SCALE;
            let wz = oz + z as f32 * CHUNK_SCALE;
            let wy = terrain.height_at(wx, wz);

            vertices.push(Vertex {
                position: [wx, wy, wz],
                normal: [0.0, 1.0, 0.0],
                color: terrain_color(wy, terrain.amplitude),
            });
        }
    }

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let tl = z * vps + x;
            let tr = tl + 1;
            let bl = tl + vps;
            let br = bl + 1;

            indices.push(tl);
            indices.push(bl);
            indices.push(tr);

            indices.push(tr);
            indices.push(bl);
            indices.push(br);
        }
    }

    compute_smooth_normals(&mut vertices, &indices);
    (vertices, indices)
}

fn compute_smooth_normals(vertices: &mut [Vertex], indices: &[u32]) {
    for v in vertices.iter_mut() {
        v.normal = [0.0, 0.0, 0.0];
    }
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let v0 = Vec3::from(vertices[i0].position);
        let v1 = Vec3::from(vertices[i1].position);
        let v2 = Vec3::from(vertices[i2].position);
        let n = (v1 - v0).cross(v2 - v0);
        for &i in &[i0, i1, i2] {
            vertices[i].normal[0] += n.x;
            vertices[i].normal[1] += n.y;
            vertices[i].normal[2] += n.z;
        }
    }
    for v in vertices.iter_mut() {
        let n = Vec3::from(v.normal).normalize_or(Vec3::Y);
        v.normal = n.into();
    }
}

// Height-based terrain color.
fn terrain_color(height: f32, amplitude: f32) -> [f32; 3] {
    let t = ((height / amplitude) + 1.0) * 0.5;
    let t = t.clamp(0.0, 1.0);

    if t < 0.15 {
        [0.08, 0.20, 0.14]
    } else if t < 0.30 {
        [0.15, 0.45, 0.10]
    } else if t < 0.42 {
        [0.25, 0.55, 0.15]
    } else if t < 0.50 {
        [0.50, 0.52, 0.25]
    } else if t < 0.60 {
        [0.45, 0.35, 0.22]
    } else if t < 0.72 {
        [0.50, 0.48, 0.42]
    } else if t < 0.85 {
        [0.58, 0.55, 0.50]
    } else {
        [0.88, 0.88, 0.85]
    }
}
