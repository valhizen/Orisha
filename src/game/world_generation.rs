use glam::Vec3;
use noise::{NoiseFn, Perlin};

use crate::gpu::buffer::Vertex;

/// World-space UV tiling factor used for terrain and water texturing.
const UV_TILE_SCALE: f32 = 8.0;
/// Base water level before it is scaled by terrain amplitude.
const WATER_LEVEL: f64 = 0.02;

/// Procedural terrain generator.
///
/// This struct stores several Perlin noise sources that are combined to build
/// continents, mountains, rivers, moisture, and smaller surface detail.
pub struct Terrain {
    pub amplitude: f32,
    continent:  Perlin,
    erosion:    Perlin,
    detail:     Perlin,
    ridge:      Perlin,
    warp:       Perlin,
    biome:      Perlin,
    moisture:   Perlin,
    river:      Perlin,
    river_warp: Perlin,
    cave:       Perlin,
    cave2:      Perlin,
    cliff:      Perlin,
    terrace:    Perlin,
    micro:      Perlin,
}

impl Terrain {
    /// Creates a terrain generator with a fixed height scale and seed.
    ///
    /// Using the same seed gives the same world shape.
    pub fn new(amplitude: f32, seed: u32) -> Self {
        Self {
            amplitude,
            continent:  Perlin::new(seed),
            erosion:    Perlin::new(seed.wrapping_add(1)),
            detail:     Perlin::new(seed.wrapping_add(2)),
            ridge:      Perlin::new(seed.wrapping_add(3)),
            warp:       Perlin::new(seed.wrapping_add(4)),
            biome:      Perlin::new(seed.wrapping_add(5)),
            moisture:   Perlin::new(seed.wrapping_add(6)),
            river:      Perlin::new(seed.wrapping_add(7)),
            river_warp: Perlin::new(seed.wrapping_add(8)),
            cave:       Perlin::new(seed.wrapping_add(9)),
            cave2:      Perlin::new(seed.wrapping_add(10)),
            cliff:      Perlin::new(seed.wrapping_add(11)),
            terrace:    Perlin::new(seed.wrapping_add(12)),
            micro:      Perlin::new(seed.wrapping_add(13)),
        }
    }

    /// Builds the base terrain shape before rivers/lakes are carved out.
    ///
    /// Returns:
    /// - base height
    /// - biome factor
    /// - moisture factor
    fn base_height(&self, wx: f64, wz: f64) -> (f64, f64, f64) {
        // Warp the sampling position so the terrain feels less grid-like.
        let warp_str = 150.0;
        let wx_w = wx + fbm(&self.warp, wx, wz, 0.001, 3, 2.0, 0.5) * warp_str;
        let wz_w = wz + fbm(&self.warp, wx + 3000.0, wz + 3000.0, 0.001, 3, 2.0, 0.5) * warp_str;

        let biome_raw = fbm(&self.biome, wx_w, wz_w, 0.00035, 3, 2.0, 0.5);
        let biome_t = ((biome_raw * 2.2).clamp(-1.0, 1.0) * 0.5 + 0.5).clamp(0.0, 1.0);

        let moisture_raw = fbm(&self.moisture, wx_w + 5000.0, wz_w + 5000.0, 0.0005, 3, 2.0, 0.5);
        let moisture = ((moisture_raw * 1.6).clamp(-1.0, 1.0) * 0.5 + 0.5).clamp(0.0, 1.0);

        let continent = fbm(&self.continent, wx_w, wz_w, 0.00025, 5, 2.0, 0.5);
        let continent2 = fbm(&self.continent, wx_w + 10000.0, wz_w + 10000.0, 0.0006, 4, 2.0, 0.5);
        let continent_shaped = continent * 0.7 + continent2 * 0.4;

        let hills_raw = fbm(&self.erosion, wx_w, wz_w, 0.003, 4, 2.0, 0.5);
        let hills = hills_raw * lerp(0.25, 0.10, biome_t);

        let angle = 0.55_f64;
        let (sa, ca) = (angle.sin(), angle.cos());
        let rx = wx_w * ca - wz_w * sa;
        let rz = wx_w * sa + wz_w * ca;
        let ridge = ridged_fbm(&self.ridge, rx * 1.8, rz, 0.001, 5, 2.0, 0.5);
        let ridge_shaped = (ridge - 0.45) * 3.0;
        let mountain_mask = smoothstep(0.45, 0.75, biome_t);
        let ridge_contrib = ridge_shaped * mountain_mask * mountain_mask;

        let plateau_raw = self.cliff.get([wx_w * 0.0008, wz_w * 0.0008]);
        let plateau_mask = smoothstep(0.2, 0.55, plateau_raw)
            * smoothstep(0.3, 0.6, biome_t)
            * (1.0 - mountain_mask);

        let detail_raw = fbm(&self.detail, wx_w, wz_w, 0.012, 3, 2.3, 0.45);
        let detail_strength = lerp(0.10, 0.40, biome_t);
        let detail = detail_raw * detail_strength;

        let micro_raw = fbm(&self.micro, wx, wz, 0.05, 2, 2.5, 0.4);
        let micro = micro_raw * lerp(0.015, 0.04, biome_t);

        let raw = continent_shaped + hills + ridge_contrib * 0.8 + detail + micro;

        let with_plateau = if plateau_mask > 0.01 {
            let edge_noise = fbm(&self.micro, wx * 0.8, wz * 0.8, 0.008, 2, 2.0, 0.5) * 0.03;
            let lift = 0.15 + edge_noise;
            let lifted = raw + lift * plateau_mask;
            let edge = smoothstep(0.0, 0.15, plateau_mask);
            lerp(raw, lifted, edge)
        } else {
            raw
        };

        let terrace_raw = self.terrace.get([wx_w * 0.002, wz_w * 0.002]);
        let terrace_mask = smoothstep(0.7, 0.95, biome_t) * smoothstep(0.1, 0.5, terrace_raw);
        let steps = 4.0;
        let quantised = (with_plateau * steps).round() / steps;
        let terraced = with_plateau + (quantised - with_plateau) * terrace_mask * 0.25;

        let valley_f = lerp(1.0, 0.35, (1.0 - biome_t) * moisture);
        let h = terraced * valley_f;

        (h, biome_t, moisture)
    }

    /// Computes how strongly a river should carve into the terrain here.
    fn river_depth(&self, wx: f64, wz: f64, base_h: f64, biome_t: f64) -> f64 {
        // Warp river paths so they meander instead of following straight noise bands.
        let warp_s = 350.0;
        let rw_x = wx + self.river_warp.get([wx * 0.0005, wz * 0.0005]) * warp_s
            + self.river_warp.get([wx * 0.002 + 700.0, wz * 0.002 + 700.0]) * 80.0;
        let rw_z = wz + self.river_warp.get([wx * 0.0005 + 500.0, wz * 0.0005 + 500.0]) * warp_s
            + self.river_warp.get([wx * 0.002 + 1200.0, wz * 0.002 + 1200.0]) * 80.0;

        let r1 = self.river.get([rw_x * 0.0008, rw_z * 0.0008]);
        let width1 = lerp(0.08, 0.04, smoothstep(-0.1, 0.3, base_h));
        let r1_band = 1.0 - smoothstep(0.0, width1, r1.abs());

        let r2 = self.river.get([rw_x * 0.002 + 100.0, rw_z * 0.002 + 100.0]);
        let r2_band = 1.0 - smoothstep(0.0, 0.025, r2.abs());

        let trib = self.river.get([rw_x * 0.005 + 300.0, rw_z * 0.005 + 300.0]);
        let trib_band = (1.0 - smoothstep(0.0, 0.015, trib.abs())) * 0.35;

        let low_preference = smoothstep(0.2, -0.1, base_h);
        let mountain_suppress = 1.0 - smoothstep(0.6, 0.85, biome_t);

        let river_val = (r1_band.max(r2_band * 0.5) + trib_band).min(1.0);
        river_val * lerp(0.5, 1.0, low_preference) * mountain_suppress
    }

    /// Computes how strongly a lake basin should form here.
    fn lake_depth(&self, wx: f64, wz: f64, base_h: f64) -> f64 {
        let lake_noise = fbm(&self.moisture, wx + 8000.0, wz + 8000.0, 0.0006, 2, 2.0, 0.5);
        let lake_basin = smoothstep(-0.05, 0.2, -lake_noise);
        let low_area = smoothstep(0.08, -0.2, base_h);
        lake_basin * low_area
    }

    /// Adds extra vertical displacement that helps create sharper cliffy shapes.
    ///
    /// Even though this uses names like "cave", this is still heightmap terrain:
    /// one final height value is produced for each world (x, z) position.
    fn cave_displacement(&self, wx: f64, wz: f64, biome_t: f64) -> f64 {
        let c1 = self.cave.get([wx * 0.006, wz * 0.006]);
        let c2 = self.cave2.get([wx * 0.009 + 200.0, wz * 0.009 + 200.0]);
        let cave_shape = (c1 * c2).abs();
        let cave_mask = smoothstep(0.0, 0.015, cave_shape);
        let overhang_noise = self.cave.get([wx * 0.003, wz * 0.003 + 1000.0]);
        let overhang_mask = smoothstep(0.4, 0.7, overhang_noise.abs());
        let mountain_mask = smoothstep(0.4, 0.8, biome_t);
        let cave_depth = (1.0 - cave_mask) * mountain_mask * 0.2;
        let overhang = overhang_mask * mountain_mask * 0.08;
        -(cave_depth + overhang)
    }

    /// Full terrain sample used by both gameplay and rendering.
    ///
    /// Returns:
    /// - final height
    /// - water carve amount
    /// - biome factor
    /// - moisture factor
    fn compute_full(&self, xd: f64, zd: f64) -> (f64, f64, f64, f64) {
        let (base_h, biome_t, moisture) = self.base_height(xd, zd);

        let river = self.river_depth(xd, zd, base_h, biome_t);
        let river_carve = river * lerp(0.15, 0.45, moisture) * (1.0 - biome_t * 0.3);

        let lake = self.lake_depth(xd, zd, base_h);
        let lake_carve = lake * 0.25;

        let water_carve = river_carve.max(lake_carve);
        let cave_disp = self.cave_displacement(xd, zd, biome_t);

        let h = base_h - water_carve + cave_disp;

        (h, water_carve, biome_t, moisture)
    }

    /// Returns the final terrain height in world space.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let (h, water, _, _) = self.compute_full(x as f64, z as f64);
        let final_h = if h < WATER_LEVEL && water > 0.05 {
            WATER_LEVEL
        } else {
            h
        };
        final_h as f32 * self.amplitude
    }

    /// Returns a fuller terrain sample for rendering decisions.
    ///
    /// This includes terrain height plus extra data used to color the mesh.
    pub fn full_sample_at(&self, x: f32, z: f32) -> (f32, f32, f32, f32) {
        let (h, water, biome, moisture) = self.compute_full(x as f64, z as f64);
        (
            h as f32 * self.amplitude,
            water as f32,
            biome as f32,
            moisture as f32,
        )
    }

    /// Approximates the terrain normal by sampling nearby heights.
    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        let e = 0.2;
        let h_left  = self.height_at(x - e, z);
        let h_right = self.height_at(x + e, z);
        let h_down  = self.height_at(x, z - e);
        let h_up    = self.height_at(x, z + e);
        Vec3::new(h_left - h_right, 2.0 * e, h_down - h_up).normalize_or(Vec3::Y)
    }
}

/// Linear interpolation helper.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Smooth interpolation helper often used for soft masks.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Standard fractal Brownian motion noise built from multiple octaves.
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

/// Ridged noise variant often useful for mountain-like shapes.
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

/// Number of quads along one side of a terrain chunk.
pub const CHUNK_SIZE: u32 = 64;
/// World-space spacing between terrain vertices.
pub const CHUNK_SCALE: f32 = 2.0;
/// Final world-space size of one chunk.
pub const CHUNK_WORLD_SIZE: f32 = CHUNK_SIZE as f32 * CHUNK_SCALE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Integer chunk coordinate in chunk space, not world space.
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

impl ChunkCoord {
    /// Converts chunk coordinates into the world-space origin of that chunk.
    pub fn world_origin(&self) -> (f32, f32) {
        (self.x as f32 * CHUNK_WORLD_SIZE, self.z as f32 * CHUNK_WORLD_SIZE)
    }

    /// Finds which chunk contains a world-space position.
    pub fn from_world(wx: f32, wz: f32) -> Self {
        Self {
            x: (wx / CHUNK_WORLD_SIZE).floor() as i32,
            z: (wz / CHUNK_WORLD_SIZE).floor() as i32,
        }
    }
}

/// Builds CPU-side mesh data for one terrain chunk.
///
/// Returns:
/// - terrain vertices
/// - terrain indices
/// - water vertices
/// - water indices
pub fn build_chunk_mesh(terrain: &Terrain, coord: ChunkCoord)
    -> (Vec<Vertex>, Vec<u32>, Vec<Vertex>, Vec<u32>)
{
    // Vertices per side is one more than the quad count.
    let vps = CHUNK_SIZE + 1;
    let vert_count = (vps * vps) as usize;
    let idx_count = (CHUNK_SIZE * CHUNK_SIZE * 6) as usize;

    let mut vertices = Vec::with_capacity(vert_count);
    let mut indices = Vec::with_capacity(idx_count);
    let mut is_water = Vec::with_capacity(vert_count);

    let water_surface_y = WATER_LEVEL as f32 * terrain.amplitude;
    let (ox, oz) = coord.world_origin();

    // Build the terrain vertex grid in world space.
    for z in 0..vps {
        for x in 0..vps {
            let wx = ox + x as f32 * CHUNK_SCALE;
            let wz = oz + z as f32 * CHUNK_SCALE;
            let (wy, water, biome, moisture) = terrain.full_sample_at(wx, wz);

            let submerged = water > 0.05 && wy < water_surface_y;
            is_water.push(submerged);

            let color = if submerged {
                underwater_color(wy, water_surface_y)
            } else {
                terrain_color(wy, terrain.amplitude, biome, moisture, water)
            };

            vertices.push(Vertex {
                position: [wx, wy, wz],
                normal:   [0.0, 1.0, 0.0],
                color,
                uv:       [wx / UV_TILE_SCALE, wz / UV_TILE_SCALE],
            });
        }
    }

    // Connect the grid into triangles.
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

    // Recompute normals from triangle geometry for smoother lighting.
    compute_smooth_normals(&mut vertices, &indices);

    let mut water_verts = Vec::new();
    let mut water_idxs: Vec<u32> = Vec::new();
    let mut water_remap = vec![u32::MAX; vert_count];

    // Build a second flat mesh for water anywhere the terrain is submerged.
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let tl = (z * vps + x) as usize;
            let tr = tl + 1;
            let bl = tl + vps as usize;
            let br = bl + 1;

            if is_water[tl] || is_water[tr] || is_water[bl] || is_water[br] {
                for &vi in &[tl, tr, bl, br] {
                    if water_remap[vi] == u32::MAX {
                        water_remap[vi] = water_verts.len() as u32;
                        let wx = vertices[vi].position[0];
                        let wz = vertices[vi].position[2];
                        water_verts.push(Vertex {
                            position: [wx, water_surface_y, wz],
                            normal:   [0.0, 1.0, 0.0],
                            color:    [0.02, 0.10, 0.30],
                            uv:       [wx / UV_TILE_SCALE, wz / UV_TILE_SCALE],
                        });
                    }
                }
                water_idxs.push(water_remap[tl]);
                water_idxs.push(water_remap[bl]);
                water_idxs.push(water_remap[tr]);
                water_idxs.push(water_remap[tr]);
                water_idxs.push(water_remap[bl]);
                water_idxs.push(water_remap[br]);
            }
        }
    }

    (vertices, indices, water_verts, water_idxs)
}

/// Rebuilds vertex normals by averaging triangle normals.
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

/// Interpolates between two RGB colors.
fn color_lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Chooses a seabed color based on how deep below the water surface it is.
fn underwater_color(height: f32, water_y: f32) -> [f32; 3] {
    let depth = (water_y - height).max(0.0);
    let t = (depth / 20.0).clamp(0.0, 1.0);
    let shallow_bed: [f32; 3] = [0.32, 0.28, 0.18];
    let deep_bed:    [f32; 3] = [0.06, 0.05, 0.03];
    color_lerp(shallow_bed, deep_bed, t)
}

/// Picks a terrain color from height, biome, moisture, and nearby water amount.
fn terrain_color(height: f32, amplitude: f32, biome: f32, moisture: f32, water: f32) -> [f32; 3] {
    let beach:          [f32; 3] = [0.78, 0.72, 0.52];
    let wet_sand:       [f32; 3] = [0.55, 0.48, 0.30];
    let mud:            [f32; 3] = [0.32, 0.24, 0.13];
    let grass_lush:     [f32; 3] = [0.12, 0.40, 0.06];
    let grass_dry:      [f32; 3] = [0.42, 0.47, 0.14];
    let grass_meadow:   [f32; 3] = [0.20, 0.52, 0.12];
    let grass_dark:     [f32; 3] = [0.08, 0.30, 0.04];
    let forest_floor:   [f32; 3] = [0.05, 0.18, 0.03];
    let shrub:          [f32; 3] = [0.28, 0.35, 0.10];
    let dirt:           [f32; 3] = [0.42, 0.32, 0.20];
    let dirt_red:       [f32; 3] = [0.50, 0.28, 0.14];
    let rock_light:     [f32; 3] = [0.52, 0.50, 0.44];
    let rock_warm:      [f32; 3] = [0.48, 0.40, 0.32];
    let rock_dark:      [f32; 3] = [0.32, 0.30, 0.27];
    let cliff_face:     [f32; 3] = [0.40, 0.37, 0.32];
    let alpine:         [f32; 3] = [0.35, 0.38, 0.28];
    let snow_rock:      [f32; 3] = [0.72, 0.74, 0.77];
    let snow:           [f32; 3] = [0.94, 0.96, 0.98];

    if water > 0.03 {
        let shore_t = ((water - 0.03) / 0.12).clamp(0.0, 1.0);
        let shore = color_lerp(beach, wet_sand, moisture);
        return color_lerp(shore, mud, shore_t);
    }

    let h = height / amplitude;

    if h < -0.20 {
        let t = ((h + 0.40) / 0.20).clamp(0.0, 1.0);
        let low = color_lerp(mud, wet_sand, 1.0 - moisture);
        color_lerp(low, grass_lush, t * moisture)
    } else if h < -0.05 {
        let t = ((h + 0.20) / 0.15).clamp(0.0, 1.0);
        let grass = color_lerp(grass_dry, grass_meadow, moisture);
        color_lerp(grass_lush, grass, t)
    } else if h < 0.08 {
        let t = ((h + 0.05) / 0.13).clamp(0.0, 1.0);
        let low = color_lerp(grass_meadow, grass_dark, moisture);
        let high = if moisture > 0.5 {
            color_lerp(grass_dark, forest_floor, (moisture - 0.5) * 2.0)
        } else {
            color_lerp(shrub, grass_dry, 1.0 - moisture * 2.0)
        };
        color_lerp(low, high, t)
    } else if h < 0.20 {
        let t = ((h - 0.08) / 0.12).clamp(0.0, 1.0);
        let low = if moisture > 0.4 {
            color_lerp(forest_floor, grass_dark, (1.0 - moisture) * 1.5)
        } else {
            color_lerp(shrub, dirt, 1.0 - moisture * 2.5)
        };
        let high = color_lerp(dirt, dirt_red, 1.0 - moisture);
        color_lerp(low, high, t)
    } else if h < 0.35 {
        let t = ((h - 0.20) / 0.15).clamp(0.0, 1.0);
        let low = color_lerp(dirt, dirt_red, biome * 0.5);
        let high = color_lerp(rock_warm, rock_light, moisture);
        color_lerp(low, high, t)
    } else if h < 0.55 {
        let t = ((h - 0.35) / 0.20).clamp(0.0, 1.0);
        let base_rock = color_lerp(rock_light, cliff_face, biome);
        let with_alpine = color_lerp(base_rock, alpine, moisture * 0.4);
        color_lerp(with_alpine, rock_dark, t)
    } else if h < 0.75 {
        let t = ((h - 0.55) / 0.20).clamp(0.0, 1.0);
        let rock = color_lerp(rock_dark, cliff_face, biome * 0.5);
        color_lerp(rock, snow_rock, t)
    } else {
        let t = ((h - 0.75) / 0.25).clamp(0.0, 1.0);
        color_lerp(snow_rock, snow, t)
    }
}
