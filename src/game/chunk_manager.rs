use std::collections::{HashMap, HashSet, VecDeque};

use crate::gpu::{
    allocator::GpuAllocator,
    buffer::GpuMesh,
    commands::{Commands, MAX_FRAMES_IN_FLIGHT},
    device::Device,
};

use super::world_generation::{self, ChunkCoord, Terrain};

const VIEW_RADIUS: i32 = 8;

const MAX_CHUNKS_PER_FRAME: usize = 1;

const GRAVEYARD_DELAY: u64 = (MAX_FRAMES_IN_FLIGHT as u64) + 1;

struct LoadedChunk {
    mesh: GpuMesh,
    water_mesh: Option<GpuMesh>,
}

struct GraveyardEntry {
    mesh: GpuMesh,
    water_mesh: Option<GpuMesh>,
    retired_frame: u64,
}

pub struct ChunkManager {
    chunks: HashMap<ChunkCoord, LoadedChunk>,
    last_centre: Option<ChunkCoord>,
    pending: Vec<ChunkCoord>,
    graveyard: VecDeque<GraveyardEntry>,
    frame: u64,
}

impl ChunkManager {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            last_centre: None,
            pending: Vec::new(),
            graveyard: VecDeque::new(),
            frame: 0,
        }
    }

    pub fn update(
        &mut self,
        center_x: f32,
        center_z: f32,
        terrain: &Terrain,
        device: &Device,
        allocator: &GpuAllocator,
        commands: &Commands,
    ) {
        self.frame += 1;

        while let Some(front) = self.graveyard.front() {
            if self.frame - front.retired_frame >= GRAVEYARD_DELAY {
                let entry = self.graveyard.pop_front().unwrap();
                entry.mesh.destroy(device, allocator);
                if let Some(wm) = entry.water_mesh {
                    wm.destroy(device, allocator);
                }
            } else {
                break;
            }
        }

        let centre = ChunkCoord::from_world(center_x, center_z);

        if self.last_centre != Some(centre) {
            self.last_centre = Some(centre);

            let mut desired = HashSet::new();
            for dz in -VIEW_RADIUS..=VIEW_RADIUS {
                for dx in -VIEW_RADIUS..=VIEW_RADIUS {
                    desired.insert(ChunkCoord {
                        x: centre.x + dx,
                        z: centre.z + dz,
                    });
                }
            }

            let to_remove: Vec<ChunkCoord> = self
                .chunks
                .keys()
                .filter(|c| !desired.contains(c))
                .copied()
                .collect();

            for coord in to_remove {
                if let Some(chunk) = self.chunks.remove(&coord) {
                    self.graveyard.push_back(GraveyardEntry {
                        mesh: chunk.mesh,
                        water_mesh: chunk.water_mesh,
                        retired_frame: self.frame,
                    });
                }
            }

            self.pending.clear();
            for &coord in &desired {
                if !self.chunks.contains_key(&coord) {
                    self.pending.push(coord);
                }
            }
            self.pending.sort_by_key(|c| {
                let dx = c.x - centre.x;
                let dz = c.z - centre.z;
                std::cmp::Reverse(dx * dx + dz * dz)
            });
        }

        let to_gen = self.pending.len().min(MAX_CHUNKS_PER_FRAME);
        for _ in 0..to_gen {
            let Some(coord) = self.pending.pop() else { break };
            if !self.chunks.contains_key(&coord) {
                let (verts, idxs, w_verts, w_idxs) =
                    world_generation::build_chunk_mesh(terrain, coord);
                let mesh = GpuMesh::upload(device, allocator, commands, &verts, &idxs);
                let water_mesh = if !w_verts.is_empty() {
                    Some(GpuMesh::upload(device, allocator, commands, &w_verts, &w_idxs))
                } else {
                    None
                };
                self.chunks.insert(coord, LoadedChunk { mesh, water_mesh });
            }
        }
    }

    pub fn meshes(&self) -> impl Iterator<Item = &GpuMesh> {
        self.chunks.values().map(|c| &c.mesh)
    }

    pub fn water_meshes(&self) -> impl Iterator<Item = &GpuMesh> {
        self.chunks.values().filter_map(|c| c.water_mesh.as_ref())
    }

    pub fn destroy_all(&mut self, device: &Device, allocator: &GpuAllocator) {
        for (_, chunk) in self.chunks.drain() {
            chunk.mesh.destroy(device, allocator);
            if let Some(wm) = chunk.water_mesh {
                wm.destroy(device, allocator);
            }
        }
        for entry in self.graveyard.drain(..) {
            entry.mesh.destroy(device, allocator);
            if let Some(wm) = entry.water_mesh {
                wm.destroy(device, allocator);
            }
        }
    }
}
