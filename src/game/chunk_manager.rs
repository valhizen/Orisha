use std::collections::{HashMap, VecDeque};

use crate::gpu::{
    allocator::GpuAllocator,
    buffer::GpuMesh,
    commands::{Commands, MAX_FRAMES_IN_FLIGHT},
    device::Device,
};

use super::world_generation::{self, ChunkCoord, Terrain};

// Chunk load radius around the player.
const VIEW_RADIUS: i32 = 500;

const MAX_CHUNKS_PER_FRAME: usize = 1;

// Frames before a retired mesh is destroyed.
const GRAVEYARD_DELAY: u64 = (MAX_FRAMES_IN_FLIGHT as u64) + 1;

struct LoadedChunk {
    mesh: GpuMesh,
}

struct GraveyardEntry {
    mesh: GpuMesh,
    retired_frame: u64,
}

// Streams terrain chunks around the player.
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
        player_x: f32,
        player_z: f32,
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
            } else {
                break;
            }
        }

        let centre = ChunkCoord::from_world(player_x, player_z);

        // Recalculate when the player crosses a chunk boundary.
        if self.last_centre != Some(centre) {
            self.last_centre = Some(centre);

            let mut desired: HashMap<ChunkCoord, ()> = HashMap::new();
            for dz in -VIEW_RADIUS..=VIEW_RADIUS {
                for dx in -VIEW_RADIUS..=VIEW_RADIUS {
                    desired.insert(
                        ChunkCoord {
                            x: centre.x + dx,
                            z: centre.z + dz,
                        },
                        (),
                    );
                }
            }

            let to_remove: Vec<ChunkCoord> = self
                .chunks
                .keys()
                .filter(|c| !desired.contains_key(c))
                .copied()
                .collect();

            for coord in to_remove {
                if let Some(chunk) = self.chunks.remove(&coord) {
                    self.graveyard.push_back(GraveyardEntry {
                        mesh: chunk.mesh,
                        retired_frame: self.frame,
                    });
                }
            }

            // Queue pending chunks, closest first.
            self.pending.clear();
            for &coord in desired.keys() {
                if !self.chunks.contains_key(&coord) {
                    self.pending.push(coord);
                }
            }
            self.pending.sort_by_key(|c| {
                let dx = c.x - centre.x;
                let dz = c.z - centre.z;
                dx * dx + dz * dz
            });
        }

        let to_gen = self.pending.len().min(MAX_CHUNKS_PER_FRAME);
        for _ in 0..to_gen {
            let coord = self.pending.remove(0);
            if !self.chunks.contains_key(&coord) {
                let (verts, idxs) = world_generation::build_chunk_mesh(terrain, coord);
                let mesh = GpuMesh::upload(device, allocator, commands, &verts, &idxs);
                self.chunks.insert(coord, LoadedChunk { mesh });
            }
        }
    }

    pub fn meshes(&self) -> impl Iterator<Item = &GpuMesh> {
        self.chunks.values().map(|c| &c.mesh)
    }

    pub fn destroy_all(&mut self, device: &Device, allocator: &GpuAllocator) {
        for (_, chunk) in self.chunks.drain() {
            chunk.mesh.destroy(device, allocator);
        }
        for entry in self.graveyard.drain(..) {
            entry.mesh.destroy(device, allocator);
        }
    }
}
