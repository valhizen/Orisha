use std::collections::{HashMap, HashSet, VecDeque};

use crate::gpu::{
    allocator::GpuAllocator,
    buffer::GpuMesh,
    commands::{Commands, MAX_FRAMES_IN_FLIGHT},
    device::Device,
};

use super::world_generation::{self, ChunkCoord, Terrain};

// How many chunks around the player/camera should stay loaded.
const VIEW_RADIUS: i32 = 8;

// Limit chunk generation per frame so movement does not cause a huge stall.
const MAX_CHUNKS_PER_FRAME: usize = 1;

// Wait a few frames before freeing old chunk GPU resources.
// This helps avoid destroying meshes that may still be used by the GPU.
const GRAVEYARD_DELAY: u64 = (MAX_FRAMES_IN_FLIGHT as u64) + 1;

// One fully loaded terrain chunk currently living on the GPU.
struct LoadedChunk {
    mesh: GpuMesh,
    water_mesh: Option<GpuMesh>,
}

// A chunk that was removed from the world but is being kept alive briefly
// until it is safe to destroy its GPU buffers.
struct GraveyardEntry {
    mesh: GpuMesh,
    water_mesh: Option<GpuMesh>,
    retired_frame: u64,
}

// Handles chunk streaming around the current centre point.
// It loads nearby chunks, unloads far ones, and delays destruction safely.
pub struct ChunkManager {
    chunks: HashMap<ChunkCoord, LoadedChunk>,
    last_centre: Option<ChunkCoord>,
    pending: Vec<ChunkCoord>,
    graveyard: VecDeque<GraveyardEntry>,
    frame: u64,
}

impl ChunkManager {
    /// Creates an empty chunk manager.
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            last_centre: None,
            pending: Vec::new(),
            graveyard: VecDeque::new(),
            frame: 0,
        }
    }

    /// Updates which chunks should exist around the current world position.
    ///
    /// This function does three main jobs:
    /// 1. safely destroy old chunks after a delay
    /// 2. decide which chunks should be loaded near the centre
    /// 3. generate a small number of missing chunks this frame
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

        // Destroy chunks that have waited long enough in the graveyard.
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

        // Only rebuild the wanted chunk list if the centre chunk changed.
        if self.last_centre != Some(centre) {
            self.last_centre = Some(centre);

            // Build the set of chunks that should be loaded around the centre.
            let mut desired = HashSet::new();
            for dz in -VIEW_RADIUS..=VIEW_RADIUS {
                for dx in -VIEW_RADIUS..=VIEW_RADIUS {
                    desired.insert(ChunkCoord {
                        x: centre.x + dx,
                        z: centre.z + dz,
                    });
                }
            }

            // Find loaded chunks that are now outside the wanted area.
            let to_remove: Vec<ChunkCoord> = self
                .chunks
                .keys()
                .filter(|c| !desired.contains(c))
                .copied()
                .collect();

            // Move removed chunks into the graveyard instead of freeing them immediately.
            for coord in to_remove {
                if let Some(chunk) = self.chunks.remove(&coord) {
                    self.graveyard.push_back(GraveyardEntry {
                        mesh: chunk.mesh,
                        water_mesh: chunk.water_mesh,
                        retired_frame: self.frame,
                    });
                }
            }

            // Rebuild the pending generation list with chunks we still need.
            self.pending.clear();
            for &coord in &desired {
                if !self.chunks.contains_key(&coord) {
                    self.pending.push(coord);
                }
            }

            // Sort so nearer chunks are generated first.
            self.pending.sort_by_key(|c| {
                let dx = c.x - centre.x;
                let dz = c.z - centre.z;
                std::cmp::Reverse(dx * dx + dz * dz)
            });
        }

        // Generate only a few chunks per frame to spread the work over time.
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

    /// Returns all solid terrain meshes currently loaded.
    pub fn meshes(&self) -> impl Iterator<Item = &GpuMesh> {
        self.chunks.values().map(|c| &c.mesh)
    }

    /// Returns all water meshes currently loaded.
    pub fn water_meshes(&self) -> impl Iterator<Item = &GpuMesh> {
        self.chunks.values().filter_map(|c| c.water_mesh.as_ref())
    }

    /// Frees all chunk resources immediately.
    /// Used during shutdown when the app is being destroyed.
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
