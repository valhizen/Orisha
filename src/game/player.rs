use glam::{Vec3, Quat};

use crate::gpu::buffer::Vertex;

use super::physics::{self, Capsule, CollisionResult};
use super::world_generation::Terrain;

pub struct Player {
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub movement_speed: f32,
    pub sprint_multiplier: f32,
    pub jump_force: f32,
    pub is_grounded: bool,
    pub ground_normal: Vec3,
    pub height: f32,
    pub radius: f32,
    pub collider: Capsule,
}

impl Player {
    pub fn new(position: Vec3) -> Self {
        let height = 2.0;
        let radius = 0.4;
        Self {
            position,
            rotation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            movement_speed: 5.0,
            sprint_multiplier: 1.8,
            jump_force: 8.0,
            is_grounded: false,
            ground_normal: Vec3::Y,
            height,
            radius,
            collider: Capsule::new(height, radius),
        }
    }

    pub fn move_direction(&mut self, direction: Vec3, sprinting: bool) {
        if direction.length_squared() > 0.0 {
            let speed = if sprinting {
                self.movement_speed * self.sprint_multiplier
            } else {
                self.movement_speed
            };

            let movement = direction.normalize() * speed;
            self.velocity.x = movement.x;
            self.velocity.z = movement.z;

            if movement.length_squared() > 0.01 {
                self.rotation = Quat::from_rotation_y(movement.x.atan2(movement.z));
            }
        } else {
            self.velocity.x *= 0.8;
            self.velocity.z *= 0.8;
        }
    }

    pub fn update(&mut self, delta_time: f32, terrain: &Terrain) {
        let result: CollisionResult =
            physics::step(self.position, self.velocity, delta_time, &self.collider, terrain);

        self.position = result.position;
        self.velocity = result.velocity;
        self.is_grounded = result.grounded;
        self.ground_normal = result.ground_normal;
    }

    pub fn jump(&mut self) {
        if self.is_grounded {
            self.velocity.y = self.jump_force;
            self.is_grounded = false;
        }
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn shoulder_position(&self) -> Vec3 {
        self.position + Vec3::new(0.0, self.height * 0.3, 0.0)
    }
}

pub fn player_geometry() -> (Vec<Vertex>, Vec<u32>) {
    let hw = 0.5;
    let hh = 1.0;
    let hd = 0.5;

    let faces: [([f32; 3], [[f32; 3]; 4], [f32; 3]); 6] = [
        ([0.0, 0.0, 1.0], [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]], [0.9, 0.7, 0.55]),
        ([0.0, 0.0, -1.0], [[hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd]], [0.25, 0.25, 0.28]),
        ([1.0, 0.0, 0.0], [[hw, -hh, hd], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd]], [0.2, 0.6, 0.3]),
        ([-1.0, 0.0, 0.0], [[-hw, -hh, -hd], [-hw, -hh, hd], [-hw, hh, hd], [-hw, hh, -hd]], [0.7, 0.2, 0.2]),
        ([0.0, 1.0, 0.0], [[-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [-hw, hh, -hd]], [0.35, 0.2, 0.1]),
        ([0.0, -1.0, 0.0], [[-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [-hw, -hh, hd]], [0.15, 0.15, 0.15]),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (normal, corners, color) in &faces {
        let base = vertices.len() as u32;
        for &pos in corners {
            vertices.push(Vertex {
                position: pos,
                normal: *normal,
                color: *color,
                uv: [0.0, 0.0],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    (vertices, indices)
}
