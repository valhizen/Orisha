use glam::Vec3;

use super::world_generation::Terrain;

const GRAVITY: f32 = -25.0;
const TERMINAL_VELOCITY: f32 = -50.0;
const GROUND_STICK_THRESHOLD: f32 = 0.08;
const SLOPE_LIMIT_COS: f32 = 0.5;
const SLIDE_FORCE: f32 = 12.0;

#[derive(Clone, Copy, Debug)]
pub struct Capsule {
    pub half_height: f32,
    pub radius: f32,
}

impl Capsule {
    pub fn new(total_height: f32, radius: f32) -> Self {
        Self {
            half_height: (total_height * 0.5 - radius).max(0.0),
            radius,
        }
    }

    pub fn foot_offset(&self) -> f32 {
        self.half_height + self.radius
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CollisionResult {
    pub position: Vec3,
    pub velocity: Vec3,
    pub grounded: bool,
    pub ground_normal: Vec3,
}

pub fn step(
    position: Vec3,
    velocity: Vec3,
    dt: f32,
    capsule: &Capsule,
    terrain: &Terrain,
) -> CollisionResult {
    let mut vel = velocity;

    vel.y = (vel.y + GRAVITY * dt).max(TERMINAL_VELOCITY);
    let mut pos = position + vel * dt;
    let ground_y = terrain.height_at(pos.x, pos.z);
    let normal = terrain.normal_at(pos.x, pos.z);
    let foot_y = pos.y - capsule.foot_offset();

    let mut grounded = false;

    if foot_y < ground_y + GROUND_STICK_THRESHOLD {
        pos.y = ground_y + capsule.foot_offset();

        let slope_cos = normal.dot(Vec3::Y);

        if slope_cos >= SLOPE_LIMIT_COS {
            vel.y = 0.0;
            grounded = true;
        } else {
            let slide_dir = Vec3::new(normal.x, 0.0, normal.z).normalize_or(Vec3::ZERO);
            vel += slide_dir * SLIDE_FORCE * dt;
        }
    }

    CollisionResult {
        position: pos,
        velocity: vel,
        grounded,
        ground_normal: normal,
    }
}
