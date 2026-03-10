use glam::{Vec3, Mat4, Quat};

pub enum CameraMode {
    ThirdPerson,
    FirstPerson,
    Free,
}

pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    pub focus_point: Vec3,
    pub distance: f32,
    pub height_offset: f32,
    pub shoulder_offset: f32,
    pub smoothness: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub sensitivity: f32,
    pub mode: CameraMode,
}

impl Camera {
    pub fn new_third_person(target: Vec3) -> Self {
        Self {
            position: target + Vec3::new(1.0, 2.0, 4.0),
            rotation: Quat::IDENTITY,
            focus_point: target,
            distance: 3.5,
            height_offset: 1.6,
            shoulder_offset: 0.8,
            smoothness: 8.0,
            yaw: 0.0,
            pitch: 10.0_f32.to_radians(),
            fov: 55.0_f32.to_radians(),
            near_plane: 0.1,
            far_plane: 1000.0,
            sensitivity: 0.003,
            mode: CameraMode::ThirdPerson,
        }
    }

    pub fn new_first_person(position: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            focus_point: Vec3::ZERO,
            distance: 0.0,
            height_offset: 1.7,
            shoulder_offset: 0.0,
            smoothness: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            fov: 60.0_f32.to_radians(),
            near_plane: 0.1,
            far_plane: 1000.0,
            sensitivity: 0.003,
            mode: CameraMode::FirstPerson,
        }
    }

    pub fn follow_target(&mut self, target: Vec3, delta_time: f32) {
        match self.mode {
            CameraMode::ThirdPerson => {
                let t = (self.smoothness * delta_time).clamp(0.0, 1.0);
                self.focus_point = self.focus_point.lerp(target, t);

                let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
                let (sin_pitch, cos_pitch) = self.pitch.sin_cos();

                let behind = Vec3::new(sin_yaw, 0.0, cos_yaw);
                let cam_right = Vec3::new(cos_yaw, 0.0, -sin_yaw);

                let ideal = self.focus_point
                    + behind * (self.distance * cos_pitch)
                    + Vec3::Y * (self.height_offset + self.distance * sin_pitch)
                    + cam_right * self.shoulder_offset;

                self.position = self.position.lerp(ideal, t);
            }
            CameraMode::FirstPerson => {
                self.position = target + Vec3::new(0.0, self.height_offset, 0.0);
                let yaw_quat = Quat::from_rotation_y(self.yaw);
                let pitch_quat = Quat::from_rotation_x(self.pitch);
                self.rotation = yaw_quat * pitch_quat;
            }
            CameraMode::Free => {}
        }
    }

    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * self.sensitivity;
        self.pitch = (self.pitch - delta_y * self.sensitivity)
            .clamp(-60.0_f32.to_radians(), 60.0_f32.to_radians());
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta * 0.5).clamp(1.5, 12.0);
    }

    pub fn view_matrix(&self) -> Mat4 {
        match self.mode {
            CameraMode::ThirdPerson => {
                let look_target = self.focus_point + Vec3::new(0.0, 0.3, 0.0);
                Mat4::look_at_rh(self.position, look_target, Vec3::Y)
            }
            _ => {
                let forward = self.rotation * -Vec3::Z;
                Mat4::look_at_rh(self.position, self.position + forward, Vec3::Y)
            }
        }
    }

    /// Projection matrix with Vulkan Y-flip.
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        let mut proj = Mat4::perspective_rh(
            self.fov,
            aspect_ratio,
            self.near_plane,
            self.far_plane,
        );
        proj.y_axis.y *= -1.0;
        proj
    }
}
