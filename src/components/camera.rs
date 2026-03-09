use glam::{Vec2, Vec3, Mat4 };

pub enum CameraMovement {
    FORWARD,
    BACKWARD,
    RIGHT,
    LEFT,
    UP,
    DOWN
}


pub struct Camera3DThirdPersonPrespective{

    // Camera Position For Player
    position : Vec3,
    front : Vec3,
    up : Vec3,
    right : Vec3,
    world_up : Vec3,

    // Eulars Angle
    yaw : f32,
    pitch : f32,

    // Camera Setting/sensitiviry
    zoom : f32,

    mouse_sensitivity : f32,
    movement_speed : f32
}

impl Camera3DThirdPersonPrespective{

    pub fn new(_position : Vec3, _front : Vec3, _up : Vec3, _right : Vec3, _world_up : Vec3, _yaw:f32, _pitch : f32, _zoom : f32,
        _mouse_sensitivity : f32, _movement_speed : f32) -> Self{

        return Self{position : _position, front : _front, up : _up, right : _right, world_up : _world_up, yaw : _yaw, pitch : _pitch, zoom: _zoom,
            mouse_sensitivity : _mouse_sensitivity, movement_speed: _movement_speed};
    }

    pub fn get_view_matexi(&self) -> Mat4{
        return Mat4::look_at_rh(self.position, self.position + self.front, self.up);
    }

    pub fn get_projection_matrxi(&self, aspect_ratio : f32, near : f32, far : f32) -> Mat4 {
        return Mat4::perspective_rh(self.zoom.to_radians(), aspect_ratio, near, far);
    }
}
