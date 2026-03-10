use ash::vk;
use glam::{Mat4, Vec3};
use std::collections::HashSet;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod components;
mod game;
mod gpu;

use components::camera::Camera;
use game::player::{self, Player};
use game::time::GameClock;
use gpu::buffer::{CameraUbo, GpuMesh, PushConstants};
use gpu::renderer::Renderer;

struct App {
    // Drop order: mesh -> renderer -> window (Rust drops fields top-to-bottom)
    cube_mesh: Option<GpuMesh>,
    renderer: Option<Renderer>,
    window: Option<Window>,

    player: Option<Player>,
    camera: Option<Camera>,
    clock: Option<GameClock>,
    keys_pressed: HashSet<KeyCode>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            cube_mesh: None,
            player: None,
            camera: None,
            clock: None,
            keys_pressed: HashSet::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_title("Orisha"))
            .unwrap();

        let renderer = Renderer::new(&window).expect("Failed to initialize Vulkan renderer");

        let (vertices, indices) = player::player_geometry();
        let mesh = GpuMesh::upload(
            &renderer.device,
            &renderer.allocator,
            &renderer.commands,
            &vertices,
            &indices,
        );

        self.cube_mesh = Some(mesh);
        self.window = Some(window);
        self.renderer = Some(renderer);

        self.player = Some(Player::new(Vec3::new(0.0, 1.0, 0.0)));
        self.camera = Some(Camera::new_third_person(Vec3::ZERO));
        self.clock = Some(GameClock::new(1.0 / 60.0));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state.is_pressed() {
                        if code == KeyCode::Escape {
                            event_loop.exit();
                            return;
                        }
                        if code == KeyCode::Space {
                            if let Some(player) = &mut self.player {
                                player.jump();
                            }
                        }
                        self.keys_pressed.insert(code);
                    } else {
                        self.keys_pressed.remove(&code);
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                if let Some(camera) = &mut self.camera {
                    camera.zoom(scroll);
                }
            }

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let Some(r) = &mut self.renderer {
                        r.resize(size.width, size.height);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(clock) = &mut self.clock {
                    clock.tick();
                }

                if let (Some(player), Some(camera), Some(clock)) =
                    (&mut self.player, &mut self.camera, &mut self.clock)
                {
                    // WASD movement relative to camera yaw
                    let mut move_dir = Vec3::ZERO;
                    if self.keys_pressed.contains(&KeyCode::KeyW) { move_dir.z -= 1.0; }
                    if self.keys_pressed.contains(&KeyCode::KeyS) { move_dir.z += 1.0; }
                    if self.keys_pressed.contains(&KeyCode::KeyA) { move_dir.x -= 1.0; }
                    if self.keys_pressed.contains(&KeyCode::KeyD) { move_dir.x += 1.0; }

                    if move_dir.length_squared() > 0.0 {
                        let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
                        move_dir = Vec3::new(
                            move_dir.x * cos_yaw + move_dir.z * sin_yaw,
                            0.0,
                            -move_dir.x * sin_yaw + move_dir.z * cos_yaw,
                        );
                    }

                    let sprinting = self.keys_pressed.contains(&KeyCode::ShiftLeft);

                    player.move_direction(move_dir, clock.fixed_step, sprinting);
                    while clock.should_fixed_update() {
                        player.update(clock.fixed_step);
                    }

                    camera.follow_target(player.shoulder_position(), clock.delta);
                }

                if let (Some(renderer), Some(mesh), Some(camera), Some(player)) = (
                    &mut self.renderer,
                    &self.cube_mesh,
                    &self.camera,
                    &self.player,
                ) {
                    let extent = renderer.swapchain.surface_resolution;
                    let aspect = extent.width as f32 / extent.height.max(1) as f32;

                    let camera_ubo = CameraUbo {
                        view: camera.view_matrix().to_cols_array_2d(),
                        proj: camera.projection_matrix(aspect).to_cols_array_2d(),
                    };

                    let push = PushConstants {
                        model: Mat4::from_translation(player.position).to_cols_array_2d(),
                    };

                    renderer.draw_frame(&camera_ubo, |device, cmd, layout| unsafe {
                        device.cmd_push_constants(
                            cmd, layout, vk::ShaderStageFlags::VERTEX, 0,
                            bytemuck::bytes_of(&push),
                        );
                        device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer.buffer], &[0]);
                        device.cmd_bind_index_buffer(cmd, mesh.index_buffer.buffer, 0, vk::IndexType::UINT32);
                        device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
                    });
                }

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if let Some(camera) = &mut self.camera {
                camera.rotate(dx as f32, dy as f32);
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(renderer) = &self.renderer {
            unsafe { let _ = renderer.device.device.device_wait_idle(); }
        }
        if let (Some(mesh), Some(renderer)) = (self.cube_mesh.take(), &self.renderer) {
            mesh.destroy(&renderer.device, &renderer.allocator);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("Event loop error: {e}");
        std::process::exit(1);
    }
}
