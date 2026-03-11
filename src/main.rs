use ash::vk;
use std::time::{SystemTime, UNIX_EPOCH};
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
use game::chunk_manager::ChunkManager;
use game::model_loader;
use game::player::Player;
use game::sky;
use game::time::GameClock;
use game::world_generation::Terrain;
use gpu::buffer::{CameraUbo, GpuMesh, PushConstants, SkyPushConstants};
use gpu::pipeline_imgui::ImGuiPipeline;
use gpu::renderer::Renderer;

struct App {
    sky_mesh: Option<GpuMesh>,
    cube_mesh: Option<GpuMesh>,
    chunk_manager: Option<ChunkManager>,
    renderer: Option<Renderer>,
    window: Option<Window>,

    player: Option<Player>,
    camera: Option<Camera>,
    clock: Option<GameClock>,
    terrain: Option<Terrain>,
    model_y_offset: f32,
    keys_pressed: HashSet<KeyCode>,

    time_of_day: f32,
    time_paused: bool,
    day_speed:   f32,
    elapsed_time: f32,

    cursor_grabbed: bool,
    free_cam: bool,

    imgui_ctx:      Option<imgui::Context>,
    imgui_platform: Option<imgui_winit_support::WinitPlatform>,
    imgui_pipeline: Option<ImGuiPipeline>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            sky_mesh: None,
            cube_mesh: None,
            chunk_manager: None,
            player: None,
            camera: None,
            clock: None,
            terrain: None,
            model_y_offset: 0.0,
            keys_pressed: HashSet::new(),
            time_of_day: 8.0,
            time_paused: false,
            day_speed: 1.0,
            elapsed_time: 0.0,
            cursor_grabbed: true,
            free_cam: false,
            imgui_ctx: None,
            imgui_platform: None,
            imgui_pipeline: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_title("Orisha"))
            .unwrap();

        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked));
        window.set_cursor_visible(false);
        self.cursor_grabbed = true;

        let renderer = Renderer::new(&window).expect("Failed to initialize Vulkan renderer");

        let (vertices, indices, model_y_offset) = model_loader::load_glb("character/characte2.glb");
        let mesh = GpuMesh::upload(
            &renderer.device,
            &renderer.allocator,
            &renderer.commands,
            &vertices,
            &indices,
        );
        self.model_y_offset = model_y_offset;

        let (sky_verts, sky_idxs) = sky::sky_dome_geometry();
        let sky_gpu = GpuMesh::upload(
            &renderer.device,
            &renderer.allocator,
            &renderer.commands,
            &sky_verts,
            &sky_idxs,
        );
        let device_time: u32 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went before Unix epoch")
            .subsec_nanos() as u32;

        let terrain = Terrain::new(50.0,device_time);

        let mut chunk_mgr = ChunkManager::new();
        chunk_mgr.update(
            0.0, 0.0,
            &terrain,
            &renderer.device,
            &renderer.allocator,
            &renderer.commands,
        );

        let spawn_x = 0.0_f32;
        let spawn_z = 0.0_f32;
        let spawn_y = terrain.height_at(spawn_x, spawn_z) + 2.0;

        let mut imgui_ctx = imgui::Context::create();
        imgui_ctx.set_ini_filename(None);

        let mut platform = imgui_winit_support::WinitPlatform::new(&mut imgui_ctx);
        platform.attach_window(
            imgui_ctx.io_mut(),
            &window,
            imgui_winit_support::HiDpiMode::Default,
        );

        imgui_ctx.fonts().add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                size_pixels: 16.0,
                ..Default::default()
            }),
        }]);

        let imgui_pipeline = ImGuiPipeline::new(
            &renderer.device,
            &renderer.allocator,
            &renderer.commands,
            &renderer.swapchain,
            &mut imgui_ctx,
        )
        .expect("Failed to create ImGui pipeline");

        self.sky_mesh = Some(sky_gpu);
        self.cube_mesh = Some(mesh);
        self.chunk_manager = Some(chunk_mgr);
        self.window = Some(window);
        self.renderer = Some(renderer);

        self.player = Some(Player::new(Vec3::new(spawn_x, spawn_y, spawn_z)));
        self.camera = Some(Camera::new_third_person(Vec3::ZERO));
        self.clock = Some(GameClock::new(1.0 / 60.0));
        self.terrain = Some(terrain);

        self.imgui_ctx = Some(imgui_ctx);
        self.imgui_platform = Some(platform);
        self.imgui_pipeline = Some(imgui_pipeline);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(platform), Some(ctx), Some(window)) =
            (&mut self.imgui_platform, &mut self.imgui_ctx, &self.window)
        {
            let wrapped: winit::event::Event<()> = winit::event::Event::WindowEvent {
                window_id: window.id(),
                event: event.clone(),
            };
            platform.handle_event(ctx.io_mut(), window, &wrapped);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { event, .. } => {
                let imgui_wants_kb = self
                    .imgui_ctx
                    .as_ref()
                    .is_some_and(|ctx| ctx.io().want_capture_keyboard);

                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state.is_pressed() {
                        if code == KeyCode::Escape {
                            event_loop.exit();
                            return;
                        }
                        if code == KeyCode::Tab {
                            self.cursor_grabbed = !self.cursor_grabbed;
                            if let Some(w) = &self.window {
                                if self.cursor_grabbed {
                                    let _ = w.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                                        .or_else(|_| w.set_cursor_grab(winit::window::CursorGrabMode::Locked));
                                    w.set_cursor_visible(false);
                                } else {
                                    let _ = w.set_cursor_grab(winit::window::CursorGrabMode::None);
                                    w.set_cursor_visible(true);
                                }
                            }
                        }
                        if code == KeyCode::F5 {
                            self.free_cam = !self.free_cam;
                            if let Some(camera) = &mut self.camera {
                                camera.mode = if self.free_cam {
                                    components::camera::CameraMode::Free
                                } else {
                                    components::camera::CameraMode::ThirdPerson
                                };
                            }
                        }
                        if !imgui_wants_kb {
                            if code == KeyCode::Space {
                                if let Some(player) = &mut self.player {
                                    player.jump();
                                }
                            }
                            self.keys_pressed.insert(code);
                        }
                    } else {
                        self.keys_pressed.remove(&code);
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let imgui_wants_mouse = self
                    .imgui_ctx
                    .as_ref()
                    .is_some_and(|ctx| ctx.io().want_capture_mouse);

                if !imgui_wants_mouse {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                    };
                    if let Some(camera) = &mut self.camera {
                        camera.zoom(scroll);
                    }
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
                    self.elapsed_time += clock.delta;
                }

                if let (Some(player), Some(camera), Some(clock), Some(terrain)) =
                    (&mut self.player, &mut self.camera, &mut self.clock, &self.terrain)
                {
                    if self.free_cam {
                        let speed = if self.keys_pressed.contains(&KeyCode::ShiftLeft) { 200.0 } else { 60.0 };
                        let dt = clock.delta;

                        let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
                        let (sin_pitch, cos_pitch) = camera.pitch.sin_cos();
                        let forward = Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch).normalize();
                        let right = Vec3::new(cos_yaw, 0.0, -sin_yaw);

                        let mut fly_dir = Vec3::ZERO;
                        if self.keys_pressed.contains(&KeyCode::KeyW) { fly_dir += forward; }
                        if self.keys_pressed.contains(&KeyCode::KeyS) { fly_dir -= forward; }
                        if self.keys_pressed.contains(&KeyCode::KeyA) { fly_dir -= right; }
                        if self.keys_pressed.contains(&KeyCode::KeyD) { fly_dir += right; }
                        if self.keys_pressed.contains(&KeyCode::Space) { fly_dir += Vec3::Y; }
                        if self.keys_pressed.contains(&KeyCode::ControlLeft) { fly_dir -= Vec3::Y; }

                        if fly_dir.length_squared() > 0.0 {
                            camera.position += fly_dir.normalize() * speed * dt;
                        }

                        player.move_direction(Vec3::ZERO, false);
                        while clock.should_fixed_update() {
                            player.update(clock.fixed_step, terrain);
                        }
                    } else {
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
                        player.move_direction(move_dir, sprinting);

                        if move_dir.length_squared() < 0.01 {
                            player.rotation = glam::Quat::from_rotation_y(camera.yaw + std::f32::consts::PI);
                        }

                        while clock.should_fixed_update() {
                            player.update(clock.fixed_step, terrain);
                        }

                        camera.follow_target(player.shoulder_position(), clock.delta);

                        let terrain_at_cam = terrain.height_at(camera.position.x, camera.position.z);
                        let min_cam_y = terrain_at_cam + 0.5;
                        if camera.position.y < min_cam_y {
                            camera.position.y = min_cam_y;
                        }
                    }
                }

                if let (Some(chunk_mgr), Some(terrain), Some(renderer), Some(player), Some(camera)) = (
                    &mut self.chunk_manager,
                    &self.terrain,
                    &self.renderer,
                    &self.player,
                    &self.camera,
                ) {
                    let (cx, cz) = if self.free_cam {
                        (camera.position.x, camera.position.z)
                    } else {
                        (player.position.x, player.position.z)
                    };
                    chunk_mgr.update(
                        cx, cz,
                        terrain,
                        &renderer.device,
                        &renderer.allocator,
                        &renderer.commands,
                    );
                }


                if let (Some(platform), Some(ctx), Some(window)) =
                    (&self.imgui_platform, &mut self.imgui_ctx, &self.window)
                {
                    platform
                        .prepare_frame(ctx.io_mut(), window)
                        .expect("Failed to prepare ImGui frame");
                }

                let player_pos_text = self.player.as_ref().map(|p| {
                    format!("Player: ({:.1}, {:.1}, {:.1})", p.position.x, p.position.y, p.position.z)
                });

                let dt_text = self.clock.as_ref().map(|c| {
                    format!("FPS: {:.0} ({:.1}ms)", 1.0 / c.delta, c.delta * 1000.0)
                });

                if !self.time_paused {
                    let dt = self.clock.as_ref().map(|c| c.delta).unwrap_or(0.0);
                    self.time_of_day = (self.time_of_day + dt * self.day_speed / 60.0) % 24.0;
                }
                let tod = self.time_of_day;
                let tod_text = format!("Time: {:02}:{:02}", tod as u32, ((tod.fract()) * 60.0) as u32);

                if let (Some(ctx), Some(clock)) = (&mut self.imgui_ctx, &self.clock) {
                    ctx.io_mut()
                        .update_delta_time(std::time::Duration::from_secs_f32(clock.delta));
                }

                let draw_data_ptr: *const imgui::DrawData;

                if let (Some(ctx), Some(platform), Some(window)) =
                    (&mut self.imgui_ctx, &mut self.imgui_platform, &self.window)
                {
                    {
                        let ui = ctx.frame();


                        ui.window("Debug")
                            .size([300.0, 200.0], imgui::Condition::FirstUseEver)
                            .build(|| {
                                ui.text("Orisha Engine");
                                ui.separator();
                                if let Some(text) = &player_pos_text {
                                    ui.text(text);
                                }
                                if let Some(text) = &dt_text {
                                    ui.text(text);
                                }
                                ui.text(&tod_text);
                                ui.separator();
                                if self.free_cam {
                                    ui.text_colored([0.3, 1.0, 0.3, 1.0], "FREE CAM (F5)");
                                } else {
                                    ui.text("Player Cam (F5 to toggle)");
                                }
                                ui.separator();
                                ui.text("Day / Night Cycle");
                                let mut tod_val = tod;
                                let mut paused = self.time_paused;
                                let mut speed  = self.day_speed;
                                ui.slider_config("Hour", 0.0_f32, 23.99_f32)
                                    .display_format("%.1f")
                                    .build(&mut tod_val);
                                ui.checkbox("Pause time", &mut paused);
                                ui.slider_config("Speed", 0.1_f32, 30.0_f32)
                                    .display_format("%.1fx")
                                    .build(&mut speed);
                                if (tod_val - tod).abs() > 0.001 {
                                    self.time_of_day = tod_val;
                                }
                                self.time_paused = paused;
                                self.day_speed = speed;
                            });


                        platform.prepare_render(ui, window);
                    }
                    draw_data_ptr = ctx.render() as *const imgui::DrawData;
                } else {
                    draw_data_ptr = std::ptr::null();
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
                        view:    camera.view_matrix().to_cols_array_2d(),
                        proj:    camera.projection_matrix(aspect).to_cols_array_2d(),
                        cam_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
                    };

                    let foot = player.collider.foot_offset();
                    let player_push = PushConstants {
                        model: (Mat4::from_translation(player.position)
                            * Mat4::from_quat(player.rotation)
                            * Mat4::from_translation(Vec3::new(0.0, self.model_y_offset - foot, 0.0)))
                            .to_cols_array_2d(),
                        tex_blend: 0.0,
                        time: self.elapsed_time,
                    };

                    let terrain_push = PushConstants {
                        model:     Mat4::IDENTITY.to_cols_array_2d(),
                        tex_blend: 1.0,
                        time: self.elapsed_time,
                    };

                    let water_push = PushConstants {
                        model:     Mat4::IDENTITY.to_cols_array_2d(),
                        tex_blend: 2.0,
                        time: self.elapsed_time,
                    };

                    let sky_ref = &self.sky_mesh;
                    let chunk_ref = &self.chunk_manager;
                    let imgui_pipe = &mut self.imgui_pipeline;
                    let fb_width  = extent.width  as f32;
                    let fb_height = extent.height as f32;
                    let frame_idx = renderer.frame_index;

                    let time_of_day = self.time_of_day;
                    let sun_dir = sky::sun_direction(time_of_day);

                    let sky_push = SkyPushConstants {
                        sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                        time_of_day,
                        _pad0: 0.0,
                        _pad1: 0.0,
                        _pad2: 0.0,
                    };

                    let sky_pipeline_handle = renderer.sky_pipeline.pipeline;
                    let sky_layout_handle   = renderer.sky_pipeline.pipeline_layout;
                    let desc_set            = renderer.descriptors.camera_sets
                        [renderer.frame_index % gpu::commands::MAX_FRAMES_IN_FLIGHT];

                    let device_ptr: *const gpu::device::Device = &renderer.device;
                    let alloc_ptr:  *const gpu::allocator::GpuAllocator = &renderer.allocator;

                    renderer.draw_frame(&camera_ubo, |device, cmd, layout| unsafe {
                        device.cmd_push_constants(
                            cmd, layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::bytes_of(&terrain_push),
                        );
                        if let Some(cm) = chunk_ref {
                            for tmesh in cm.meshes() {
                                device.cmd_bind_vertex_buffers(cmd, 0, &[tmesh.vertex_buffer.buffer], &[0]);
                                device.cmd_bind_index_buffer(cmd, tmesh.index_buffer.buffer, 0, vk::IndexType::UINT32);
                                device.cmd_draw_indexed(cmd, tmesh.index_count, 1, 0, 0, 0);
                            }
                        }

                        device.cmd_push_constants(
                            cmd, layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::bytes_of(&water_push),
                        );
                        if let Some(cm) = chunk_ref {
                            for wmesh in cm.water_meshes() {
                                device.cmd_bind_vertex_buffers(cmd, 0, &[wmesh.vertex_buffer.buffer], &[0]);
                                device.cmd_bind_index_buffer(cmd, wmesh.index_buffer.buffer, 0, vk::IndexType::UINT32);
                                device.cmd_draw_indexed(cmd, wmesh.index_count, 1, 0, 0, 0);
                            }
                        }

                        device.cmd_push_constants(
                            cmd, layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::bytes_of(&player_push),
                        );
                        device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer.buffer], &[0]);
                        device.cmd_bind_index_buffer(cmd, mesh.index_buffer.buffer, 0, vk::IndexType::UINT32);
                        device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);

                        if let Some(smesh) = sky_ref {
                            device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                sky_pipeline_handle,
                            );
                            device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                sky_layout_handle,
                                0,
                                &[desc_set],
                                &[],
                            );
                            device.cmd_set_viewport(cmd, 0, &[vk::Viewport {
                                x: 0.0, y: 0.0,
                                width: fb_width, height: fb_height,
                                min_depth: 0.0, max_depth: 1.0,
                            }]);
                            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: vk::Extent2D {
                                    width: fb_width as u32,
                                    height: fb_height as u32,
                                },
                            }]);
                            device.cmd_push_constants(
                                cmd, sky_layout_handle,
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                0,
                                bytemuck::bytes_of(&sky_push),
                            );
                            device.cmd_bind_vertex_buffers(cmd, 0, &[smesh.vertex_buffer.buffer], &[0]);
                            device.cmd_bind_index_buffer(cmd, smesh.index_buffer.buffer, 0, vk::IndexType::UINT32);
                            device.cmd_draw_indexed(cmd, smesh.index_count, 1, 0, 0, 0);
                        }

                        if !draw_data_ptr.is_null() {
                            if let Some(imgui_p) = imgui_pipe {
                                let draw_data = &*draw_data_ptr;
                                imgui_p.record_commands(
                                    &*device_ptr,
                                    &*alloc_ptr,
                                    cmd,
                                    draw_data,
                                    fb_width,
                                    fb_height,
                                    frame_idx,
                                );
                            }
                        }
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
            if self.cursor_grabbed {
                if let Some(camera) = &mut self.camera {
                    camera.rotate(dx as f32, dy as f32);
                }
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(renderer) = &self.renderer {
            unsafe { let _ = renderer.device.device.device_wait_idle(); }
        }
        if let (Some(mut imgui_p), Some(renderer)) = (self.imgui_pipeline.take(), &self.renderer) {
            imgui_p.destroy(&renderer.device, &renderer.allocator);
        }
        if let (Some(mut cm), Some(renderer)) = (self.chunk_manager.take(), &self.renderer) {
            cm.destroy_all(&renderer.device, &renderer.allocator);
        }
        if let (Some(mesh), Some(renderer)) = (self.sky_mesh.take(), &self.renderer) {
            mesh.destroy(&renderer.device, &renderer.allocator);
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
