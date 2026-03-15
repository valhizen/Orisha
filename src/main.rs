use ash::vk::{self, ComponentTypeNV};
use std::time::{SystemTime, UNIX_EPOCH};
use glam::{Mat4, Vec3};
use std::collections::HashSet;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

mod components;
mod game;
mod gpu;

use components::camera::Camera;
use components::interact::Interact;
use game::chunk_manager::ChunkManager;
use game::model_loader;
use game::player::Player;
use game::sky;
use game::time::GameClock;
use game::world_generation::Terrain;
use gpu::buffer::{CameraUbo, GpuMesh, PushConstants, SkyPushConstants};
use gpu::pipeline_imgui::ImGuiPipeline;
use gpu::renderer::Renderer;

/// Main application state.
///
/// Most fields are wrapped in `Option` because the app object is created first,
/// and the real window/Vulkan resources are created later in `resumed()`.
struct App {
    /// Sky dome mesh stored on the GPU.
    sky_mesh: Option<GpuMesh>,
    /// Character mesh stored on the GPU.
    cube_mesh: Option<GpuMesh>,
    /// Handles loading/unloading nearby terrain chunks.
    chunk_manager: Option<ChunkManager>,
    /// High-level Vulkan renderer.
    renderer: Option<Renderer>,
    /// OS window managed by winit.
    window: Option<Window>,

    /// Player gameplay state.
    player: Option<Player>,
    /// Active camera.
    camera: Option<Camera>,
    /// Frame timing and fixed-step timing.
    clock: Option<GameClock>,
    /// Procedural terrain generator.
    terrain: Option<Terrain>,
    ///Dialogue System
     dialogue : Option<Interact>,
    /// Used to place the loaded character so its feet sit on the ground.
    model_y_offset: f32,
    /// Tracks which keyboard keys are currently held down.
    keys_pressed: HashSet<KeyCode>,

    /// Current time in the day/night cycle, in hours [0, 24).
    time_of_day: f32,
    /// If true, the day/night clock stops advancing.
    time_paused: bool,
    /// Speed multiplier for the day/night cycle.
    day_speed:   f32,
    /// Total elapsed gameplay time used by shaders/animation.
    elapsed_time: f32,

    /// Whether the mouse is currently captured by the window.
    cursor_grabbed: bool,
    /// Whether the camera is in free-fly mode instead of player-follow mode.
    free_cam: bool,

    /// ImGui context and platform integration state.
    imgui_ctx:      Option<imgui::Context>,
    imgui_platform: Option<imgui_winit_support::WinitPlatform>,
    imgui_pipeline: Option<ImGuiPipeline>,
}

impl App {
    /// Creates the app object in an empty state.
    ///
    /// The real runtime setup happens later in `resumed()`, when winit gives
    /// us an active event loop and allows window creation.
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
            dialogue: None,
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
    /// Called by winit when the application becomes active.
    ///
    /// This is where the real startup happens:
    /// - create the window
    /// - create Vulkan renderer
    /// - load meshes
    /// - build terrain/chunks
    /// - create player/camera/clock
    /// - initialize ImGui
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            // Will later Add a icon
            .create_window(Window::default_attributes().with_title("Aelkyn").with_window_icon(None))
            .unwrap();

        // Capture and hide the cursor for camera control.
        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked));
        window.set_cursor_visible(false);
        self.cursor_grabbed = true;

        // Create the Vulkan renderer after the window exists.
        let renderer = Renderer::new(&window).expect("Failed to initialize Vulkan renderer");

        // Load the character mesh from the GLB file and upload it to the GPU.
        let (vertices, indices, model_y_offset) = model_loader::load_glb("character/characte2.glb");
        let mesh = GpuMesh::upload(
            &renderer.device,
            &renderer.allocator,
            &renderer.commands,
            &vertices,
            &indices,
        );
        self.model_y_offset = model_y_offset;

        // Build and upload the sky dome mesh.
        let (sky_verts, sky_idxs) = sky::sky_dome_geometry();
        let sky_gpu = GpuMesh::upload(
            &renderer.device,
            &renderer.allocator,
            &renderer.commands,
            &sky_verts,
            &sky_idxs,
        );

        // Use current time as a quick random seed for the terrain.
        let device_time: u32 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went before Unix epoch")
            .subsec_nanos() as u32;

        let terrain = Terrain::new(100000.0,0);

        // Generate the first set of nearby chunks around the origin.
        let mut chunk_mgr = ChunkManager::new();
        chunk_mgr.update(
            0.0, 0.0,
            &terrain,
            &renderer.device,
            &renderer.allocator,
            &renderer.commands,
        );

        // Spawn the player slightly above the terrain so it can fall onto the ground.
        let spawn_x = 0.0_f32;
        let spawn_z = 0.0_f32;
        let spawn_y = terrain.height_at(spawn_x, spawn_z) + 2.0;

        // Set up Dear ImGui.
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

        // Store GPU/runtime resources inside the app state.
        self.sky_mesh = Some(sky_gpu);
        self.cube_mesh = Some(mesh);
        self.chunk_manager = Some(chunk_mgr);
        self.window = Some(window);
        self.renderer = Some(renderer);

        // Create gameplay systems.
        self.player = Some(Player::new(Vec3::new(spawn_x, spawn_y, spawn_z)));
        self.camera = Some(Camera::new_third_person(Vec3::ZERO));
        self.clock = Some(GameClock::new(1.0 / 60.0));
        self.terrain = Some(terrain);
        self.dialogue =Some(Interact::new("THis is the Dialogue".to_string()));

        self.imgui_ctx = Some(imgui_ctx);
        self.imgui_platform = Some(platform);
        self.imgui_pipeline = Some(imgui_pipeline);
    }

    /// Called by winit for window-related events such as input, resize, and redraw.
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let ImGui inspect the event first so it can capture keyboard/mouse when needed.
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

            // Handle keyboard input for gameplay and app controls.
            WindowEvent::KeyboardInput { event, .. } => {
                let imgui_wants_kb = self
                    .imgui_ctx
                    .as_ref()
                    .is_some_and(|ctx| ctx.io().want_capture_keyboard);

                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state.is_pressed() {
                        // Escape closes the application.
                        if code == KeyCode::Escape {
                            event_loop.exit();
                            return;
                        }

                        // if code == KeyCode::KeyE  {
                        //     if let Some(dialogue) =
                        //     &self.dialogue {
                        //         dialogue.show_dialogue();
                        //     }};
                        if code == KeyCode::KeyE && event.state.is_pressed() {
                            if let Some(dialogue) = &mut self.dialogue {
                                dialogue.toggle();
                            }
                        }

                        // Tab toggles mouse capture.
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

                        // F5 switches between free camera and player-follow camera.
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

                        // Only forward input to gameplay if ImGui is not using the keyboard.
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

            // Mouse wheel controls third-person camera zoom.
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

            // Recreate size-dependent renderer resources when the window changes size.
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    if let Some(r) = &mut self.renderer {
                        r.resize(size.width, size.height);
                    }
                }
            }

            // This is the real frame loop:
            // update time, update gameplay, build UI, then render.
            WindowEvent::RedrawRequested => {
                if let Some(clock) = &mut self.clock {
                    clock.tick();
                    self.elapsed_time += clock.delta;
                }

                // Update player and camera behavior.
                if let (Some(player), Some(camera), Some(clock), Some(terrain)) =
                    (&mut self.player, &mut self.camera, &mut self.clock, &self.terrain)
                {
                    if self.free_cam {
                        // Free camera movement uses the camera orientation directly.
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

                        // Player still runs fixed-step physics so it stays valid on terrain.
                        player.move_direction(Vec3::ZERO, false);
                        while clock.should_fixed_update() {
                            player.update(clock.fixed_step, terrain);
                        }
                    } else {
                        // Build player movement from WASD input.
                        let mut move_dir = Vec3::ZERO;
                        if self.keys_pressed.contains(&KeyCode::KeyW) { move_dir.z -= 1.0; }
                        if self.keys_pressed.contains(&KeyCode::KeyS) { move_dir.z += 1.0; }
                        if self.keys_pressed.contains(&KeyCode::KeyA) { move_dir.x -= 1.0; }
                        if self.keys_pressed.contains(&KeyCode::KeyD) { move_dir.x += 1.0; }

                        // Rotate movement from local camera space into world space.
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

                        // If the player is idle, keep the character facing the camera direction.
                        if move_dir.length_squared() < 0.01 {
                            player.rotation = glam::Quat::from_rotation_y(camera.yaw + std::f32::consts::PI);
                        }

                        // Physics runs at a fixed time step.
                        while clock.should_fixed_update() {
                            player.update(clock.fixed_step, terrain);
                        }

                        // Third-person camera follows the player smoothly.
                        camera.follow_target(player.shoulder_position(), clock.delta);

                        // Prevent the camera from going below the terrain.
                        let terrain_at_cam = terrain.height_at(camera.position.x, camera.position.z);
                        let min_cam_y = terrain_at_cam + 0.5;
                        if camera.position.y < min_cam_y {
                            camera.position.y = min_cam_y;
                        }
                    }
                }

                // Stream terrain chunks around the current player/camera location.
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


                // Start a new ImGui frame.
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

                // Advance the in-game time unless the debug UI paused it.
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

                // Build the debug UI and keep a pointer to its draw data for later rendering.
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

                                // Push UI values back into the game state.
                                if (tod_val - tod).abs() > 0.001 {
                                    self.time_of_day = tod_val;
                                }
                                self.time_paused = paused;
                                self.day_speed = speed;
                            });


                        if let Some(dialogue) = &self.dialogue {
                            if dialogue.visible {
                                let [screen_w, screen_h] = ui.io().display_size;

                                ui.window("Dialogue")
                                    .position([screen_w * 0.5 - 300.0, screen_h - 180.0], imgui::Condition::Always)
                                    .size([600.0, 140.0], imgui::Condition::Always)
                                    .movable(false)
                                    .resizable(false)
                                    .collapsible(false)
                                    .title_bar(false)
                                    .build(|| {
                                        ui.text("Narrator");
                                        ui.separator();
                                        ui.text_wrapped(&dialogue.dialogue);
                                        ui.separator();
                                        ui.text("Press E to close");
                                    });
                            }
                        }

                        platform.prepare_render(ui, window);
                    }
                    draw_data_ptr = ctx.render() as *const imgui::DrawData;
                } else {
                    draw_data_ptr = std::ptr::null();
                }


                // Prepare GPU data and submit one rendered frame.
                if let (Some(renderer), Some(mesh), Some(camera), Some(player)) = (
                    &mut self.renderer,
                    &self.cube_mesh,
                    &self.camera,
                    &self.player,
                ) {
                    let extent = renderer.swapchain.surface_resolution;
                    let aspect = extent.width as f32 / extent.height.max(1) as f32;

                    // Camera matrices uploaded every frame.
                    let camera_ubo = CameraUbo {
                        view:    camera.view_matrix().to_cols_array_2d(),
                        proj:    camera.projection_matrix(aspect).to_cols_array_2d(),
                        cam_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
                    };

                    // Push constants for the player mesh.
                    let foot = player.collider.foot_offset();
                    let player_push = PushConstants {
                        model: (Mat4::from_translation(player.position)
                            * Mat4::from_quat(player.rotation)
                            * Mat4::from_translation(Vec3::new(0.0, self.model_y_offset - foot, 0.0)))
                            .to_cols_array_2d(),
                        tex_blend: 0.0,
                        time: self.elapsed_time,
                    };

                    // Push constants for terrain and water.
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

                    // Build sky shader parameters from the current time of day.
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
                        // Draw terrain chunks.
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

                        // Draw water meshes.
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

                        // Draw the character model.
                        device.cmd_push_constants(
                            cmd, layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::bytes_of(&player_push),
                        );
                        device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer.buffer], &[0]);
                        device.cmd_bind_index_buffer(cmd, mesh.index_buffer.buffer, 0, vk::IndexType::UINT32);
                        device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);

                        // Draw the sky with its own pipeline.
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



                        // Draw ImGui last as an overlay.
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

                // Ask for another redraw so the app keeps running like a game loop.
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }

    /// Called for raw device events such as mouse motion.
    ///
    /// Mouse motion is used to rotate the camera while the cursor is captured.
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
    /// Manual cleanup for resources owned directly by `App`.
    ///
    /// Most GPU-related global objects are cleaned up by `Renderer`, but meshes,
    /// chunks, and ImGui pipeline objects owned here are destroyed in this order.
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
    // Create the event loop. After `run_app`, winit takes control and starts
    // calling the `ApplicationHandler` methods such as `resumed()` and `window_event()`.
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    // Build the app object in an empty state.
    let mut app = App::new();

    // Start the real application lifecycle.
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("Event loop error: {e}");
        std::process::exit(1);
    }
}
