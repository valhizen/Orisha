/// Keeps track of frame time and fixed-step update time.
///
/// `delta` is the scaled frame time used by most gameplay code.
/// `fixed_step` is used for systems that should run at a stable rate,
/// such as physics.
pub struct GameClock {
    start: std::time::Instant,
    last_frame: std::time::Instant,
    pub delta: f32,
    pub elapsed: f32,
    pub time_scale: f32,
    pub real_delta: f32,
    pub fixed_step: f32,
    accumulator: f32,
}

impl GameClock {
    /// Creates a clock with a chosen fixed update step.
    ///
    /// Example: `1.0 / 60.0` means fixed updates try to run at 60 Hz.
    pub fn new(fixed_step: f32) -> Self {
        let now = std::time::Instant::now();
        Self {
            start: now,
            last_frame: now,
            delta: 0.0,
            elapsed: 0.0,
            time_scale: 1.0,
            real_delta: 0.0,
            fixed_step,
            accumulator: 0.0,
        }
    }

    /// Updates per-frame timing values.
    ///
    /// `real_delta` is the true frame time, clamped to avoid huge jumps.
    /// `delta` is the scaled frame time after applying `time_scale`.
    pub fn tick(&mut self) {
        let now = std::time::Instant::now();
        let raw = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Clamp long frames so gameplay does not explode after a stall.
        self.real_delta = raw.min(0.25);
        self.delta = self.real_delta * self.time_scale;
        self.elapsed += self.delta;
        self.accumulator += self.delta;
    }

    /// Returns true when enough time has built up for one fixed update.
    ///
    /// This is commonly used in a loop:
    /// while clock.should_fixed_update() { ... }
    pub fn should_fixed_update(&mut self) -> bool {
        if self.accumulator >= self.fixed_step {
            self.accumulator -= self.fixed_step;
            true
        } else {
            false
        }
    }

    /// Returns real wall-clock time since the clock was created.
    pub fn real_elapsed(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    /// Pauses scaled game time.
    pub fn pause(&mut self)  { self.time_scale = 0.0; }

    /// Resumes scaled game time.
    pub fn resume(&mut self) { self.time_scale = 1.0; }

    /// Returns whether scaled game time is paused.
    pub fn is_paused(&self) -> bool { self.time_scale == 0.0 }
}
