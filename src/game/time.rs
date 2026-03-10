// Frame timing and fixed-step accumulator.
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

    pub fn tick(&mut self) {
        let now = std::time::Instant::now();
        let raw = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.real_delta = raw.min(0.25);
        self.delta = self.real_delta * self.time_scale;
        self.elapsed += self.delta;
        self.accumulator += self.delta;
    }

    pub fn should_fixed_update(&mut self) -> bool {
        if self.accumulator >= self.fixed_step {
            self.accumulator -= self.fixed_step;
            true
        } else {
            false
        }
    }

    pub fn real_elapsed(&self) -> f32 {
        self.start.elapsed().as_secs_f32()
    }

    pub fn pause(&mut self)  { self.time_scale = 0.0; }
    pub fn resume(&mut self) { self.time_scale = 1.0; }
    pub fn is_paused(&self) -> bool { self.time_scale == 0.0 }
}
