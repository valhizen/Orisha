pub struct Interact {
    pub dialogue: String,
    pub visible: bool,
}

impl Interact {
    pub fn new(dialogue: String) -> Self {
        Self {
            dialogue,
            visible: false,
        }
    }

    pub fn open(&mut self) {
        self.visible = true;
    }

    pub fn close(&mut self) {
        self.visible = false;
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }
}
