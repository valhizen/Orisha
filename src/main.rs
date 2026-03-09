use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};


#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App{
    fn resumed(&mut self, event_loop : &ActiveEventLoop) {
        self.window = Some(event_loop.create_window(Window::default_attributes()).unwrap());
    }

    fn window_event(&mut self, event_loop : &ActiveEventLoop, _id:WindowId, event : WindowEvent){
        match event{
            WindowEvent::CloseRequested => {
                println!("The closed Button Was Pressed ; Stopping");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),

        }
    }

}

fn main(){

    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);


    let mut app = App::default();
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("Event loop error: {}", e);
        std::process::exit(1);
    }

}
