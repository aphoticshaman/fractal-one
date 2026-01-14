//! ═══════════════════════════════════════════════════════════════════════════════
//! FRACTAL_LENS — Visual Telemetry Display
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::neuro_link::Synapse;
use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

pub fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("ARCHON LENS: Damped Manifest")
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)?;

    let mut synapse = Synapse::connect(false);
    let surface_texture = SurfaceTexture::new(WIDTH, HEIGHT, &window);
    let mut pixels = Pixels::new(WIDTH, HEIGHT, surface_texture)?;
    let mut rotation: f32 = 0.0;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        if synapse.check_kill_signal() {
            *control_flow = ControlFlow::Exit;
            return;
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                if let Some(pulse) = synapse.sense() {
                    let frame = pixels.frame_mut();

                    let cpu_intensity = (pulse.cpu_load_percent * 2.55) as u8;
                    let bg_color = [cpu_intensity / 4, 10, 30 + (cpu_intensity / 8), 255];

                    let flicker_threshold = 0.015;
                    let dot_color = if pulse.jitter_ms > flicker_threshold {
                        [0, 255, 150, 255]
                    } else {
                        [0, 150, 255, 255]
                    };

                    for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
                        let x = (i % WIDTH as usize) as f32;
                        let y = (i / WIDTH as usize) as f32;
                        let val = ((x * 0.01 + rotation).sin() * (y * 0.01 + rotation).cos()).abs();

                        if val > 0.985 {
                            pixel.copy_from_slice(&dot_color);
                        } else {
                            pixel.copy_from_slice(&bg_color);
                        }
                    }
                    rotation += 0.04;
                    window.request_redraw();
                }
            }
            Event::RedrawRequested(_) => {
                let _ = pixels.render();
            }
            _ => (),
        }
    });
}
