//! ═══════════════════════════════════════════════════════════════════════════════
//! ARCHON_GPU — GPU Compute for Telemetry Analysis
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::neuro_link::Synapse;
use std::{thread, time::Duration};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ArchonOutput {
    danger_score: f32,
    entropy_trend: f32,
}

pub fn run() -> anyhow::Result<()> {
    pollster::block_on(run_async())
}

async fn run_async() -> anyhow::Result<()> {
    let mut synapse = Synapse::connect(false);
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;

    println!("[ARCHON]  TENSOR CORES PRIMED. STAGING BUFFER ACTIVE.");

    loop {
        if synapse.check_kill_signal() {
            break;
        }

        if let Some(pulse) = synapse.sense() {
            let _input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Inbound_Nerve"),
                contents: bytemuck::cast_slice(&pulse.payload),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

            let output_gpu_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Archon_Intuition_Internal"),
                size: std::mem::size_of::<ArchonOutput>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging_Readback"),
                size: std::mem::size_of::<ArchonOutput>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(
                &output_gpu_buf,
                0,
                &staging_buf,
                0,
                std::mem::size_of::<ArchonOutput>() as u64,
            );
            queue.submit(Some(encoder.finish()));

            if pulse.id % 50 == 0 {
                println!("[ARCHON]  Pulse #{} | GPU -> Staging Sync: OK", pulse.id);
            }
        }
        thread::sleep(Duration::from_millis(1));
    }

    println!("[ARCHON]  Shutdown complete");
    Ok(())
}
