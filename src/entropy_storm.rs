//! ═══════════════════════════════════════════════════════════════════════════════
//! ENTROPY_STORM — Chaos Generator for Stress Testing
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::neuro_link::Synapse;
use std::{
    hint,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

const THREAD_COUNT: usize = 4;
const ARRAY_SIZE: usize = 100 * 1024 * 1024; // 100MB

pub fn run() -> anyhow::Result<()> {
    println!("[CHAOS] INITIALIZING ENTROPY STORM...");

    let running = Arc::new(AtomicBool::new(true));
    let synapse = Synapse::connect(false);

    for i in 0..THREAD_COUNT {
        let r = running.clone();
        thread::spawn(move || {
            let mut memory_hog = vec![0u8; ARRAY_SIZE];
            let mut seed = (i as u32 + 1) * 123456789;
            while r.load(Ordering::Relaxed) {
                for _ in 0..1000 {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    let idx = (seed as usize) % ARRAY_SIZE;
                    memory_hog[idx] = memory_hog[idx].wrapping_add(1);
                }
                let mut x: f64 = 1.0;
                for _ in 0..1000 {
                    x = x.sin() * x.cos();
                }
                hint::black_box(x);
                thread::yield_now();
            }
        });
    }

    println!(
        "[CHAOS] {} threads spawned, consuming {}MB each",
        THREAD_COUNT,
        ARRAY_SIZE / (1024 * 1024)
    );

    loop {
        if synapse.check_kill_signal() {
            running.store(false, Ordering::SeqCst);
            println!("[CHAOS] Shutdown signal received");
            break;
        }
        thread::sleep(Duration::from_secs(1));
    }

    Ok(())
}
