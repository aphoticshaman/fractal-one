//! ═══════════════════════════════════════════════════════════════════════════════
//! HEART — Core Timing Loop
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::neuro_link::{Pulse, Synapse};
use anyhow::Result;
use std::{
    hint,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use sysinfo::{CpuRefreshKind, ProcessRefreshKind, RefreshKind, System};
use thread_priority::*;

struct SharedPerception {
    cpu_load: AtomicU64,
    bad_actor_pid: AtomicU32,
    history: std::sync::Mutex<std::collections::VecDeque<f32>>,
}

pub async fn run() -> Result<()> {
    println!("\x1b[31m[HEART]   IGNITING DAMPED PACER CORE...\x1b[0m");
    let synapse = Arc::new(std::sync::Mutex::new(Synapse::connect(true)));
    let perception = Arc::new(SharedPerception {
        cpu_load: AtomicU64::new(0),
        bad_actor_pid: AtomicU32::new(0),
        history: std::sync::Mutex::new(std::collections::VecDeque::from(vec![0.0; 32])),
    });

    // --- SYSTEM INTROSPECTION THREAD ---
    let sense_perception = perception.clone();
    thread::spawn(move || {
        let mut sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_processes(ProcessRefreshKind::everything()),
        );
        loop {
            sys.refresh_cpu();
            sys.refresh_processes();
            sense_perception
                .cpu_load
                .store(sys.global_cpu_info().cpu_usage() as u64, Ordering::Relaxed);

            let pid = sys
                .processes()
                .iter()
                .max_by(|a, b| a.1.cpu_usage().partial_cmp(&b.1.cpu_usage()).unwrap())
                .map(|(p, _)| p.as_u32())
                .unwrap_or(0);
            sense_perception.bad_actor_pid.store(pid, Ordering::Relaxed);

            thread::sleep(Duration::from_millis(100));
        }
    });

    // --- THE PACER LOOP (HARD REAL-TIME) ---
    let pacer_synapse = synapse.clone();
    let pacer_perception = perception.clone();
    let pacer_handle = thread::spawn(move || {
        // Set Thread Priority to Max to minimize OS interference
        let _ = set_current_thread_priority(ThreadPriority::Max);

        let mut syn = pacer_synapse.lock().unwrap();
        let mut current_interval_ms = syn.get_target_interval();
        let mut next_wake = Instant::now();
        let mut last_jitter_ms = 0.0;
        let mut id = 0;

        loop {
            if syn.check_kill_signal() {
                break;
            }

            // Check for Sovereign Commands or Stress Triggers
            let target_ms = syn.get_target_interval();

            if target_ms == 999 {
                // TRIGGER: NOISE BURST (Simulate 0.1ms/100us jitter)
                let burst_start = Instant::now();
                while burst_start.elapsed().as_micros() < 100 {
                    hint::spin_loop();
                }
                // Return to original rhythm immediately
                syn.set_target_interval(current_interval_ms);
                println!("\x1b[31m[HEART]   NOISE BURST EXECUTED (100us SPIN).\x1b[0m");
            } else if target_ms != current_interval_ms && target_ms != 0 {
                current_interval_ms = target_ms;
                println!(
                    "\x1b[32m[HEART]   RHYTHM CHANGE: {}ms\x1b[0m",
                    current_interval_ms
                );
                next_wake = Instant::now();
            }

            id += 1;
            next_wake += Duration::from_millis(current_interval_ms);

            // Precision Wait Logic
            let now = Instant::now();
            if next_wake > now {
                let rem = next_wake - now;
                // If remaining time > 15ms, sleep. Otherwise, spin.
                if rem.as_millis() > 15 {
                    thread::sleep(rem - Duration::from_millis(15));
                }
                while Instant::now() < next_wake {
                    hint::spin_loop();
                }
            } else {
                // We missed the window; reset pacer to 'now'
                next_wake = Instant::now();
            }

            let jitter_ms = Instant::now().duration_since(next_wake).as_secs_f64() * 1000.0;

            // CALCULATE DAMPING: Reacts to the CHANGE in jitter (Acceleration)
            let entropy_damping = ((jitter_ms - last_jitter_ms).abs() as f32) * 10.0;
            last_jitter_ms = jitter_ms;

            // Update Signal Payload
            let mut payload = [0.0f32; 32];
            {
                let mut hist = pacer_perception.history.lock().unwrap();
                hist.push_back(jitter_ms as f32);
                hist.pop_front();
                for (i, val) in hist.iter().enumerate() {
                    payload[i] = *val;
                }
            }

            // Emit the Pulse
            syn.fire(Pulse {
                id,
                telemetry_sequence: id,
                jitter_ms,
                cpu_load_percent: pacer_perception.cpu_load.load(Ordering::Relaxed) as f64,
                current_interval_ms,
                bad_actor_id: pacer_perception.bad_actor_pid.load(Ordering::Relaxed),
                entropy_damping,
                payload,
                scheduler_override: 0,
            });
        }
    });

    pacer_handle.join().unwrap();
    Ok(())
}
