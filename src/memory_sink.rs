//! ═══════════════════════════════════════════════════════════════════════════════
//! MEMORY_SINK — Telemetry Persistence
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::neuro_link::Synapse;
use anyhow::Result;
use std::{collections::VecDeque, fs::OpenOptions, io::Write, thread, time::Duration};

const BUFFER_CAPACITY: usize = 100;
const LAMINAR_THRESHOLD: f64 = 0.005;

pub async fn run() -> Result<()> {
    println!("\x1b[35m[MEMORY]  TELEMETRY SINK ACTIVE\x1b[0m");

    let mut synapse = Synapse::connect(false);
    let mut vault: VecDeque<String> = VecDeque::with_capacity(BUFFER_CAPACITY);

    loop {
        if synapse.check_kill_signal() {
            println!("\x1b[31m[MEMORY] Shutdown. Final Flush...\x1b[0m");
            flush_to_disk(&mut vault);
            println!("\x1b[32m[MEMORY] Persistence Secured.\x1b[0m");
            break;
        }

        if let Some(pulse) = synapse.sense() {
            let entry = format!(
                "{{\"id\":{},\"j\":{:.6},\"l\":{:.2},\"d\":{:.6}}}\n",
                pulse.id, pulse.jitter_ms, pulse.cpu_load_percent, pulse.entropy_damping
            );
            vault.push_back(entry);

            let is_full = vault.len() >= BUFFER_CAPACITY;
            let is_laminar = pulse.jitter_ms < LAMINAR_THRESHOLD && pulse.cpu_load_percent < 40.0;

            if is_full || (is_laminar && !vault.is_empty()) {
                if is_full {
                    println!("\x1b[33m[MEMORY] Buffer full. Flush.\x1b[0m");
                }
                flush_to_disk(&mut vault);
            }
        }
        thread::sleep(Duration::from_millis(10));
    }
    Ok(())
}

fn flush_to_disk(vault: &mut VecDeque<String>) {
    if vault.is_empty() {
        return;
    }

    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open("archon_journal.jsonl")
    {
        Ok(f) => f,
        Err(_) => return,
    };

    while let Some(line) = vault.pop_front() {
        let _ = file.write_all(line.as_bytes());
    }
    let _ = file.flush();
}
