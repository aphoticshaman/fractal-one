//! ═══════════════════════════════════════════════════════════════════════════════
//! CORTEX — Real-Time Health Monitor
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Consumes telemetry pulses and displays system health status. Uses rolling
//! jitter average over 10 samples to assess temporal stability.
//!
//! ## Limitations
//!
//! - **Display only**: No alerting, logging, or remediation actions.
//! - **Fixed thresholds**: PERFECT (<0.05ms) and NOMINAL are hardcoded.
//! - **10-sample window**: May miss transient spikes or react slowly to
//!   sustained degradation.
//! - **Single metric focus**: Jitter-centric; doesn't synthesize other signals.

use crate::neuro_link::Synapse;
use anyhow::Result;
use std::{collections::VecDeque, thread, time::Duration};

pub async fn run() -> Result<()> {
    println!("[CORTEX]  GROUNDED MONITOR ACTIVE");
    let mut synapse = loop {
        if let Ok(s) = std::panic::catch_unwind(|| Synapse::connect(false)) {
            break s;
        }
        thread::sleep(Duration::from_millis(100));
    };
    let mut jitter_history: VecDeque<f64> = VecDeque::with_capacity(10);
    loop {
        if synapse.check_kill_signal() {
            break;
        }
        if let Some(pulse) = synapse.sense() {
            jitter_history.push_back(pulse.jitter_ms);
            if jitter_history.len() > 10 {
                jitter_history.pop_front();
            }
            let avg_jitter: f64 = jitter_history.iter().sum::<f64>() / jitter_history.len() as f64;
            let status = if avg_jitter < 0.05 {
                "\x1b[32mPERFECT\x1b[0m"
            } else {
                "\x1b[33mNOMINAL\x1b[0m"
            };
            println!(
                "[CORTEX]  P#{} | Jitter: {:.4}ms | CPU: {:.1}% | Status: {}",
                pulse.id, pulse.jitter_ms, pulse.cpu_load_percent, status
            );
        }
        thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}
