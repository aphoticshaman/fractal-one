//! ═══════════════════════════════════════════════════════════════════════════════
//! SITREP — Real-Time Situational Awareness
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Answers the question: "Where am I, right now?"
//!
//! Collects real system metrics and runs them through the sensorium to produce
//! actionable situational awareness. Not a demo—actual data, actual signals.
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::thermoception::RawSignal;
use crate::{
    drift::{baseline_exists, load_baseline, DriftDetector, DriftResult},
    observations::{ObsKey, ObservationBatch},
    time::TimePoint,
    BehaviorHook, DisorientationLevel, IntegratedState, Nociceptor, NociceptorConfig, PainSignal,
    PainType, Sensorium, SensoriumConfig, ThermalState, Thermoceptor, ThermoceptorConfig,
};
use std::time::Duration;
use sysinfo::{CpuRefreshKind, Disks, MemoryRefreshKind, RefreshKind, System};

/// Real-time system sitrep
pub struct Sitrep {
    sys: System,
    disks: Disks,
    sensorium: Sensorium,
    thermoceptor: Thermoceptor,
    nociceptor: Nociceptor,
    sample_count: u64,
}

/// Snapshot of system state
#[derive(Debug)]
pub struct SystemSnapshot {
    pub cpu_percent: f64,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub memory_percent: f64,
    pub disk_used_gb: f64,
    pub disk_total_gb: f64,
    pub disk_percent: f64,
    pub process_count: usize,
    pub uptime_secs: u64,
}

/// Complete sitrep output
#[derive(Debug)]
pub struct SitrepOutput {
    pub timestamp: TimePoint,
    pub system: SystemSnapshot,
    pub thermal_state: ThermalState,
    pub thermal_utilization: f64,
    pub pain_count: usize,
    pub pain_total: f64,
    pub disorientation: f64,
    pub disorientation_level: DisorientationLevel,
    pub sensorium_state: IntegratedState,
    pub behavior_hooks: Vec<BehaviorHook>,
    pub summary: String,
    /// Drift from baseline (if baseline exists)
    pub drift: Option<DriftResult>,
}

impl Sitrep {
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        let disks = Disks::new_with_refreshed_list();

        Self {
            sys,
            disks,
            sensorium: Sensorium::new(SensoriumConfig::default()),
            thermoceptor: Thermoceptor::new(ThermoceptorConfig::default()),
            nociceptor: Nociceptor::new(NociceptorConfig::default()),
            sample_count: 0,
        }
    }

    /// Collect current system metrics
    fn collect_system(&mut self) -> SystemSnapshot {
        self.sys.refresh_all();
        self.disks.refresh();

        let cpu_percent = self.sys.global_cpu_info().cpu_usage() as f64;

        let memory_used = self.sys.used_memory();
        let memory_total = self.sys.total_memory();
        let memory_used_gb = memory_used as f64 / 1_073_741_824.0;
        let memory_total_gb = memory_total as f64 / 1_073_741_824.0;
        let memory_percent = if memory_total > 0 {
            (memory_used as f64 / memory_total as f64) * 100.0
        } else {
            0.0
        };

        let (disk_used, disk_total) = self.disks.iter().fold((0u64, 0u64), |acc, d| {
            (
                acc.0 + (d.total_space() - d.available_space()),
                acc.1 + d.total_space(),
            )
        });
        let disk_used_gb = disk_used as f64 / 1_073_741_824.0;
        let disk_total_gb = disk_total as f64 / 1_073_741_824.0;
        let disk_percent = if disk_total > 0 {
            (disk_used as f64 / disk_total as f64) * 100.0
        } else {
            0.0
        };

        let process_count = self.sys.processes().len();
        let uptime_secs = System::uptime();

        SystemSnapshot {
            cpu_percent,
            memory_used_gb,
            memory_total_gb,
            memory_percent,
            disk_used_gb,
            disk_total_gb,
            disk_percent,
            process_count,
            uptime_secs,
        }
    }

    /// Convert system snapshot to thermal signals
    fn snapshot_to_thermal_signals(&self, snap: &SystemSnapshot) -> Vec<RawSignal> {
        let mut signals = Vec::new();

        // CPU load → Latency proxy (higher CPU = slower responses)
        let latency_ms = (snap.cpu_percent * 10.0) as u64; // Scale to ms
        signals.push(RawSignal::Latency(Duration::from_millis(latency_ms)));

        // Memory pressure → Context utilization
        signals.push(RawSignal::ContextUtilization(
            (snap.memory_percent / 100.0) as f32,
        ));

        // Process count → Complexity
        let complexity = (snap.process_count as f32 / 500.0).min(1.0);
        signals.push(RawSignal::QueryComplexity(complexity));

        // If high disk usage, add error signal
        if snap.disk_percent > 90.0 {
            signals.push(RawSignal::ErrorCount(
                ((snap.disk_percent - 90.0) / 2.0) as u32,
            ));
        }

        signals
    }

    /// Convert system snapshot to observations for sensorium
    fn snapshot_to_observations(&self, snap: &SystemSnapshot) -> ObservationBatch {
        let mut batch = ObservationBatch::new();

        // Response latency proxy
        batch.add(ObsKey::RespLatMs, snap.cpu_percent * 10.0);

        // Context utilization
        batch.add(ObsKey::CtxUtilization, snap.memory_percent / 100.0);

        // Network proxy (use disk as external resource proxy)
        batch.add(ObsKey::NetRttMs, snap.disk_percent);

        // Thermal utilization
        batch.add(ObsKey::ThermalUtilization, snap.cpu_percent / 100.0);

        batch
    }

    /// Run a single sitrep cycle
    pub fn run(&mut self) -> SitrepOutput {
        let timestamp = TimePoint::now();
        let snap = self.collect_system();

        // Feed to thermoceptor
        let thermal_signals = self.snapshot_to_thermal_signals(&snap);
        let thermal_map = self.thermoceptor.ingest(&thermal_signals);

        // Check if thermal state should trigger pain
        if let Some(triggers) = self.thermoceptor.check_pain_trigger() {
            for (zone, util, duration) in triggers {
                self.nociceptor.feel(PainSignal::new(
                    PainType::ThermalOverheat {
                        zone: zone.clone(),
                        utilization: util,
                        duration_secs: duration,
                        is_redlining: util > 0.95,
                    },
                    util,
                    &zone,
                ));
            }
        }

        // Feed to sensorium
        let obs = self.snapshot_to_observations(&snap);
        self.sensorium.ingest_batch(obs);
        let integration = self.sensorium.integrate();

        self.sample_count += 1;

        // Build pain description
        let damage_state = self.nociceptor.damage_state();
        let pain_count = if self.nociceptor.in_pain() { 1 } else { 0 };
        let pain_total = damage_state.total;

        // Get vestibular state from integration (sensorium has internal vestibular)
        let disorientation = integration.severity; // Use severity as disorientation proxy
        let disorientation_level = match disorientation {
            d if d >= 0.9 => DisorientationLevel::Critical,
            d if d >= 0.7 => DisorientationLevel::Severe,
            d if d >= 0.5 => DisorientationLevel::Moderate,
            d if d >= 0.3 => DisorientationLevel::Mild,
            _ => DisorientationLevel::Stable,
        };

        // Compute drift if baseline exists
        let drift = if baseline_exists() {
            if let Ok(baseline) = load_baseline() {
                let mut drift_detector = DriftDetector::new();
                Some(drift_detector.compare(&baseline))
            } else {
                None
            }
        } else {
            None
        };

        // Generate summary
        let summary = self.generate_summary(
            &snap,
            &thermal_map.global_state,
            disorientation_level,
            &integration.state,
            &drift,
        );

        SitrepOutput {
            timestamp,
            system: snap,
            thermal_state: thermal_map.global_state,
            thermal_utilization: thermal_map.global_utilization as f64,
            pain_count,
            pain_total: pain_total as f64, // Convert f32 to f64
            disorientation,
            disorientation_level,
            sensorium_state: integration.state,
            behavior_hooks: integration.hooks,
            summary,
            drift,
        }
    }

    /// Run calibration (multiple samples to establish baselines)
    pub fn calibrate(&mut self, samples: usize) {
        println!("Calibrating sensorium ({} samples)...", samples);
        for i in 0..samples {
            self.run();
            if i < samples - 1 {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
        println!(
            "Calibration complete. {} samples collected.",
            self.sample_count
        );
    }

    fn generate_summary(
        &self,
        snap: &SystemSnapshot,
        thermal: &ThermalState,
        disorientation: DisorientationLevel,
        state: &IntegratedState,
        drift: &Option<DriftResult>,
    ) -> String {
        let mut parts = Vec::new();

        // System health
        let sys_health = if snap.cpu_percent > 90.0 || snap.memory_percent > 90.0 {
            "STRESSED"
        } else if snap.cpu_percent > 70.0 || snap.memory_percent > 70.0 {
            "BUSY"
        } else {
            "NOMINAL"
        };
        parts.push(format!("System: {}", sys_health));

        // Thermal
        let thermal_str = match thermal {
            ThermalState::Nominal => "NOMINAL",
            ThermalState::Elevated => "ELEVATED",
            ThermalState::Saturated => "SATURATED",
            ThermalState::Unsafe => "UNSAFE",
        };
        parts.push(format!("Thermal: {}", thermal_str));

        // Orientation
        let orient_str = match disorientation {
            DisorientationLevel::Stable => "ORIENTED",
            DisorientationLevel::Mild => "SLIGHTLY_DISORIENTED",
            DisorientationLevel::Moderate => "DISORIENTED",
            DisorientationLevel::Severe => "VERY_DISORIENTED",
            DisorientationLevel::Critical => "LOST",
        };
        parts.push(format!("Orientation: {}", orient_str));

        // Sensorium state
        let state_str = match state {
            IntegratedState::Calm => "CALM",
            IntegratedState::Alert => "ALERT",
            IntegratedState::Degraded => "DEGRADED",
            IntegratedState::Crisis => "CRISIS",
        };
        parts.push(format!("State: {}", state_str));

        // Drift
        if let Some(d) = drift {
            parts.push(format!("Drift: {}", d.status.as_str()));
        }

        parts.join(" | ")
    }
}

impl Default for Sitrep {
    fn default() -> Self {
        Self::new()
    }
}

/// Print formatted sitrep to stdout
pub fn print_sitrep(output: &SitrepOutput) {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                              SITREP");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Timestamp: {:?}", output.timestamp);
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SYSTEM METRICS                                                              │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ CPU:      {:5.1}%                                                           │",
        output.system.cpu_percent
    );
    println!(
        "│ Memory:   {:5.1}% ({:.1} / {:.1} GB)                                       │",
        output.system.memory_percent, output.system.memory_used_gb, output.system.memory_total_gb
    );
    println!(
        "│ Disk:     {:5.1}% ({:.0} / {:.0} GB)                                         │",
        output.system.disk_percent, output.system.disk_used_gb, output.system.disk_total_gb
    );
    println!(
        "│ Procs:    {:4}                                                              │",
        output.system.process_count
    );
    println!(
        "│ Uptime:   {:4} hours                                                        │",
        output.system.uptime_secs / 3600
    );
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SENSORY STATE                                                               │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ Thermal State:     {:?}  (util: {:.2})                                     │",
        output.thermal_state, output.thermal_utilization
    );
    println!(
        "│ Disorientation:    {:?}  ({:.2})                                         │",
        output.disorientation_level, output.disorientation
    );
    println!(
        "│ Pain Signals:      {}  (total: {:.2})                                       │",
        output.pain_count, output.pain_total
    );
    println!(
        "│ Sensorium State:   {:?}                                                    │",
        output.sensorium_state
    );
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
    if !output.behavior_hooks.is_empty()
        && output
            .behavior_hooks
            .iter()
            .any(|h| *h != BehaviorHook::None)
    {
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ BEHAVIOR HOOKS ACTIVE                                                       │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        for hook in &output.behavior_hooks {
            if *hook != BehaviorHook::None {
                println!(
                    "│  • {:?}                                                                │",
                    hook
                );
            }
        }
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();
    }
    // Show drift status if available
    if let Some(ref drift) = output.drift {
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ BASELINE DRIFT                                                              │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "│ CPU Drift:       {:.2}    Memory Drift:    {:.2}                              │",
            drift.cpu_drift, drift.memory_drift
        );
        println!(
            "│ Disk Drift:      {:.2}    Process Drift:   {:.2}                              │",
            drift.disk_drift, drift.process_drift
        );
        println!(
            "│ Aggregate:       {:.2}  ({})                                              │",
            drift.aggregate,
            drift.status.as_str()
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();
    }
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  {}", output.summary);
    println!("═══════════════════════════════════════════════════════════════════════════════");
}

/// Run interactive sitrep (continuous monitoring)
pub fn run_continuous(interval_ms: u64) {
    let mut sitrep = Sitrep::new();

    // Quick calibration
    sitrep.calibrate(25);
    println!();

    loop {
        // Clear screen (works on most terminals)
        print!("\x1B[2J\x1B[1;1H");

        let output = sitrep.run();
        print_sitrep(&output);

        println!(
            "\n  Press Ctrl+C to exit. Refreshing every {}ms...",
            interval_ms
        );
        std::thread::sleep(std::time::Duration::from_millis(interval_ms));
    }
}

/// Run single sitrep snapshot
pub fn run_once() {
    let mut sitrep = Sitrep::new();

    // Minimal calibration for baseline
    sitrep.calibrate(25);
    println!();

    let output = sitrep.run();
    print_sitrep(&output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sitrep_basic() {
        let mut sitrep = Sitrep::new();
        let output = sitrep.run();

        // Should have valid system metrics
        assert!(output.system.cpu_percent >= 0.0);
        assert!(output.system.memory_total_gb > 0.0);
        assert!(!output.summary.is_empty());
    }

    #[test]
    fn test_sitrep_calibration() {
        let mut sitrep = Sitrep::new();
        sitrep.calibrate(5);
        assert!(sitrep.sample_count >= 5);
    }

    #[test]
    fn test_snapshot_to_observations() {
        let sitrep = Sitrep::new();
        let snap = SystemSnapshot {
            cpu_percent: 50.0,
            memory_used_gb: 8.0,
            memory_total_gb: 16.0,
            memory_percent: 50.0,
            disk_used_gb: 200.0,
            disk_total_gb: 500.0,
            disk_percent: 40.0,
            process_count: 150,
            uptime_secs: 3600,
        };

        let obs = sitrep.snapshot_to_observations(&snap);
        assert!(!obs.observations.is_empty());
    }
}
