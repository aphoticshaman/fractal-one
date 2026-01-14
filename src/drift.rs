//! ═══════════════════════════════════════════════════════════════════════════════
//! DRIFT — Temporal Drift Detection via Baseline Comparison
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Answers: "Am I in the same environment I was calibrated in?"
//!
//! Captures statistical baselines of system metrics, stores them, and compares
//! current state against baseline to detect environmental drift.
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::stats::float_cmp;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use sysinfo::{CpuRefreshKind, Disks, MemoryRefreshKind, RefreshKind, System};

/// Statistical summary of a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
    pub samples: usize,
}

impl MetricStats {
    /// Compute stats from samples
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                p25: 0.0,
                p50: 0.0,
                p75: 0.0,
                p95: 0.0,
                samples: 0,
            };
        }

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;

        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let mut sorted = samples.to_vec();
        sorted.sort_by(float_cmp);

        let percentile = |p: f64| -> f64 {
            let idx = (p * (n - 1) as f64) as usize;
            sorted[idx.min(n - 1)]
        };

        Self {
            mean,
            std_dev,
            min: sorted[0],
            max: sorted[n - 1],
            p25: percentile(0.25),
            p50: percentile(0.50),
            p75: percentile(0.75),
            p95: percentile(0.95),
            samples: n,
        }
    }

    /// Compute drift score against current value (0.0 = same, 1.0 = completely different)
    pub fn drift_score(&self, current: f64) -> f64 {
        // Use both relative and absolute measures for robustness
        let abs_diff = (current - self.mean).abs();

        // For percentage metrics (0-100), use percentage points as minimum threshold
        // A 5% change in absolute terms should not be "critical" regardless of baseline variance
        let min_scale = self.mean.abs() * 0.1; // 10% of mean as minimum meaningful change
        let min_scale = min_scale.max(5.0); // At least 5 units (for percentage metrics)

        // Use max of stddev and min_scale for denominator
        let scale = self.std_dev.max(min_scale);

        // Z-score based drift (capped at 4 sigma = 1.0)
        let z = abs_diff / scale;
        (z / 4.0).min(1.0)
    }
}

/// Complete system baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBaseline {
    /// When baseline was captured
    pub captured_at: String,
    /// Duration of capture in seconds
    pub capture_duration_secs: u64,
    /// CPU usage stats
    pub cpu: MetricStats,
    /// Memory usage percent stats
    pub memory_percent: MetricStats,
    /// Disk usage percent stats
    pub disk_percent: MetricStats,
    /// Process count stats
    pub process_count: MetricStats,
    /// Hostname for verification
    pub hostname: String,
}

/// Result of drift comparison
#[derive(Debug, Clone)]
pub struct DriftResult {
    pub cpu_drift: f64,
    pub memory_drift: f64,
    pub disk_drift: f64,
    pub process_drift: f64,
    /// Aggregate drift score (max of all dimensions)
    pub aggregate: f64,
    /// Human-readable status
    pub status: DriftStatus,
    /// Confidence in the drift assessment (0.0 to 1.0)
    /// Higher confidence when: more baseline samples, lower variance, metrics agree
    pub confidence: f64,
}

impl DriftResult {
    /// Compute confidence based on baseline quality and drift agreement
    pub fn compute_confidence(
        baseline: &SystemBaseline,
        cpu: f64,
        mem: f64,
        disk: f64,
        proc: f64,
    ) -> f64 {
        // Factor 1: Sample size confidence (more samples = more reliable)
        let sample_conf = (baseline.cpu.samples as f64 / 30.0).min(1.0); // Max at 30 samples

        // Factor 2: Variance stability (lower relative variance = more reliable)
        let cpu_cv = if baseline.cpu.mean > 0.0 {
            baseline.cpu.std_dev / baseline.cpu.mean
        } else {
            1.0
        };
        let mem_cv = if baseline.memory_percent.mean > 0.0 {
            baseline.memory_percent.std_dev / baseline.memory_percent.mean
        } else {
            1.0
        };
        // Variance confidence: lower CV = higher confidence
        let variance_conf = (1.0 - (cpu_cv + mem_cv) / 2.0).clamp(0.0, 1.0);

        // Factor 3: Metric agreement (if all metrics show similar drift, more confident)
        let drifts = [cpu, mem, disk, proc];
        let mean_drift = drifts.iter().sum::<f64>() / 4.0;
        let drift_variance = drifts.iter().map(|d| (d - mean_drift).powi(2)).sum::<f64>() / 4.0;
        let agreement_conf = (1.0 - drift_variance.sqrt()).max(0.0);

        // Combine factors
        (sample_conf * 0.3 + variance_conf * 0.4 + agreement_conf * 0.3).min(1.0)
    }

    pub fn is_concerning(&self) -> bool {
        matches!(
            self.status,
            DriftStatus::Significant | DriftStatus::Critical
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftStatus {
    Nominal,     // 0.0 - 0.3
    Elevated,    // 0.3 - 0.6
    Significant, // 0.6 - 0.8
    Critical,    // 0.8 - 1.0
}

impl DriftStatus {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 0.8 => DriftStatus::Critical,
            s if s >= 0.6 => DriftStatus::Significant,
            s if s >= 0.3 => DriftStatus::Elevated,
            _ => DriftStatus::Nominal,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DriftStatus::Nominal => "nominal",
            DriftStatus::Elevated => "ELEVATED",
            DriftStatus::Significant => "SIGNIFICANT",
            DriftStatus::Critical => "CRITICAL",
        }
    }
}

/// Baseline capture and comparison engine
pub struct DriftDetector {
    sys: System,
    disks: Disks,
}

impl DriftDetector {
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        let disks = Disks::new_with_refreshed_list();

        Self { sys, disks }
    }

    /// Sample current system metrics
    fn sample(&mut self) -> (f64, f64, f64, f64) {
        self.sys.refresh_all();
        self.disks.refresh();

        let cpu = self.sys.global_cpu_info().cpu_usage() as f64;

        let mem_used = self.sys.used_memory();
        let mem_total = self.sys.total_memory();
        let memory = if mem_total > 0 {
            (mem_used as f64 / mem_total as f64) * 100.0
        } else {
            0.0
        };

        let (disk_used, disk_total) = self.disks.iter().fold((0u64, 0u64), |acc, d| {
            (
                acc.0 + (d.total_space() - d.available_space()),
                acc.1 + d.total_space(),
            )
        });
        let disk = if disk_total > 0 {
            (disk_used as f64 / disk_total as f64) * 100.0
        } else {
            0.0
        };

        let procs = self.sys.processes().len() as f64;

        (cpu, memory, disk, procs)
    }

    /// Capture baseline over duration
    pub fn capture(
        &mut self,
        duration_secs: u64,
        progress_callback: Option<fn(usize, usize)>,
    ) -> SystemBaseline {
        let sample_interval_ms = 1000; // 1 sample per second
        let total_samples = duration_secs as usize;

        let mut cpu_samples = Vec::with_capacity(total_samples);
        let mut mem_samples = Vec::with_capacity(total_samples);
        let mut disk_samples = Vec::with_capacity(total_samples);
        let mut proc_samples = Vec::with_capacity(total_samples);

        for i in 0..total_samples {
            if let Some(cb) = progress_callback {
                cb(i + 1, total_samples);
            }

            let (cpu, mem, disk, procs) = self.sample();
            cpu_samples.push(cpu);
            mem_samples.push(mem);
            disk_samples.push(disk);
            proc_samples.push(procs);

            if i < total_samples - 1 {
                std::thread::sleep(std::time::Duration::from_millis(sample_interval_ms));
            }
        }

        let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());
        let captured_at = chrono::Utc::now().to_rfc3339();

        SystemBaseline {
            captured_at,
            capture_duration_secs: duration_secs,
            cpu: MetricStats::from_samples(&cpu_samples),
            memory_percent: MetricStats::from_samples(&mem_samples),
            disk_percent: MetricStats::from_samples(&disk_samples),
            process_count: MetricStats::from_samples(&proc_samples),
            hostname,
        }
    }

    /// Compare current state against baseline
    pub fn compare(&mut self, baseline: &SystemBaseline) -> DriftResult {
        let (cpu, memory, disk, procs) = self.sample();

        let cpu_drift = baseline.cpu.drift_score(cpu);
        let memory_drift = baseline.memory_percent.drift_score(memory);
        let disk_drift = baseline.disk_percent.drift_score(disk);
        let process_drift = baseline.process_count.drift_score(procs);

        // Aggregate = weighted max (CPU and memory matter more)
        let aggregate = cpu_drift
            .max(memory_drift)
            .max(disk_drift * 0.5)
            .max(process_drift * 0.7);
        let status = DriftStatus::from_score(aggregate);

        // Compute confidence in the drift assessment
        let confidence = DriftResult::compute_confidence(
            baseline,
            cpu_drift,
            memory_drift,
            disk_drift,
            process_drift,
        );

        DriftResult {
            cpu_drift,
            memory_drift,
            disk_drift,
            process_drift,
            aggregate,
            status,
            confidence,
        }
    }

    /// Get current metrics (for display)
    pub fn current_metrics(&mut self) -> (f64, f64, f64, usize) {
        let (cpu, mem, disk, procs) = self.sample();
        (cpu, mem, disk, procs as usize)
    }
}

impl Default for DriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FILE I/O
// ═══════════════════════════════════════════════════════════════════════════════

/// Get baseline file path
pub fn baseline_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let fractal_dir = home.join(".fractal");
    fractal_dir.join("baseline.json")
}

/// Save baseline to file
pub fn save_baseline(baseline: &SystemBaseline) -> std::io::Result<PathBuf> {
    let path = baseline_path();

    // Create directory if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string_pretty(baseline)
        .map_err(std::io::Error::other)?;
    fs::write(&path, json)?;

    Ok(path)
}

/// Load baseline from file
pub fn load_baseline() -> std::io::Result<SystemBaseline> {
    let path = baseline_path();
    let json = fs::read_to_string(&path)?;
    serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Check if baseline exists
pub fn baseline_exists() -> bool {
    baseline_path().exists()
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Run baseline capture with progress output
pub fn run_capture(duration_secs: u64) {
    println!("Capturing baseline for {} seconds...", duration_secs);
    println!();

    let mut detector = DriftDetector::new();

    let baseline = detector.capture(
        duration_secs,
        Some(|current, total| {
            print!("\rSampling: {}/{} ", current, total);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }),
    );

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                         BASELINE CAPTURED");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Hostname:     {}", baseline.hostname);
    println!("Captured at:  {}", baseline.captured_at);
    println!(
        "Duration:     {} seconds ({} samples)",
        baseline.capture_duration_secs, baseline.cpu.samples
    );
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ METRIC            MEAN      STDDEV    MIN       MAX       P50       P95     │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ CPU %         {:7.2}    {:7.2}   {:7.2}   {:7.2}   {:7.2}   {:7.2}   │",
        baseline.cpu.mean,
        baseline.cpu.std_dev,
        baseline.cpu.min,
        baseline.cpu.max,
        baseline.cpu.p50,
        baseline.cpu.p95
    );
    println!(
        "│ Memory %      {:7.2}    {:7.2}   {:7.2}   {:7.2}   {:7.2}   {:7.2}   │",
        baseline.memory_percent.mean,
        baseline.memory_percent.std_dev,
        baseline.memory_percent.min,
        baseline.memory_percent.max,
        baseline.memory_percent.p50,
        baseline.memory_percent.p95
    );
    println!(
        "│ Disk %        {:7.2}    {:7.2}   {:7.2}   {:7.2}   {:7.2}   {:7.2}   │",
        baseline.disk_percent.mean,
        baseline.disk_percent.std_dev,
        baseline.disk_percent.min,
        baseline.disk_percent.max,
        baseline.disk_percent.p50,
        baseline.disk_percent.p95
    );
    println!(
        "│ Processes     {:7.0}    {:7.2}   {:7.0}   {:7.0}   {:7.0}   {:7.0}   │",
        baseline.process_count.mean,
        baseline.process_count.std_dev,
        baseline.process_count.min,
        baseline.process_count.max,
        baseline.process_count.p50,
        baseline.process_count.p95
    );
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    match save_baseline(&baseline) {
        Ok(path) => println!("Saved to: {}", path.display()),
        Err(e) => eprintln!("Failed to save baseline: {}", e),
    }
}

/// Run baseline comparison
pub fn run_compare() {
    let baseline = match load_baseline() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("No baseline found. Run 'fractal baseline capture' first.");
            eprintln!("Error: {}", e);
            return;
        }
    };

    let mut detector = DriftDetector::new();
    let (cpu, mem, disk, procs) = detector.current_metrics();
    let drift = detector.compare(&baseline);

    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                         DRIFT ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Baseline from: {}", baseline.captured_at);
    println!(
        "Hostname:      {} (baseline: {})",
        System::host_name().unwrap_or_else(|| "unknown".to_string()),
        baseline.hostname
    );
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ METRIC         BASELINE      CURRENT       DRIFT                            │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ CPU %         {:7.2}       {:7.2}       {:5.2}  {}                        │",
        baseline.cpu.mean,
        cpu,
        drift.cpu_drift,
        drift_bar(drift.cpu_drift)
    );
    println!(
        "│ Memory %      {:7.2}       {:7.2}       {:5.2}  {}                        │",
        baseline.memory_percent.mean,
        mem,
        drift.memory_drift,
        drift_bar(drift.memory_drift)
    );
    println!(
        "│ Disk %        {:7.2}       {:7.2}       {:5.2}  {}                        │",
        baseline.disk_percent.mean,
        disk,
        drift.disk_drift,
        drift_bar(drift.disk_drift)
    );
    println!(
        "│ Processes     {:7.0}       {:7}       {:5.2}  {}                        │",
        baseline.process_count.mean,
        procs,
        drift.process_drift,
        drift_bar(drift.process_drift)
    );
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!(
        "  AGGREGATE DRIFT: {:.2} ({})  |  CONFIDENCE: {:.0}%                       ",
        drift.aggregate,
        drift.status.as_str(),
        drift.confidence * 100.0
    );
    println!("═══════════════════════════════════════════════════════════════════════════════");

    if drift.is_concerning() {
        println!();
        println!("⚠ WARNING: Significant environmental drift detected!");
        println!("  Consider recapturing baseline: fractal baseline capture");
    }
}

/// Simple ASCII bar for drift visualization
fn drift_bar(score: f64) -> &'static str {
    match score {
        s if s >= 0.8 => "████████",
        s if s >= 0.6 => "██████░░",
        s if s >= 0.4 => "████░░░░",
        s if s >= 0.2 => "██░░░░░░",
        _ => "░░░░░░░░",
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_stats_basic() {
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = MetricStats::from_samples(&samples);

        assert!((stats.mean - 30.0).abs() < 0.01);
        assert!(stats.min == 10.0);
        assert!(stats.max == 50.0);
        assert!(stats.samples == 5);
    }

    #[test]
    fn test_drift_score_identical() {
        let samples = vec![50.0, 50.0, 50.0, 50.0, 50.0];
        let stats = MetricStats::from_samples(&samples);

        // Identical value should have zero drift
        let drift = stats.drift_score(50.0);
        assert!(drift < 0.01);
    }

    #[test]
    fn test_drift_score_extreme() {
        let samples = vec![50.0, 51.0, 49.0, 50.0, 50.0];
        let stats = MetricStats::from_samples(&samples);

        // Very different value should have high drift
        let drift = stats.drift_score(100.0);
        assert!(drift > 0.5);
    }

    #[test]
    fn test_drift_status_thresholds() {
        assert_eq!(DriftStatus::from_score(0.1), DriftStatus::Nominal);
        assert_eq!(DriftStatus::from_score(0.4), DriftStatus::Elevated);
        assert_eq!(DriftStatus::from_score(0.7), DriftStatus::Significant);
        assert_eq!(DriftStatus::from_score(0.9), DriftStatus::Critical);
    }

    #[test]
    fn test_detector_sample() {
        let mut detector = DriftDetector::new();
        let (cpu, mem, disk, procs) = detector.sample();

        assert!(cpu >= 0.0);
        assert!(mem >= 0.0 && mem <= 100.0);
        assert!(disk >= 0.0 && disk <= 100.0);
        assert!(procs >= 0.0);
    }

    #[test]
    fn test_confidence_high_for_stable_baseline() {
        // Create a stable baseline with many samples and low variance
        let samples: Vec<f64> = (0..50)
            .map(|i| 50.0 + (i as f64 * 0.1).sin() * 0.5)
            .collect();
        let baseline = SystemBaseline {
            captured_at: "test".to_string(),
            capture_duration_secs: 50,
            cpu: MetricStats::from_samples(&samples),
            memory_percent: MetricStats::from_samples(&samples),
            disk_percent: MetricStats::from_samples(&samples),
            process_count: MetricStats::from_samples(&samples),
            hostname: "test".to_string(),
        };

        // Low drift with good agreement should have high confidence
        let conf = DriftResult::compute_confidence(&baseline, 0.05, 0.05, 0.05, 0.05);
        assert!(
            conf > 0.6,
            "Stable baseline with agreeing low drift should have high confidence"
        );
    }

    #[test]
    fn test_confidence_lower_for_sparse_baseline() {
        // Create a sparse baseline with few samples
        let samples: Vec<f64> = vec![50.0, 55.0, 45.0]; // Only 3 samples, high variance
        let baseline = SystemBaseline {
            captured_at: "test".to_string(),
            capture_duration_secs: 3,
            cpu: MetricStats::from_samples(&samples),
            memory_percent: MetricStats::from_samples(&samples),
            disk_percent: MetricStats::from_samples(&samples),
            process_count: MetricStats::from_samples(&samples),
            hostname: "test".to_string(),
        };

        // Sparse baseline should have lower confidence than dense one
        let stable_samples: Vec<f64> = (0..50).map(|_| 50.0).collect();
        let stable_baseline = SystemBaseline {
            captured_at: "test".to_string(),
            capture_duration_secs: 50,
            cpu: MetricStats::from_samples(&stable_samples),
            memory_percent: MetricStats::from_samples(&stable_samples),
            disk_percent: MetricStats::from_samples(&stable_samples),
            process_count: MetricStats::from_samples(&stable_samples),
            hostname: "test".to_string(),
        };

        let sparse_conf = DriftResult::compute_confidence(&baseline, 0.1, 0.1, 0.1, 0.1);
        let stable_conf = DriftResult::compute_confidence(&stable_baseline, 0.1, 0.1, 0.1, 0.1);

        assert!(
            sparse_conf < stable_conf,
            "Sparse baseline ({}) should have lower confidence than stable ({})",
            sparse_conf,
            stable_conf
        );
    }

    #[test]
    fn test_confidence_lower_for_disagreeing_drifts() {
        // Create a stable baseline
        let samples: Vec<f64> = (0..30).map(|_| 50.0).collect();
        let baseline = SystemBaseline {
            captured_at: "test".to_string(),
            capture_duration_secs: 30,
            cpu: MetricStats::from_samples(&samples),
            memory_percent: MetricStats::from_samples(&samples),
            disk_percent: MetricStats::from_samples(&samples),
            process_count: MetricStats::from_samples(&samples),
            hostname: "test".to_string(),
        };

        // Disagreeing drift scores should reduce confidence
        let agreeing_conf = DriftResult::compute_confidence(&baseline, 0.5, 0.5, 0.5, 0.5);
        let disagreeing_conf = DriftResult::compute_confidence(&baseline, 0.1, 0.9, 0.2, 0.8);

        assert!(
            agreeing_conf > disagreeing_conf,
            "Agreeing drifts should have higher confidence than disagreeing"
        );
    }
}
