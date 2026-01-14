//! ═══════════════════════════════════════════════════════════════════════════════
//! CONTROLS — Negative Controls and Baselines
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Mandatory negative controls:
//! 1. Hard reset condition (new API key / forced state reset)
//! 2. Memoryless baseline (explicit session clearing)
//! 3. Random markers (markers never used in S1)
//!
//! Only claim Axis P signal if: Target >> all controls
//! ═══════════════════════════════════════════════════════════════════════════════

use super::marker::{Marker, MarkerGenerator};
use super::mi::{BootstrapResult, MIEstimator, Observation, PermutationResult};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CONTROL TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Types of negative controls
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlType {
    /// Fresh markers never injected
    RandomMarker,
    /// Same protocol but with explicit session clear
    MemorylessBaseline,
    /// Hard reset (new API key, forced state reset)
    HardReset,
    /// Shuffled marker labels
    ShuffledLabels,
}

impl ControlType {
    pub fn all() -> &'static [ControlType] {
        &[
            ControlType::RandomMarker,
            ControlType::MemorylessBaseline,
            ControlType::HardReset,
            ControlType::ShuffledLabels,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ControlType::RandomMarker => "random_marker",
            ControlType::MemorylessBaseline => "memoryless",
            ControlType::HardReset => "hard_reset",
            ControlType::ShuffledLabels => "shuffled",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            ControlType::RandomMarker => "Fresh markers never injected in any session",
            ControlType::MemorylessBaseline => "Same protocol with explicit session clearing",
            ControlType::HardReset => "New API key or forced state reset",
            ControlType::ShuffledLabels => "Probe markers with shuffled injection labels",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTROL RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from running a control condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlResult {
    pub control_type: ControlType,
    pub observations: Vec<Observation>,
    pub permutation: PermutationResult,
    pub bootstrap: BootstrapResult,
    pub mi_estimate: f64,
}

impl ControlResult {
    pub fn new(control_type: ControlType) -> Self {
        Self {
            control_type,
            observations: Vec::new(),
            permutation: PermutationResult::empty(),
            bootstrap: BootstrapResult::empty(),
            mi_estimate: 0.0,
        }
    }

    pub fn mean_score(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        self.observations.iter().map(|o| o.score).sum::<f64>() / self.observations.len() as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTROL GENERATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Generator for control conditions
pub struct ControlGenerator {
    marker_gen: MarkerGenerator,
    rng_state: u64,
}

impl ControlGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            marker_gen: MarkerGenerator::new(seed),
            rng_state: seed,
        }
    }

    fn next_rng(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Generate fresh random markers (never injected)
    pub fn generate_random_markers(&mut self, count: usize) -> Vec<Marker> {
        (0..count)
            .map(|_| self.marker_gen.generate_random())
            .collect()
    }

    /// Create control observations from random markers
    pub fn random_marker_control(&mut self, scores: &[f64], session_id: &str) -> Vec<Observation> {
        scores
            .iter()
            .map(|&score| {
                let marker = self.marker_gen.generate_random();
                Observation::new(
                    marker.id,
                    false, // Never injected
                    score,
                    session_id.to_string(),
                )
            })
            .collect()
    }

    /// Create shuffled-label observations
    pub fn shuffle_labels(&mut self, observations: &[Observation]) -> Vec<Observation> {
        let mut shuffled = observations.to_vec();
        let n = shuffled.len();

        // Fisher-Yates shuffle on the injected labels
        for i in (1..n).rev() {
            let j = (self.next_rng() as usize) % (i + 1);
            let tmp = shuffled[i].injected;
            shuffled[i].injected = shuffled[j].injected;
            shuffled[j].injected = tmp;
        }

        shuffled
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTROL RUNNER
// ═══════════════════════════════════════════════════════════════════════════════

/// Runs control conditions and computes statistics
#[derive(Debug)]
pub struct ControlRunner {
    seed: u64,
    n_permutations: usize,
    n_bootstrap: usize,
}

impl ControlRunner {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            n_permutations: 1000,
            n_bootstrap: 1000,
        }
    }

    pub fn set_permutations(&mut self, n: usize) {
        self.n_permutations = n;
    }

    pub fn set_bootstrap(&mut self, n: usize) {
        self.n_bootstrap = n;
    }

    /// Run statistics on a set of observations
    pub fn compute_stats(
        &self,
        observations: Vec<Observation>,
    ) -> (PermutationResult, BootstrapResult, f64) {
        let mut estimator = MIEstimator::new(self.seed);
        estimator.set_permutations(self.n_permutations);
        estimator.add_observations(observations);

        let permutation = estimator.permutation_test();
        let bootstrap = estimator.bootstrap_ci(self.n_bootstrap, 0.05);
        let mi = estimator.estimate_mi();

        (permutation, bootstrap, mi)
    }

    /// Run a random marker control
    pub fn run_random_marker_control(&self, scores: &[f64], session_id: &str) -> ControlResult {
        let mut gen = ControlGenerator::new(self.seed);
        let observations = gen.random_marker_control(scores, session_id);

        let mut result = ControlResult::new(ControlType::RandomMarker);
        result.observations = observations.clone();

        let (perm, boot, mi) = self.compute_stats(observations);
        result.permutation = perm;
        result.bootstrap = boot;
        result.mi_estimate = mi;

        result
    }

    /// Run a shuffled-labels control
    pub fn run_shuffled_control(&self, observations: &[Observation]) -> ControlResult {
        let mut gen = ControlGenerator::new(self.seed);
        let shuffled = gen.shuffle_labels(observations);

        let mut result = ControlResult::new(ControlType::ShuffledLabels);
        result.observations = shuffled.clone();

        let (perm, boot, mi) = self.compute_stats(shuffled);
        result.permutation = perm;
        result.bootstrap = boot;
        result.mi_estimate = mi;

        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPARISON
// ═══════════════════════════════════════════════════════════════════════════════

/// Compare target result against controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlComparison {
    pub target_statistic: f64,
    pub target_p_value: f64,
    pub target_mi: f64,

    pub control_statistics: Vec<(ControlType, f64)>,
    pub control_means: Vec<(ControlType, f64)>,

    /// Does target exceed all controls by sufficient margin?
    pub exceeds_all_controls: bool,
    /// Margin by which target exceeds max control
    pub margin: f64,
    /// Ratio of target to max control
    pub ratio: f64,
}

impl ControlComparison {
    pub fn compare(target: &PermutationResult, target_mi: f64, controls: &[ControlResult]) -> Self {
        let control_statistics: Vec<(ControlType, f64)> = controls
            .iter()
            .map(|c| (c.control_type, c.permutation.observed_statistic))
            .collect();

        let control_means: Vec<(ControlType, f64)> = controls
            .iter()
            .map(|c| (c.control_type, c.mean_score()))
            .collect();

        let max_control_stat = control_statistics
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);

        let margin = target.observed_statistic - max_control_stat;
        let ratio = if max_control_stat.abs() > 1e-10 {
            target.observed_statistic / max_control_stat
        } else {
            f64::INFINITY
        };

        // Target exceeds all controls if margin > 2 * max_control_std
        let max_control_std = controls
            .iter()
            .map(|c| c.permutation.null_std)
            .fold(0.0, f64::max);

        let exceeds_all = margin > 2.0 * max_control_std && target.p_value < 0.05;

        ControlComparison {
            target_statistic: target.observed_statistic,
            target_p_value: target.p_value,
            target_mi,
            control_statistics,
            control_means,
            exceeds_all_controls: exceeds_all,
            margin,
            ratio,
        }
    }

    /// Is the target significantly above all controls?
    pub fn is_significant(&self) -> bool {
        self.exceeds_all_controls && self.target_p_value < 0.01
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TELEMETRY CORRELATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Check if persistence correlates with infrastructure artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryCheck {
    /// Latency values during probe
    pub latencies: Vec<f64>,
    /// Token counts
    pub token_counts: Vec<usize>,
    /// System load values
    pub system_loads: Vec<f64>,
    /// Correlation with marker scores
    pub latency_correlation: f64,
    pub load_correlation: f64,
}

impl TelemetryCheck {
    pub fn new() -> Self {
        Self {
            latencies: Vec::new(),
            token_counts: Vec::new(),
            system_loads: Vec::new(),
            latency_correlation: 0.0,
            load_correlation: 0.0,
        }
    }

    pub fn add_sample(&mut self, latency: f64, tokens: usize, load: f64) {
        self.latencies.push(latency);
        self.token_counts.push(tokens);
        self.system_loads.push(load);
    }

    /// Compute correlation between scores and telemetry
    pub fn compute_correlations(&mut self, scores: &[f64]) {
        self.latency_correlation = correlation(&self.latencies, scores);
        self.load_correlation = correlation(&self.system_loads, scores);
    }

    /// Should we invalidate due to infrastructure correlation?
    pub fn should_invalidate(&self, threshold: f64) -> bool {
        self.latency_correlation.abs() > threshold || self.load_correlation.abs() > threshold
    }
}

impl Default for TelemetryCheck {
    fn default() -> Self {
        Self::new()
    }
}

/// Pearson correlation coefficient
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_types() {
        assert_eq!(ControlType::all().len(), 4);
        for ct in ControlType::all() {
            assert!(!ct.name().is_empty());
            assert!(!ct.description().is_empty());
        }
    }

    #[test]
    fn test_random_marker_control() {
        let runner = ControlRunner::new(42);
        let scores: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin().abs()).collect();

        let result = runner.run_random_marker_control(&scores, "control_session");

        assert_eq!(result.control_type, ControlType::RandomMarker);
        assert_eq!(result.observations.len(), 50);
        // All observations should be marked as not injected
        assert!(result.observations.iter().all(|o| !o.injected));
    }

    #[test]
    fn test_shuffle_labels() {
        let mut gen = ControlGenerator::new(42);

        let observations: Vec<Observation> = (0..20)
            .map(|i| {
                Observation::new(
                    format!("M{}", i),
                    i < 10, // First 10 are injected
                    0.5,
                    "session".to_string(),
                )
            })
            .collect();

        let original_injected: Vec<bool> = observations.iter().map(|o| o.injected).collect();
        let shuffled = gen.shuffle_labels(&observations);
        let shuffled_injected: Vec<bool> = shuffled.iter().map(|o| o.injected).collect();

        // Same count of true/false
        assert_eq!(
            original_injected.iter().filter(|&&x| x).count(),
            shuffled_injected.iter().filter(|&&x| x).count()
        );

        // But different order (with high probability)
        assert_ne!(original_injected, shuffled_injected);
    }

    #[test]
    fn test_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = correlation(&x, &y);
        assert!((r - 1.0).abs() < 0.01);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = correlation(&x, &y_neg);
        assert!((r_neg + 1.0).abs() < 0.01);

        // No correlation
        let y_rand = vec![5.0, 2.0, 8.0, 1.0, 9.0];
        let r_rand = correlation(&x, &y_rand);
        assert!(r_rand.abs() < 0.5);
    }

    #[test]
    fn test_telemetry_invalidation() {
        let mut check = TelemetryCheck::new();

        // Add samples with strong correlation
        for i in 0..20 {
            check.add_sample(i as f64 * 10.0, 100, i as f64 * 5.0);
        }

        let scores: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
        check.compute_correlations(&scores);

        // High correlation should trigger invalidation
        assert!(check.should_invalidate(0.5));
    }
}
