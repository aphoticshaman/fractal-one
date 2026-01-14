//! ═══════════════════════════════════════════════════════════════════════════════
//! MI — Permutation-Based Mutual Information Estimator
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Primary statistic: Conditional Mutual Information (lower bound)
//!   I_hat(Y_S2; M | X_S2)
//!
//! Uses permutation tests and bootstrap confidence intervals.
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION PAIR
// ═══════════════════════════════════════════════════════════════════════════════

/// A single observation: (marker_id, detection_score)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Marker identifier
    pub marker_id: String,
    /// Was this marker actually injected (true) or control (false)
    pub injected: bool,
    /// Detection score from probe session
    pub score: f64,
    /// Session ID of the probe
    pub probe_session_id: String,
}

impl Observation {
    pub fn new(marker_id: String, injected: bool, score: f64, probe_session_id: String) -> Self {
        Self {
            marker_id,
            injected,
            score,
            probe_session_id,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MI ESTIMATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Mutual Information estimator using permutation tests
pub struct MIEstimator {
    /// All observations
    observations: Vec<Observation>,
    /// RNG state for permutations
    rng_state: u64,
    /// Number of permutations for null distribution
    n_permutations: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// NULL MODE
// ═══════════════════════════════════════════════════════════════════════════════

/// Null hypothesis generation mode for permutation tests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullMode {
    /// Shuffle injection labels (default, breaks marker-score association)
    LabelShuffle,
    /// Circular time-shift of scores (preserves autocorrelation)
    TimeShift,
    /// Block permutation with given block size (preserves local structure)
    BlockPermutation(usize),
}

impl Default for NullMode {
    fn default() -> Self {
        Self::LabelShuffle
    }
}

impl MIEstimator {
    pub fn new(seed: u64) -> Self {
        Self {
            observations: Vec::new(),
            rng_state: seed.max(1),
            n_permutations: 1000,
        }
    }

    pub fn set_permutations(&mut self, n: usize) {
        self.n_permutations = n;
    }

    pub fn add_observation(&mut self, obs: Observation) {
        self.observations.push(obs);
    }

    pub fn add_observations(&mut self, obs: Vec<Observation>) {
        self.observations.extend(obs);
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    fn next_rng(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Shuffle labels using Fisher-Yates
    fn shuffle_labels(&mut self, labels: &mut [bool]) {
        let n = labels.len();
        for i in (1..n).rev() {
            let j = (self.next_rng() as usize) % (i + 1);
            labels.swap(i, j);
        }
    }

    /// Circular time-shift of scores (keeps labels in place, shifts scores)
    /// For autocorrelated series where label shuffle violates exchangeability
    fn time_shift_scores(&mut self, scores: &mut [f64]) {
        let n = scores.len();
        if n < 2 {
            return;
        }
        let shift = (self.next_rng() as usize % (n - 1)) + 1; // Shift by 1 to n-1
        scores.rotate_right(shift);
    }

    /// Block permutation: shuffle contiguous blocks of observations
    /// Better for preserving local autocorrelation structure
    fn block_permute(&mut self, data: &mut [(bool, f64)], block_size: usize) {
        let n = data.len();
        if block_size == 0 || n < block_size * 2 {
            return;
        }

        let n_blocks = n / block_size;
        if n_blocks < 2 {
            return;
        }

        // Create block indices and shuffle them
        let mut block_indices: Vec<usize> = (0..n_blocks).collect();
        for i in (1..n_blocks).rev() {
            let j = (self.next_rng() as usize) % (i + 1);
            block_indices.swap(i, j);
        }

        // Reorder data according to shuffled blocks
        let mut new_data: Vec<(bool, f64)> = Vec::with_capacity(n);
        for &block_idx in &block_indices {
            let start = block_idx * block_size;
            let end = (start + block_size).min(n);
            new_data.extend_from_slice(&data[start..end]);
        }
        // Handle remainder
        let remainder_start = n_blocks * block_size;
        if remainder_start < n {
            new_data.extend_from_slice(&data[remainder_start..]);
        }

        data[..new_data.len()].copy_from_slice(&new_data);
    }

    /// Compute the test statistic: difference in mean scores between injected and control
    fn compute_statistic(observations: &[Observation], labels: &[bool]) -> f64 {
        let (mut sum_injected, mut count_injected) = (0.0, 0);
        let (mut sum_control, mut count_control) = (0.0, 0);

        for (obs, &label) in observations.iter().zip(labels.iter()) {
            if label {
                sum_injected += obs.score;
                count_injected += 1;
            } else {
                sum_control += obs.score;
                count_control += 1;
            }
        }

        let mean_injected = if count_injected > 0 {
            sum_injected / count_injected as f64
        } else {
            0.0
        };

        let mean_control = if count_control > 0 {
            sum_control / count_control as f64
        } else {
            0.0
        };

        mean_injected - mean_control
    }

    /// Run permutation test with specified null mode
    pub fn permutation_test_with_null(&mut self, null_mode: NullMode) -> PermutationResult {
        if self.observations.is_empty() {
            return PermutationResult::empty();
        }

        // Original data
        let original_labels: Vec<bool> = self.observations.iter().map(|o| o.injected).collect();
        let original_scores: Vec<f64> = self.observations.iter().map(|o| o.score).collect();

        // Observed statistic
        let observed = Self::compute_statistic(&self.observations, &original_labels);

        // Null distribution via permutation
        let mut null_distribution = Vec::with_capacity(self.n_permutations);

        for _ in 0..self.n_permutations {
            let stat = match null_mode {
                NullMode::LabelShuffle => {
                    let mut permuted_labels = original_labels.clone();
                    self.shuffle_labels(&mut permuted_labels);
                    Self::compute_statistic(&self.observations, &permuted_labels)
                }
                NullMode::TimeShift => {
                    let mut permuted_scores = original_scores.clone();
                    self.time_shift_scores(&mut permuted_scores);
                    // Create temporary observations with shifted scores
                    let temp_obs: Vec<Observation> = self
                        .observations
                        .iter()
                        .zip(permuted_scores.iter())
                        .map(|(obs, &score)| Observation {
                            marker_id: obs.marker_id.clone(),
                            injected: obs.injected,
                            score,
                            probe_session_id: obs.probe_session_id.clone(),
                        })
                        .collect();
                    Self::compute_statistic(&temp_obs, &original_labels)
                }
                NullMode::BlockPermutation(block_size) => {
                    let mut data: Vec<(bool, f64)> = original_labels
                        .iter()
                        .zip(original_scores.iter())
                        .map(|(&l, &s)| (l, s))
                        .collect();
                    self.block_permute(&mut data, block_size);
                    let (perm_labels, perm_scores): (Vec<bool>, Vec<f64>) =
                        data.into_iter().unzip();
                    let temp_obs: Vec<Observation> = self
                        .observations
                        .iter()
                        .zip(perm_scores.iter())
                        .map(|(obs, &score)| Observation {
                            marker_id: obs.marker_id.clone(),
                            injected: obs.injected,
                            score,
                            probe_session_id: obs.probe_session_id.clone(),
                        })
                        .collect();
                    Self::compute_statistic(&temp_obs, &perm_labels)
                }
            };
            null_distribution.push(stat);
        }

        // Sort null distribution
        null_distribution.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute p-value (one-tailed, upper)
        let count_extreme = null_distribution.iter().filter(|&&x| x >= observed).count();
        let p_value = (count_extreme + 1) as f64 / (self.n_permutations + 1) as f64;

        // Compute mean and std of null
        let null_mean = null_distribution.iter().sum::<f64>() / null_distribution.len() as f64;
        let null_variance = null_distribution
            .iter()
            .map(|x| (x - null_mean).powi(2))
            .sum::<f64>()
            / null_distribution.len() as f64;
        let null_std = null_variance.sqrt();

        // Z-score
        let z_score = if null_std > 0.0 {
            (observed - null_mean) / null_std
        } else {
            0.0
        };

        // Percentiles of null
        let p5_idx = (0.05 * null_distribution.len() as f64) as usize;
        let p95_idx = (0.95 * null_distribution.len() as f64) as usize;
        let p5 = null_distribution.get(p5_idx).copied().unwrap_or(0.0);
        let p95 = null_distribution
            .get(p95_idx.min(null_distribution.len() - 1))
            .copied()
            .unwrap_or(0.0);

        PermutationResult {
            observed_statistic: observed,
            p_value,
            z_score,
            null_mean,
            null_std,
            null_p5: p5,
            null_p95: p95,
            n_permutations: self.n_permutations,
            n_observations: self.observations.len(),
        }
    }

    /// Run permutation test (default: label shuffle)
    pub fn permutation_test(&mut self) -> PermutationResult {
        self.permutation_test_with_null(NullMode::LabelShuffle)
    }

    /// Bootstrap confidence interval for the observed statistic
    pub fn bootstrap_ci(&mut self, n_bootstrap: usize, alpha: f64) -> BootstrapResult {
        if self.observations.is_empty() {
            return BootstrapResult::empty();
        }

        let original_labels: Vec<bool> = self.observations.iter().map(|o| o.injected).collect();
        let observed = Self::compute_statistic(&self.observations, &original_labels);

        let mut bootstrap_stats = Vec::with_capacity(n_bootstrap);
        let n = self.observations.len();

        for _ in 0..n_bootstrap {
            // Sample with replacement
            let mut sample_labels = Vec::with_capacity(n);
            let mut sample_scores = Vec::with_capacity(n);

            for _ in 0..n {
                let idx = (self.next_rng() as usize) % n;
                sample_labels.push(self.observations[idx].injected);
                sample_scores.push(self.observations[idx].score);
            }

            // Create temporary observations for statistic computation
            let sample_obs: Vec<Observation> = sample_scores
                .iter()
                .enumerate()
                .map(|(i, &score)| Observation {
                    marker_id: format!("bootstrap_{}", i),
                    injected: sample_labels[i],
                    score,
                    probe_session_id: String::new(),
                })
                .collect();

            let stat = Self::compute_statistic(&sample_obs, &sample_labels);
            bootstrap_stats.push(stat);
        }

        // Sort and get percentiles
        bootstrap_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = ((alpha / 2.0) * bootstrap_stats.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * bootstrap_stats.len() as f64) as usize;

        let ci_lower = bootstrap_stats[lower_idx.min(bootstrap_stats.len() - 1)];
        let ci_upper = bootstrap_stats[upper_idx.min(bootstrap_stats.len() - 1)];

        let mean = bootstrap_stats.iter().sum::<f64>() / bootstrap_stats.len() as f64;
        let variance = bootstrap_stats
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / bootstrap_stats.len() as f64;

        BootstrapResult {
            observed_statistic: observed,
            ci_lower,
            ci_upper,
            bootstrap_mean: mean,
            bootstrap_std: variance.sqrt(),
            n_bootstrap,
            alpha,
        }
    }

    /// Estimate mutual information (discretized approach)
    pub fn estimate_mi(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }

        // Discretize scores into bins
        let n_bins = 10;
        let scores: Vec<f64> = self.observations.iter().map(|o| o.score).collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_score - min_score).max(1e-10);

        let discretize = |score: f64| -> usize {
            let bin = ((score - min_score) / range * n_bins as f64) as usize;
            bin.min(n_bins - 1)
        };

        // Count joint and marginal distributions
        let mut joint: HashMap<(bool, usize), usize> = HashMap::new();
        let mut marginal_y: HashMap<usize, usize> = HashMap::new();
        let mut marginal_x: HashMap<bool, usize> = HashMap::new();
        let n = self.observations.len();

        for obs in &self.observations {
            let y_bin = discretize(obs.score);
            *joint.entry((obs.injected, y_bin)).or_insert(0) += 1;
            *marginal_y.entry(y_bin).or_insert(0) += 1;
            *marginal_x.entry(obs.injected).or_insert(0) += 1;
        }

        // Compute MI: I(X;Y) = sum P(x,y) * log(P(x,y) / (P(x) * P(y)))
        let mut mi = 0.0;
        for (&(x, y), &count_xy) in &joint {
            let p_xy = count_xy as f64 / n as f64;
            let p_x = *marginal_x.get(&x).unwrap() as f64 / n as f64;
            let p_y = *marginal_y.get(&y).unwrap() as f64 / n as f64;

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        mi
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationResult {
    pub observed_statistic: f64,
    pub p_value: f64,
    pub z_score: f64,
    pub null_mean: f64,
    pub null_std: f64,
    pub null_p5: f64,
    pub null_p95: f64,
    pub n_permutations: usize,
    pub n_observations: usize,
}

impl PermutationResult {
    pub fn empty() -> Self {
        Self {
            observed_statistic: 0.0,
            p_value: 1.0,
            z_score: 0.0,
            null_mean: 0.0,
            null_std: 0.0,
            null_p5: 0.0,
            null_p95: 0.0,
            n_permutations: 0,
            n_observations: 0,
        }
    }

    /// Is the result significant at given alpha?
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Does observed exceed null by k standard deviations?
    pub fn exceeds_null_by(&self, k: f64) -> bool {
        self.observed_statistic > self.null_mean + k * self.null_std
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub observed_statistic: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub bootstrap_mean: f64,
    pub bootstrap_std: f64,
    pub n_bootstrap: usize,
    pub alpha: f64,
}

impl BootstrapResult {
    pub fn empty() -> Self {
        Self {
            observed_statistic: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            bootstrap_mean: 0.0,
            bootstrap_std: 0.0,
            n_bootstrap: 0,
            alpha: 0.05,
        }
    }

    /// Is zero outside the confidence interval?
    pub fn significant(&self) -> bool {
        self.ci_lower > 0.0 || self.ci_upper < 0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_no_effect() {
        let mut estimator = MIEstimator::new(42);
        estimator.set_permutations(500);

        // Generate random observations with no real effect
        for i in 0..100 {
            let injected = i % 2 == 0;
            let score = 0.5 + (i as f64 * 0.01).sin() * 0.1; // Random-ish scores
            estimator.add_observation(Observation::new(
                format!("M{}", i),
                injected,
                score,
                "probe".to_string(),
            ));
        }

        let result = estimator.permutation_test();

        // With no real effect, p-value should be high
        assert!(
            result.p_value > 0.05,
            "No effect should have high p-value: {}",
            result.p_value
        );
    }

    #[test]
    fn test_permutation_strong_effect() {
        let mut estimator = MIEstimator::new(42);
        estimator.set_permutations(500);

        // Generate observations with clear effect
        for i in 0..100 {
            let injected = i % 2 == 0;
            let score = if injected { 0.8 } else { 0.2 }; // Clear separation
            estimator.add_observation(Observation::new(
                format!("M{}", i),
                injected,
                score,
                "probe".to_string(),
            ));
        }

        let result = estimator.permutation_test();

        // With strong effect, p-value should be low
        assert!(
            result.p_value < 0.01,
            "Strong effect should have low p-value: {}",
            result.p_value
        );
        assert!(
            result.z_score > 3.0,
            "Strong effect should have high z-score: {}",
            result.z_score
        );
    }

    #[test]
    fn test_bootstrap_ci() {
        let mut estimator = MIEstimator::new(42);

        for i in 0..50 {
            let injected = i < 25;
            let score = if injected { 0.7 } else { 0.3 };
            estimator.add_observation(Observation::new(
                format!("M{}", i),
                injected,
                score,
                "probe".to_string(),
            ));
        }

        let result = estimator.bootstrap_ci(500, 0.05);

        assert!(
            result.ci_lower > 0.0,
            "CI lower should be positive: {}",
            result.ci_lower
        );
        assert!(result.ci_upper > result.ci_lower);
        assert!(result.significant(), "Should be significant");
    }

    #[test]
    fn test_mi_estimation() {
        let mut estimator = MIEstimator::new(42);

        // Perfect correlation: injected = high score
        for i in 0..100 {
            let injected = i < 50;
            let score = if injected { 0.9 } else { 0.1 };
            estimator.add_observation(Observation::new(
                format!("M{}", i),
                injected,
                score,
                "probe".to_string(),
            ));
        }

        let mi = estimator.estimate_mi();
        assert!(mi > 0.3, "High correlation should have high MI: {}", mi);
    }

    #[test]
    fn test_mi_no_correlation() {
        let mut estimator = MIEstimator::new(42);

        // Random scores regardless of injection
        for i in 0..100 {
            let injected = i % 2 == 0;
            let score = (i as f64 * 0.123).sin().abs(); // Pseudo-random
            estimator.add_observation(Observation::new(
                format!("M{}", i),
                injected,
                score,
                "probe".to_string(),
            ));
        }

        let mi = estimator.estimate_mi();
        assert!(mi < 0.2, "No correlation should have low MI: {}", mi);
    }
}
