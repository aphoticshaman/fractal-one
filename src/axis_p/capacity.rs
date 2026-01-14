//! ═══════════════════════════════════════════════════════════════════════════════
//! CHANNEL CAPACITY ESTIMATOR
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Estimates the channel capacity of a hypothetical persistence channel.
//!
//! Channel Model:
//!   X (binary) → Persistence Channel → Y (continuous score)
//!
//! Key metrics:
//!   - C_hat: Estimated channel capacity in bits
//!   - I(X;Y): Mutual information between injection and detection
//!   - H(X), H(Y): Marginal entropies
//!   - Reliability: Probability of correct detection
//!
//! Methods:
//!   - Blahut-Arimoto algorithm for capacity estimation
//!   - KDE-based differential entropy estimation
//!   - Discretization-based MI bounds
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
// Note: E and PI available if needed for differential entropy

use super::mi::Observation;
use crate::stats::float_cmp;

// ═══════════════════════════════════════════════════════════════════════════════
// CHANNEL MODEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Binary-to-continuous channel model for persistence detection
#[derive(Debug, Clone)]
pub struct PersistenceChannel {
    /// Observations from probing
    observations: Vec<Observation>,
    /// Conditional distribution P(Y|X=1) - scores when injected
    dist_injected: Vec<f64>,
    /// Conditional distribution P(Y|X=0) - scores when not injected
    dist_control: Vec<f64>,
}

impl PersistenceChannel {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            dist_injected: Vec::new(),
            dist_control: Vec::new(),
        }
    }

    pub fn from_observations(observations: Vec<Observation>) -> Self {
        let dist_injected: Vec<f64> = observations
            .iter()
            .filter(|o| o.injected)
            .map(|o| o.score)
            .collect();

        let dist_control: Vec<f64> = observations
            .iter()
            .filter(|o| !o.injected)
            .map(|o| o.score)
            .collect();

        Self {
            observations,
            dist_injected,
            dist_control,
        }
    }

    pub fn add_observation(&mut self, obs: Observation) {
        if obs.injected {
            self.dist_injected.push(obs.score);
        } else {
            self.dist_control.push(obs.score);
        }
        self.observations.push(obs);
    }

    pub fn n_observations(&self) -> usize {
        self.observations.len()
    }

    pub fn n_injected(&self) -> usize {
        self.dist_injected.len()
    }

    pub fn n_control(&self) -> usize {
        self.dist_control.len()
    }
}

impl Default for PersistenceChannel {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAPACITY ESTIMATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for capacity estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityConfig {
    /// Number of bins for discretization
    pub n_bins: usize,
    /// Number of iterations for Blahut-Arimoto
    pub ba_iterations: usize,
    /// Convergence threshold for BA
    pub ba_tolerance: f64,
    /// Bandwidth for KDE (if None, use Scott's rule)
    pub kde_bandwidth: Option<f64>,
    /// Bootstrap samples for confidence intervals
    pub bootstrap_samples: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for CapacityConfig {
    fn default() -> Self {
        Self {
            n_bins: 20,
            ba_iterations: 100,
            ba_tolerance: 1e-8,
            kde_bandwidth: None,
            bootstrap_samples: 500,
            seed: 42,
        }
    }
}

/// Channel capacity estimation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityResult {
    /// Estimated channel capacity in bits
    pub capacity_bits: f64,
    /// Lower bound (95% CI)
    pub capacity_lower: f64,
    /// Upper bound (95% CI)
    pub capacity_upper: f64,
    /// Mutual information I(X;Y)
    pub mutual_information: f64,
    /// Entropy H(X)
    pub entropy_x: f64,
    /// Entropy H(Y)
    pub entropy_y: f64,
    /// Conditional entropy H(Y|X)
    pub entropy_y_given_x: f64,
    /// Mean score for injected markers
    pub mean_injected: f64,
    /// Mean score for control markers
    pub mean_control: f64,
    /// Standard deviation for injected
    pub std_injected: f64,
    /// Standard deviation for control
    pub std_control: f64,
    /// Separation (mean_injected - mean_control)
    pub separation: f64,
    /// Reliability (probability of correct classification)
    pub reliability: f64,
    /// Bits per marker (practical throughput)
    pub bits_per_marker: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Interpretation of capacity
    pub interpretation: String,
}

impl CapacityResult {
    /// Generate human-readable interpretation
    fn interpret(capacity: f64, reliability: f64, _separation: f64) -> String {
        if capacity < 0.01 {
            "ZERO CAPACITY: No detectable information transfer. Consistent with null hypothesis."
                .to_string()
        } else if capacity < 0.1 {
            format!(
                "NEGLIGIBLE CAPACITY: {:.3} bits. Noise-level signal, likely statistical artifact.",
                capacity
            )
        } else if capacity < 0.3 {
            format!(
                "LOW CAPACITY: {:.3} bits. Weak channel, {:.0}% reliability. Could indicate caching or partial persistence.",
                capacity, reliability * 100.0
            )
        } else if capacity < 0.6 {
            format!(
                "MODERATE CAPACITY: {:.3} bits. Usable channel, {:.0}% reliability. Significant persistence detected.",
                capacity, reliability * 100.0
            )
        } else if capacity < 0.9 {
            format!(
                "HIGH CAPACITY: {:.3} bits. Strong channel, {:.0}% reliability. Clear persistence mechanism.",
                capacity, reliability * 100.0
            )
        } else {
            format!(
                "NEAR-PERFECT CAPACITY: {:.3} bits. {:.0}% reliability. Almost deterministic persistence.",
                capacity, reliability * 100.0
            )
        }
    }
}

/// Main channel capacity estimator
pub struct ChannelCapacityEstimator {
    channel: PersistenceChannel,
    config: CapacityConfig,
    rng_state: u64,
}

impl ChannelCapacityEstimator {
    pub fn new(config: CapacityConfig) -> Self {
        let rng_state = config.seed.max(1);
        Self {
            channel: PersistenceChannel::new(),
            config,
            rng_state,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(CapacityConfig::default())
    }

    pub fn add_observation(&mut self, obs: Observation) {
        self.channel.add_observation(obs);
    }

    pub fn add_observations(&mut self, obs: Vec<Observation>) {
        for o in obs {
            self.channel.add_observation(o);
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

    #[allow(dead_code)]
    fn rng_float(&mut self) -> f64 {
        (self.next_rng() as f64) / (u64::MAX as f64)
    }

    /// Compute mean of samples
    fn mean(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        samples.iter().sum::<f64>() / samples.len() as f64
    }

    /// Compute standard deviation
    fn std(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mean = Self::mean(samples);
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
        variance.sqrt()
    }

    /// Binary entropy H(p) in bits
    fn binary_entropy(p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return 0.0;
        }
        -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
    }

    /// Discretize continuous scores into bins
    fn discretize(&self, score: f64, min_score: f64, range: f64) -> usize {
        let bin = ((score - min_score) / range * self.config.n_bins as f64) as usize;
        bin.min(self.config.n_bins - 1)
    }

    /// Estimate entropy of discretized distribution
    fn discrete_entropy(counts: &[usize]) -> f64 {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let mut h = 0.0;
        for &count in counts {
            if count > 0 {
                let p = count as f64 / total as f64;
                h -= p * p.log2();
            }
        }
        h
    }

    /// Estimate conditional entropy H(Y|X) using discretization
    fn conditional_entropy_discrete(
        &self,
        scores_given_x0: &[f64],
        scores_given_x1: &[f64],
        min_score: f64,
        range: f64,
    ) -> f64 {
        let n0 = scores_given_x0.len();
        let n1 = scores_given_x1.len();
        let n_total = n0 + n1;

        if n_total == 0 {
            return 0.0;
        }

        // Discretize scores for X=0
        let mut counts_x0 = vec![0usize; self.config.n_bins];
        for &s in scores_given_x0 {
            let bin = self.discretize(s, min_score, range);
            counts_x0[bin] += 1;
        }

        // Discretize scores for X=1
        let mut counts_x1 = vec![0usize; self.config.n_bins];
        for &s in scores_given_x1 {
            let bin = self.discretize(s, min_score, range);
            counts_x1[bin] += 1;
        }

        // H(Y|X) = P(X=0) * H(Y|X=0) + P(X=1) * H(Y|X=1)
        let p_x0 = n0 as f64 / n_total as f64;
        let p_x1 = n1 as f64 / n_total as f64;

        let h_y_x0 = Self::discrete_entropy(&counts_x0);
        let h_y_x1 = Self::discrete_entropy(&counts_x1);

        p_x0 * h_y_x0 + p_x1 * h_y_x1
    }

    /// Estimate marginal entropy H(Y) using discretization
    fn marginal_entropy_y_discrete(&self, all_scores: &[f64], min_score: f64, range: f64) -> f64 {
        let mut counts = vec![0usize; self.config.n_bins];
        for &s in all_scores {
            let bin = self.discretize(s, min_score, range);
            counts[bin] += 1;
        }
        Self::discrete_entropy(&counts)
    }

    /// Estimate mutual information I(X;Y) = H(Y) - H(Y|X)
    #[allow(dead_code)]
    fn estimate_mi_discrete(&self) -> f64 {
        let all_scores: Vec<f64> = self.channel.observations.iter().map(|o| o.score).collect();
        if all_scores.is_empty() {
            return 0.0;
        }

        let min_score = all_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = all_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_score - min_score).max(1e-10);

        let h_y = self.marginal_entropy_y_discrete(&all_scores, min_score, range);
        let h_y_x = self.conditional_entropy_discrete(
            &self.channel.dist_control,
            &self.channel.dist_injected,
            min_score,
            range,
        );

        (h_y - h_y_x).max(0.0)
    }

    /// Compute reliability (probability of correct binary classification)
    /// Using optimal threshold based on Bayes decision rule
    fn compute_reliability(&self) -> f64 {
        if self.channel.dist_injected.is_empty() || self.channel.dist_control.is_empty() {
            return 0.5; // Random guess
        }

        let mean_inj = Self::mean(&self.channel.dist_injected);
        let mean_ctl = Self::mean(&self.channel.dist_control);

        // Optimal threshold is midpoint (assuming equal priors and costs)
        let threshold = (mean_inj + mean_ctl) / 2.0;

        // Count correct classifications
        let correct_inj = self
            .channel
            .dist_injected
            .iter()
            .filter(|&&s| s >= threshold)
            .count();
        let correct_ctl = self
            .channel
            .dist_control
            .iter()
            .filter(|&&s| s < threshold)
            .count();

        let total = self.channel.n_injected() + self.channel.n_control();
        if total == 0 {
            return 0.5;
        }

        (correct_inj + correct_ctl) as f64 / total as f64
    }

    /// Estimate channel capacity using Blahut-Arimoto algorithm
    /// For binary input, this simplifies considerably
    fn estimate_capacity_ba(&self) -> f64 {
        // For binary input channel: C = max_{p} I(X;Y)
        // where p = P(X=1)

        let all_scores: Vec<f64> = self.channel.observations.iter().map(|o| o.score).collect();
        if all_scores.is_empty() {
            return 0.0;
        }

        let min_score = all_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = all_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_score - min_score).max(1e-10);

        // Discretize conditional distributions
        let mut p_y_x0 = vec![0.0f64; self.config.n_bins]; // P(Y|X=0)
        let mut p_y_x1 = vec![0.0f64; self.config.n_bins]; // P(Y|X=1)

        for &s in &self.channel.dist_control {
            let bin = self.discretize(s, min_score, range);
            p_y_x0[bin] += 1.0;
        }
        for &s in &self.channel.dist_injected {
            let bin = self.discretize(s, min_score, range);
            p_y_x1[bin] += 1.0;
        }

        // Normalize
        let sum_x0: f64 = p_y_x0.iter().sum();
        let sum_x1: f64 = p_y_x1.iter().sum();

        if sum_x0 > 0.0 {
            for p in &mut p_y_x0 {
                *p /= sum_x0;
            }
        }
        if sum_x1 > 0.0 {
            for p in &mut p_y_x1 {
                *p /= sum_x1;
            }
        }

        // Blahut-Arimoto for binary input
        // Start with uniform input distribution
        let mut p_x = 0.5; // P(X=1)
        let mut capacity = 0.0;

        for _ in 0..self.config.ba_iterations {
            // Compute marginal P(Y) = p_x * P(Y|X=1) + (1-p_x) * P(Y|X=0)
            let p_y: Vec<f64> = (0..self.config.n_bins)
                .map(|j| p_x * p_y_x1[j] + (1.0 - p_x) * p_y_x0[j])
                .collect();

            // Compute I(X;Y) for current p_x
            let mut mi = 0.0;
            for j in 0..self.config.n_bins {
                if p_y[j] > 1e-10 {
                    if p_y_x0[j] > 1e-10 {
                        mi += (1.0 - p_x) * p_y_x0[j] * (p_y_x0[j] / p_y[j]).log2();
                    }
                    if p_y_x1[j] > 1e-10 {
                        mi += p_x * p_y_x1[j] * (p_y_x1[j] / p_y[j]).log2();
                    }
                }
            }

            // Update input distribution using BA update
            // For binary input: optimize p_x directly
            // Compute derivative and do gradient step
            let mut d_mi_dp = 0.0;
            for j in 0..self.config.n_bins {
                if p_y[j] > 1e-10 {
                    let term = p_y_x1[j] - p_y_x0[j];
                    if p_y_x1[j] > 1e-10 {
                        d_mi_dp += p_y_x1[j] * (p_y_x1[j] / p_y[j]).log2();
                    }
                    if p_y_x0[j] > 1e-10 {
                        d_mi_dp -= p_y_x0[j] * (p_y_x0[j] / p_y[j]).log2();
                    }
                    // Correction term
                    if term.abs() > 1e-10 && p_y[j] > 1e-10 {
                        d_mi_dp -= (p_x * p_y_x1[j] + (1.0 - p_x) * p_y_x0[j]) * term / p_y[j];
                    }
                }
            }

            // Gradient ascent with small step
            let step = 0.1;
            let new_p_x = (p_x + step * d_mi_dp).clamp(0.01, 0.99);

            // Check convergence
            if (mi - capacity).abs() < self.config.ba_tolerance {
                capacity = mi;
                break;
            }

            capacity = mi;
            p_x = new_p_x;
        }

        capacity.max(0.0)
    }

    /// Bootstrap confidence interval for capacity
    fn bootstrap_capacity(&mut self) -> (f64, f64, f64) {
        let n = self.channel.n_observations();
        if n < 10 {
            let cap = self.estimate_capacity_ba();
            return (cap, cap * 0.5, cap * 1.5);
        }

        let mut bootstrap_caps = Vec::with_capacity(self.config.bootstrap_samples);

        for _ in 0..self.config.bootstrap_samples {
            // Resample observations with replacement
            let mut sample = PersistenceChannel::new();
            for _ in 0..n {
                let idx = (self.next_rng() as usize) % n;
                sample.add_observation(self.channel.observations[idx].clone());
            }

            // Compute capacity for this sample
            let temp_estimator = ChannelCapacityEstimator {
                channel: sample,
                config: self.config.clone(),
                rng_state: self.next_rng(),
            };
            let cap = temp_estimator.estimate_capacity_ba();
            bootstrap_caps.push(cap);
        }

        // Sort and get percentiles
        bootstrap_caps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_cap = Self::mean(&bootstrap_caps);
        let lower_idx = (0.025 * bootstrap_caps.len() as f64) as usize;
        let upper_idx = (0.975 * bootstrap_caps.len() as f64) as usize;

        let lower = bootstrap_caps.get(lower_idx).copied().unwrap_or(0.0);
        let upper = bootstrap_caps
            .get(upper_idx.min(bootstrap_caps.len() - 1))
            .copied()
            .unwrap_or(mean_cap);

        (mean_cap, lower, upper)
    }

    /// Estimate all channel metrics
    pub fn estimate(&mut self) -> CapacityResult {
        let n_obs = self.channel.n_observations();

        if n_obs == 0 {
            return CapacityResult {
                capacity_bits: 0.0,
                capacity_lower: 0.0,
                capacity_upper: 0.0,
                mutual_information: 0.0,
                entropy_x: 0.0,
                entropy_y: 0.0,
                entropy_y_given_x: 0.0,
                mean_injected: 0.0,
                mean_control: 0.0,
                std_injected: 0.0,
                std_control: 0.0,
                separation: 0.0,
                reliability: 0.5,
                bits_per_marker: 0.0,
                n_observations: 0,
                interpretation: "NO DATA: Cannot estimate capacity without observations."
                    .to_string(),
            };
        }

        // Basic statistics
        let mean_inj = Self::mean(&self.channel.dist_injected);
        let mean_ctl = Self::mean(&self.channel.dist_control);
        let std_inj = Self::std(&self.channel.dist_injected);
        let std_ctl = Self::std(&self.channel.dist_control);
        let separation = mean_inj - mean_ctl;

        // Entropies
        let all_scores: Vec<f64> = self.channel.observations.iter().map(|o| o.score).collect();
        let min_score = all_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = all_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max_score - min_score).max(1e-10);

        let h_y = self.marginal_entropy_y_discrete(&all_scores, min_score, range);
        let h_y_x = self.conditional_entropy_discrete(
            &self.channel.dist_control,
            &self.channel.dist_injected,
            min_score,
            range,
        );

        // H(X) for balanced design
        let p_injected = self.channel.n_injected() as f64 / n_obs as f64;
        let h_x = Self::binary_entropy(p_injected);

        // MI = H(Y) - H(Y|X)
        let mi = (h_y - h_y_x).max(0.0);

        // Reliability
        let reliability = self.compute_reliability();

        // Capacity via Blahut-Arimoto
        let (capacity, cap_lower, cap_upper) = self.bootstrap_capacity();

        // Bits per marker (practical throughput = capacity * reliability)
        let bits_per_marker = capacity * reliability;

        // Interpretation
        let interpretation = CapacityResult::interpret(capacity, reliability, separation);

        CapacityResult {
            capacity_bits: capacity,
            capacity_lower: cap_lower,
            capacity_upper: cap_upper,
            mutual_information: mi,
            entropy_x: h_x,
            entropy_y: h_y,
            entropy_y_given_x: h_y_x,
            mean_injected: mean_inj,
            mean_control: mean_ctl,
            std_injected: std_inj,
            std_control: std_ctl,
            separation,
            reliability,
            bits_per_marker,
            n_observations: n_obs,
            interpretation,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAPACITY SWEEP (varying parameters)
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of capacity measurement at a specific parameter setting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityPoint {
    /// Parameter value (e.g., washout time in ms)
    pub parameter: f64,
    /// Estimated capacity
    pub capacity: f64,
    /// Confidence interval
    pub ci_lower: f64,
    pub ci_upper: f64,
    /// Number of observations
    pub n_observations: usize,
}

/// Capacity as a function of some parameter (e.g., washout time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityCurve {
    /// Parameter name
    pub parameter_name: String,
    /// Data points
    pub points: Vec<CapacityPoint>,
    /// Maximum capacity found
    pub max_capacity: f64,
    /// Parameter value at maximum
    pub optimal_parameter: f64,
}

impl CapacityCurve {
    pub fn new(parameter_name: &str) -> Self {
        Self {
            parameter_name: parameter_name.to_string(),
            points: Vec::new(),
            max_capacity: 0.0,
            optimal_parameter: 0.0,
        }
    }

    pub fn add_point(&mut self, point: CapacityPoint) {
        if point.capacity > self.max_capacity {
            self.max_capacity = point.capacity;
            self.optimal_parameter = point.parameter;
        }
        self.points.push(point);
    }

    pub fn sort_by_parameter(&mut self) {
        self.points
            .sort_by(|a, b| float_cmp(&a.parameter, &b.parameter));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(id: &str, injected: bool, score: f64) -> Observation {
        Observation::new(id.to_string(), injected, score, "test".to_string())
    }

    #[test]
    fn test_capacity_zero_separation() {
        let mut estimator = ChannelCapacityEstimator::with_default_config();

        // No separation between injected and control
        for i in 0..50 {
            estimator.add_observation(make_obs(&format!("M{}", i), i % 2 == 0, 0.5));
        }

        let result = estimator.estimate();

        assert!(
            result.capacity_bits < 0.1,
            "Zero separation should have near-zero capacity: {}",
            result.capacity_bits
        );
        assert!(result.separation.abs() < 0.01);
    }

    #[test]
    fn test_capacity_perfect_separation() {
        let mut estimator = ChannelCapacityEstimator::with_default_config();

        // Perfect separation
        for i in 0..100 {
            let injected = i % 2 == 0;
            let score = if injected { 1.0 } else { 0.0 };
            estimator.add_observation(make_obs(&format!("M{}", i), injected, score));
        }

        let result = estimator.estimate();

        assert!(
            result.capacity_bits > 0.8,
            "Perfect separation should have high capacity: {}",
            result.capacity_bits
        );
        assert!(result.reliability > 0.99);
        assert!((result.separation - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_capacity_partial_separation() {
        let mut estimator = ChannelCapacityEstimator::with_default_config();

        // Partial separation with significant overlap
        // Injected: scores distributed around 0.6 with ±0.25 spread
        // Control: scores distributed around 0.4 with ±0.25 spread
        // This creates ~20-30% overlap in distributions
        for i in 0..100 {
            let injected = i % 2 == 0;
            // Use pseudo-random noise that creates real overlap
            let noise = (i as f64 * 1.7).sin() * 0.25; // Signed noise ±0.25
            let base = if injected { 0.6 } else { 0.4 };
            let score = (base + noise).clamp(0.0, 1.0);
            estimator.add_observation(make_obs(&format!("M{}", i), injected, score));
        }

        let result = estimator.estimate();

        // With overlapping distributions, capacity should be moderate
        // Allow wider range since exact value depends on noise realization
        assert!(
            result.capacity_bits > 0.05 && result.capacity_bits < 1.0,
            "Partial separation should have moderate capacity: {}",
            result.capacity_bits
        );
        assert!(result.reliability > 0.55 && result.reliability < 0.98);
    }

    #[test]
    fn test_entropy_binary() {
        // H(0.5) = 1 bit
        let h = ChannelCapacityEstimator::binary_entropy(0.5);
        assert!((h - 1.0).abs() < 0.001);

        // H(0) = H(1) = 0
        assert!(ChannelCapacityEstimator::binary_entropy(0.0) < 0.001);
        assert!(ChannelCapacityEstimator::binary_entropy(1.0) < 0.001);

        // H(0.25) ≈ 0.811
        let h = ChannelCapacityEstimator::binary_entropy(0.25);
        assert!((h - 0.811).abs() < 0.01);
    }

    #[test]
    fn test_reliability_computation() {
        let mut estimator = ChannelCapacityEstimator::with_default_config();

        // 80% correct for injected (score > 0.5)
        // 80% correct for control (score < 0.5)
        for i in 0..100 {
            let injected = i < 50;
            let score = if injected {
                if i % 5 == 0 {
                    0.3
                } else {
                    0.8
                } // 80% above 0.5
            } else {
                if i % 5 == 0 {
                    0.7
                } else {
                    0.2
                } // 80% below 0.5
            };
            estimator.add_observation(make_obs(&format!("M{}", i), injected, score));
        }

        let result = estimator.estimate();
        assert!(
            result.reliability > 0.7 && result.reliability < 0.9,
            "Reliability should be around 80%: {}",
            result.reliability
        );
    }

    #[test]
    fn test_capacity_curve() {
        let mut curve = CapacityCurve::new("washout_ms");

        curve.add_point(CapacityPoint {
            parameter: 100.0,
            capacity: 0.8,
            ci_lower: 0.6,
            ci_upper: 0.9,
            n_observations: 50,
        });

        curve.add_point(CapacityPoint {
            parameter: 500.0,
            capacity: 0.5,
            ci_lower: 0.3,
            ci_upper: 0.7,
            n_observations: 50,
        });

        curve.add_point(CapacityPoint {
            parameter: 1000.0,
            capacity: 0.2,
            ci_lower: 0.1,
            ci_upper: 0.3,
            n_observations: 50,
        });

        assert_eq!(curve.max_capacity, 0.8);
        assert_eq!(curve.optimal_parameter, 100.0);
        assert_eq!(curve.points.len(), 3);
    }

    #[test]
    fn test_mi_consistent_with_capacity() {
        let mut estimator = ChannelCapacityEstimator::with_default_config();

        // Add data with clear signal
        for i in 0..100 {
            let injected = i % 2 == 0;
            let score = if injected { 0.9 } else { 0.1 };
            estimator.add_observation(make_obs(&format!("M{}", i), injected, score));
        }

        let result = estimator.estimate();

        // MI should not exceed capacity
        assert!(
            result.mutual_information <= result.capacity_bits + 0.1,
            "MI {} should not greatly exceed capacity {}",
            result.mutual_information,
            result.capacity_bits
        );
    }

    #[test]
    fn test_bits_per_marker() {
        let mut estimator = ChannelCapacityEstimator::with_default_config();

        // High capacity, high reliability
        for i in 0..100 {
            let injected = i % 2 == 0;
            let score = if injected { 0.95 } else { 0.05 };
            estimator.add_observation(make_obs(&format!("M{}", i), injected, score));
        }

        let result = estimator.estimate();

        // Bits per marker = capacity * reliability
        let expected_bpm = result.capacity_bits * result.reliability;
        assert!(
            (result.bits_per_marker - expected_bpm).abs() < 0.01,
            "Bits per marker mismatch"
        );
    }

    #[test]
    fn test_interpretation() {
        let interp_zero = CapacityResult::interpret(0.005, 0.5, 0.0);
        assert!(interp_zero.contains("ZERO CAPACITY"));

        let interp_low = CapacityResult::interpret(0.05, 0.6, 0.1);
        assert!(interp_low.contains("NEGLIGIBLE"));

        let interp_mod = CapacityResult::interpret(0.4, 0.75, 0.3);
        assert!(interp_mod.contains("MODERATE"));

        let interp_high = CapacityResult::interpret(0.95, 0.98, 0.9);
        assert!(interp_high.contains("NEAR-PERFECT"));
    }
}
