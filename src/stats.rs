//! ═══════════════════════════════════════════════════════════════════════════════
//! STATS — Statistical Primitives for Sensory Processing
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Core statistical tools:
//! - EWMA (Exponentially Weighted Moving Average) for smoothing
//! - Median/MAD for robust central tendency and scale
//! - Change-point detection for regime shifts
//!
//! These are the building blocks for all sensorium estimators.
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════════
// EWMA — Exponentially Weighted Moving Average
// ═══════════════════════════════════════════════════════════════════════════════

/// Exponentially Weighted Moving Average
/// New value weighted by α, history by (1-α)
#[derive(Debug, Clone)]
pub struct Ewma {
    /// Smoothing factor (0 < α ≤ 1)
    /// Higher = more responsive, Lower = more smooth
    alpha: f64,
    /// Current smoothed value
    value: f64,
    /// Whether initialized with at least one sample
    initialized: bool,
    /// Sample count
    count: u64,
}

impl Ewma {
    /// Create new EWMA with specified alpha
    /// α = 2/(N+1) where N is effective window size
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "Alpha must be in (0, 1]");
        Self {
            alpha,
            value: 0.0,
            initialized: false,
            count: 0,
        }
    }

    /// Create EWMA with specified effective window size
    /// N = 2/α - 1
    pub fn with_window(window_size: usize) -> Self {
        let alpha = 2.0 / (window_size as f64 + 1.0);
        Self::new(alpha)
    }

    /// Update with new sample
    pub fn update(&mut self, sample: f64) {
        if !self.initialized {
            self.value = sample;
            self.initialized = true;
        } else {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        }
        self.count += 1;
    }

    /// Current smoothed value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Is EWMA initialized?
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Sample count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset to uninitialized state
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
        self.count = 0;
    }

    /// Compute residual (deviation from current value)
    pub fn residual(&self, sample: f64) -> f64 {
        sample - self.value
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROBUST STATS — Median and MAD
// ═══════════════════════════════════════════════════════════════════════════════

/// Rolling window for robust statistics (median, MAD)
#[derive(Debug, Clone)]
pub struct RobustStats {
    /// Sample buffer
    samples: VecDeque<f64>,
    /// Maximum window size
    max_size: usize,
    /// Cached median (invalidated on update)
    cached_median: Option<f64>,
    /// Cached MAD (invalidated on update)
    cached_mad: Option<f64>,
}

impl RobustStats {
    pub fn new(max_size: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_size),
            max_size,
            cached_median: None,
            cached_mad: None,
        }
    }

    /// Add a sample
    pub fn add(&mut self, sample: f64) {
        if self.samples.len() >= self.max_size {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        // Invalidate cache
        self.cached_median = None;
        self.cached_mad = None;
    }

    /// Current sample count
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Compute median (cached)
    pub fn median(&mut self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }

        if let Some(cached) = self.cached_median {
            return Some(cached);
        }

        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };

        self.cached_median = Some(median);
        Some(median)
    }

    /// Compute MAD (Median Absolute Deviation) (cached)
    /// Scale factor 1.4826 makes it consistent with std dev for normal distribution
    pub fn mad(&mut self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }

        if let Some(cached) = self.cached_mad {
            return Some(cached);
        }

        let median = self.median()?;

        let mut deviations: Vec<f64> = self.samples.iter().map(|&x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = deviations.len() / 2;
        let mad = if deviations.len() % 2 == 0 {
            (deviations[mid - 1] + deviations[mid]) / 2.0
        } else {
            deviations[mid]
        };

        // Scale for consistency with standard deviation
        let scaled_mad = mad * 1.4826;
        self.cached_mad = Some(scaled_mad);
        Some(scaled_mad)
    }

    /// Compute z-score using robust estimates
    pub fn robust_z_score(&mut self, value: f64) -> Option<f64> {
        let median = self.median()?;
        let mad = self.mad()?;

        if mad < 1e-10 {
            return Some(0.0); // No variation
        }

        Some((value - median) / mad)
    }

    /// Is this value an outlier? (|z| > threshold)
    pub fn is_outlier(&mut self, value: f64, threshold: f64) -> bool {
        self.robust_z_score(value)
            .map(|z| z.abs() > threshold)
            .unwrap_or(false)
    }

    /// Reset the buffer
    pub fn reset(&mut self) {
        self.samples.clear();
        self.cached_median = None;
        self.cached_mad = None;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHANGE-POINT DETECTION — CUSUM-based
// ═══════════════════════════════════════════════════════════════════════════════

/// CUSUM (Cumulative Sum) change-point detector
/// Detects shifts in the mean of a time series
#[derive(Debug, Clone)]
pub struct CusumDetector {
    /// Target mean (baseline)
    target: f64,
    /// Standard deviation estimate
    sigma: f64,
    /// Slack parameter (deadband around target)
    slack: f64,
    /// Decision threshold for alarm
    threshold: f64,
    /// Positive CUSUM statistic
    cusum_pos: f64,
    /// Negative CUSUM statistic
    cusum_neg: f64,
    /// Whether currently in alarm state
    in_alarm: bool,
    /// Sample count since last reset
    samples_since_reset: u64,
}

impl CusumDetector {
    /// Create new CUSUM detector
    /// - target: expected mean
    /// - sigma: expected standard deviation
    /// - slack: allowable slack (typically 0.5 * sigma)
    /// - threshold: alarm threshold (typically 4-5 * sigma)
    pub fn new(target: f64, sigma: f64, slack: f64, threshold: f64) -> Self {
        Self {
            target,
            sigma,
            slack,
            threshold,
            cusum_pos: 0.0,
            cusum_neg: 0.0,
            in_alarm: false,
            samples_since_reset: 0,
        }
    }

    /// Create with default parameters from baseline stats
    pub fn from_baseline(mean: f64, std_dev: f64) -> Self {
        let sigma = std_dev.max(1e-6);
        Self::new(
            mean,
            sigma,
            0.5 * sigma, // slack = 0.5σ
            4.0 * sigma, // threshold = 4σ
        )
    }

    /// Update with new sample, returns true if change-point detected
    pub fn update(&mut self, sample: f64) -> bool {
        let deviation = sample - self.target;

        // Update positive CUSUM (detects upward shift)
        self.cusum_pos = (self.cusum_pos + deviation - self.slack).max(0.0);

        // Update negative CUSUM (detects downward shift)
        self.cusum_neg = (self.cusum_neg - deviation - self.slack).max(0.0);

        self.samples_since_reset += 1;

        // Check for alarm
        let alarm_triggered = self.cusum_pos > self.threshold || self.cusum_neg > self.threshold;

        if alarm_triggered && !self.in_alarm {
            self.in_alarm = true;
            return true;
        }

        false
    }

    /// Current CUSUM statistics
    pub fn cusum_stats(&self) -> (f64, f64) {
        (self.cusum_pos, self.cusum_neg)
    }

    /// Is detector in alarm state?
    pub fn in_alarm(&self) -> bool {
        self.in_alarm
    }

    /// Reset detector (after acknowledged change-point)
    pub fn reset(&mut self) {
        self.cusum_pos = 0.0;
        self.cusum_neg = 0.0;
        self.in_alarm = false;
        self.samples_since_reset = 0;
    }

    /// Update baseline parameters
    pub fn update_baseline(&mut self, target: f64, sigma: f64) {
        self.target = target;
        self.sigma = sigma;
        self.slack = 0.5 * sigma;
        self.threshold = 4.0 * sigma;
    }

    /// Normalized distance from target (z-score-like)
    pub fn normalized_distance(&self, sample: f64) -> f64 {
        (sample - self.target) / self.sigma.max(1e-6)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VARIANCE TRACKER — Online variance estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Welford's online variance algorithm
#[derive(Debug, Clone)]
pub struct VarianceTracker {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squared deviations
}

impl VarianceTracker {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Update with new sample
    pub fn update(&mut self, sample: f64) {
        self.count += 1;
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = sample - self.mean;
        self.m2 += delta * delta2;
    }

    /// Current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Current variance (population)
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Current variance (sample, Bessel corrected)
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sample count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

impl Default for VarianceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RATE ESTIMATOR — Events per time window
// ═══════════════════════════════════════════════════════════════════════════════

use std::time::{Duration, Instant};

/// Rolling rate estimator (events per second)
#[derive(Debug)]
pub struct RateEstimator {
    /// Event timestamps
    events: VecDeque<Instant>,
    /// Window size
    window: Duration,
}

impl RateEstimator {
    pub fn new(window: Duration) -> Self {
        Self {
            events: VecDeque::new(),
            window,
        }
    }

    /// Record an event
    pub fn record(&mut self) {
        let now = Instant::now();
        self.events.push_back(now);
        self.prune();
    }

    /// Prune old events outside window
    fn prune(&mut self) {
        let cutoff = Instant::now() - self.window;
        while let Some(&front) = self.events.front() {
            if front < cutoff {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Current rate (events per second)
    pub fn rate(&mut self) -> f64 {
        self.prune();
        self.events.len() as f64 / self.window.as_secs_f64()
    }

    /// Event count in window
    pub fn count(&mut self) -> usize {
        self.prune();
        self.events.len()
    }

    /// Reset
    pub fn reset(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_basic() {
        let mut ewma = Ewma::new(0.5);
        ewma.update(10.0);
        assert_eq!(ewma.value(), 10.0); // First value

        ewma.update(20.0);
        assert_eq!(ewma.value(), 15.0); // 0.5*20 + 0.5*10

        ewma.update(20.0);
        assert_eq!(ewma.value(), 17.5); // 0.5*20 + 0.5*15
    }

    #[test]
    fn test_robust_stats_median() {
        let mut stats = RobustStats::new(100);
        for i in 1..=9 {
            stats.add(i as f64);
        }

        assert_eq!(stats.median(), Some(5.0));
    }

    #[test]
    fn test_robust_stats_mad() {
        let mut stats = RobustStats::new(100);
        // 1, 2, 3, 4, 5, 6, 7, 8, 9 -> median = 5
        // deviations: 4, 3, 2, 1, 0, 1, 2, 3, 4 -> MAD = 2
        for i in 1..=9 {
            stats.add(i as f64);
        }

        let mad = stats.mad().unwrap();
        assert!((mad - 2.0 * 1.4826).abs() < 0.01);
    }

    #[test]
    fn test_cusum_no_change() {
        let mut cusum = CusumDetector::from_baseline(10.0, 1.0);

        // Samples around mean - no alarm
        for _ in 0..100 {
            let detected = cusum.update(10.0 + 0.1);
            assert!(!detected);
        }
    }

    #[test]
    fn test_cusum_upward_shift() {
        let mut cusum = CusumDetector::from_baseline(10.0, 1.0);

        // Significant upward shift
        let mut detected = false;
        for _ in 0..20 {
            if cusum.update(15.0) {
                detected = true;
                break;
            }
        }
        assert!(detected, "Should detect upward shift");
    }

    #[test]
    fn test_variance_tracker() {
        let mut tracker = VarianceTracker::new();
        for i in 1..=5 {
            tracker.update(i as f64);
        }

        assert_eq!(tracker.mean(), 3.0);
        assert!((tracker.variance() - 2.0).abs() < 0.01);
    }
}
