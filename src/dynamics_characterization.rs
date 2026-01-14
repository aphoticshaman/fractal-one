//! ═══════════════════════════════════════════════════════════════════════════════
//! DYNAMICS CHARACTERIZATION — Governing Equations, Not Internal State
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Extension to threshold_detector that characterizes the SHAPE of dynamics,
//! not just point values. Implements insights from white-box analysis:
//!
//! - Meta-stability: how stable are the measurements themselves?
//! - Operating envelope: what bounds does the system respect?
//! - Dynamics shape: linear, exponential, bounded, chaotic?
//! - Invariant detection: what properties never change?
//!
//! Key principle: "White-box at governing equations, not internal state"
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// META-STABILITY — Stability of the measurements themselves
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks stability of a measurement over time
#[derive(Debug, Clone)]
pub struct MetaStabilityTracker {
    /// Recent measurement values
    history: VecDeque<f32>,
    /// Maximum history length
    max_history: usize,
    /// Computed statistics
    mean: f32,
    variance: f32,
    /// Trend direction (-1, 0, +1)
    trend: i8,
}

impl MetaStabilityTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            mean: 0.0,
            variance: 0.0,
            trend: 0,
        }
    }

    /// Record a new measurement
    pub fn record(&mut self, value: f32) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(value);
        self.recompute();
    }

    fn recompute(&mut self) {
        let n = self.history.len();
        if n == 0 {
            return;
        }

        // Mean
        self.mean = self.history.iter().sum::<f32>() / n as f32;

        // Variance
        self.variance = if n > 1 {
            self.history
                .iter()
                .map(|x| (x - self.mean).powi(2))
                .sum::<f32>()
                / (n - 1) as f32
        } else {
            0.0
        };

        // Trend (simple linear regression sign)
        if n >= 3 {
            let first_half: f32 = self.history.iter().take(n / 2).sum::<f32>() / (n / 2) as f32;
            let second_half: f32 =
                self.history.iter().skip(n / 2).sum::<f32>() / (n - n / 2) as f32;
            let diff = second_half - first_half;
            self.trend = if diff > 0.05 {
                1
            } else if diff < -0.05 {
                -1
            } else {
                0
            };
        }
    }

    /// Meta-stability score [0, 1] — higher = more stable
    ///
    /// Computed as: 1 - normalized_variance
    /// Where normalized_variance = variance / max_possible_variance
    /// For `[0,1]` bounded values, max_variance = 0.25
    pub fn stability(&self) -> f32 {
        if self.history.len() < 3 {
            return 0.5; // Uncertain
        }
        let normalized = (self.variance / 0.25).min(1.0);
        1.0 - normalized
    }

    /// Coefficient of variation (std_dev / mean)
    pub fn cv(&self) -> f32 {
        if self.mean.abs() < 0.001 {
            return 0.0;
        }
        self.variance.sqrt() / self.mean.abs()
    }

    /// Current trend
    pub fn trend(&self) -> Trend {
        match self.trend {
            1 => Trend::Increasing,
            -1 => Trend::Decreasing,
            _ => Trend::Stable,
        }
    }

    /// Recent values for analysis
    pub fn recent(&self, n: usize) -> Vec<f32> {
        self.history.iter().rev().take(n).copied().collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Increasing,
    Stable,
    Decreasing,
}

// ═══════════════════════════════════════════════════════════════════════════════
// OPERATING ENVELOPE — Constraint boundary detection
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks operating bounds for a metric
#[derive(Debug, Clone)]
pub struct BoundTracker {
    /// Observed minimum
    pub min: f32,
    /// Observed maximum
    pub max: f32,
    /// Sample count
    pub n: usize,
    /// Running mean for percentile estimation
    mean: f32,
    /// Running M2 for variance
    m2: f32,
}

impl BoundTracker {
    pub fn new() -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Record observation using Welford's online algorithm
    pub fn observe(&mut self, value: f32) {
        self.n += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Welford's algorithm for running mean/variance
        let delta = value - self.mean;
        self.mean += delta / self.n as f32;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Range of observed values
    pub fn range(&self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        self.max - self.min
    }

    /// Standard deviation
    pub fn std_dev(&self) -> f32 {
        if self.n < 2 {
            return 0.0;
        }
        (self.m2 / (self.n - 1) as f32).sqrt()
    }

    /// Tightness: how narrow is the envelope? [0, 1]
    /// 1.0 = all values identical, 0.0 = values span full possible range
    pub fn tightness(&self, full_range: f32) -> f32 {
        if full_range <= 0.0 || self.n == 0 {
            return 0.0;
        }
        1.0 - (self.range() / full_range).min(1.0)
    }

    /// Would this value violate observed bounds?
    pub fn is_violation(&self, value: f32, margin: f32) -> bool {
        if self.n < 10 {
            return false; // Not enough data
        }
        value < self.min - margin || value > self.max + margin
    }

    /// Estimate percentile of a value within observed distribution
    pub fn percentile(&self, value: f32) -> f32 {
        if self.n == 0 || self.range() < 0.001 {
            return 0.5;
        }
        ((value - self.min) / self.range()).clamp(0.0, 1.0)
    }
}

impl Default for BoundTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Operating envelope across multiple metrics
#[derive(Debug, Clone)]
pub struct OperatingEnvelope {
    /// Bound trackers per metric
    metrics: HashMap<String, BoundTracker>,
    /// Violation events (metric, value, timestamp)
    violations: Vec<(String, f32, Instant)>,
    /// Max violations to retain
    max_violations: usize,
}

impl OperatingEnvelope {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            violations: Vec::new(),
            max_violations: 100,
        }
    }

    /// Record observation for a metric
    pub fn observe(&mut self, metric: &str, value: f32) {
        let tracker = self.metrics.entry(metric.to_string()).or_default();

        // Check for violation before updating
        if tracker.is_violation(value, tracker.std_dev() * 2.0) {
            if self.violations.len() >= self.max_violations {
                self.violations.remove(0);
            }
            self.violations
                .push((metric.to_string(), value, Instant::now()));
        }

        tracker.observe(value);
    }

    /// Get bounds for a metric
    pub fn bounds(&self, metric: &str) -> Option<(f32, f32)> {
        self.metrics.get(metric).map(|t| (t.min, t.max))
    }

    /// Get overall envelope tightness [0, 1]
    pub fn tightness(&self) -> f32 {
        if self.metrics.is_empty() {
            return 0.0;
        }

        let sum: f32 = self
            .metrics
            .values()
            .map(|t| t.tightness(1.0)) // Assume [0,1] range
            .sum();

        sum / self.metrics.len() as f32
    }

    /// Recent violations
    pub fn recent_violations(&self, n: usize) -> Vec<&(String, f32, Instant)> {
        self.violations.iter().rev().take(n).collect()
    }

    /// Violation rate (per metric count)
    pub fn violation_rate(&self) -> f32 {
        let total_obs: usize = self.metrics.values().map(|t| t.n).sum();
        if total_obs == 0 {
            return 0.0;
        }
        self.violations.len() as f32 / total_obs as f32
    }

    /// Get constraint strength for a metric [0, 1]
    /// Higher = tighter bounds = stronger constraints
    pub fn constraint_strength(&self, metric: &str) -> f32 {
        self.metrics
            .get(metric)
            .map(|t| t.tightness(1.0))
            .unwrap_or(0.0)
    }
}

impl Default for OperatingEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DYNAMICS SHAPE — Characterizing the governing equation
// ═══════════════════════════════════════════════════════════════════════════════

/// Shape of dynamics for a measurement
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicsShape {
    /// Insufficient data
    Unknown,
    /// Value remains approximately constant
    Constant { value: f32, tolerance: f32 },
    /// Linear growth or decay
    Linear { slope: f32, r_squared: f32 },
    /// Exponential growth or decay
    Exponential { rate: f32, r_squared: f32 },
    /// Bounded oscillation within range
    Bounded {
        min: f32,
        max: f32,
        period: Option<f32>,
    },
    /// No discernible pattern
    Chaotic { entropy: f32 },
}

impl DynamicsShape {
    pub fn name(&self) -> &'static str {
        match self {
            DynamicsShape::Unknown => "Unknown",
            DynamicsShape::Constant { .. } => "Constant",
            DynamicsShape::Linear { .. } => "Linear",
            DynamicsShape::Exponential { .. } => "Exponential",
            DynamicsShape::Bounded { .. } => "Bounded",
            DynamicsShape::Chaotic { .. } => "Chaotic",
        }
    }
}

/// Characterizes dynamics from a time series
pub struct DynamicsCharacterizer {
    /// Minimum samples for characterization
    min_samples: usize,
}

impl DynamicsCharacterizer {
    pub fn new(min_samples: usize) -> Self {
        Self { min_samples }
    }

    /// Characterize dynamics from samples (oldest first)
    pub fn characterize(&self, samples: &[f32]) -> DynamicsShape {
        if samples.len() < self.min_samples {
            return DynamicsShape::Unknown;
        }

        let n = samples.len();

        // Basic statistics
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let variance: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std_dev = variance.sqrt();
        let min = samples.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max = samples.iter().fold(f32::MIN, |a, &b| a.max(b));
        let range = max - min;

        // Check for constant
        if std_dev < 0.01 || (range / mean.abs().max(0.001)) < 0.1 {
            return DynamicsShape::Constant {
                value: mean,
                tolerance: std_dev,
            };
        }

        // Linear regression
        let (slope, r_sq_linear) = self.linear_fit(samples);

        // Check for strong linear trend
        if r_sq_linear > 0.8 {
            return DynamicsShape::Linear {
                slope,
                r_squared: r_sq_linear,
            };
        }

        // Check for exponential (fit log-transformed)
        let positive: Vec<f32> = samples.iter().map(|&x| x.max(0.001)).collect();
        let log_samples: Vec<f32> = positive.iter().map(|x| x.ln()).collect();
        let (rate, r_sq_exp) = self.linear_fit(&log_samples);

        if r_sq_exp > 0.8 && r_sq_exp > r_sq_linear {
            return DynamicsShape::Exponential {
                rate,
                r_squared: r_sq_exp,
            };
        }

        // Check for bounded
        if range > 0.1 && r_sq_linear < 0.3 {
            // Look for oscillation (simple autocorrelation check)
            let period = self.detect_period(samples);
            return DynamicsShape::Bounded { min, max, period };
        }

        // Default to chaotic
        let entropy = self.estimate_entropy(samples);
        DynamicsShape::Chaotic { entropy }
    }

    /// Simple linear regression, returns (slope, r_squared)
    fn linear_fit(&self, samples: &[f32]) -> (f32, f32) {
        let n = samples.len() as f32;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        // x = [0, 1, 2, ...]
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f32 = samples.iter().sum::<f32>() / n;

        let mut ss_xy = 0.0f32;
        let mut ss_xx = 0.0f32;
        let mut ss_yy = 0.0f32;

        for (i, &y) in samples.iter().enumerate() {
            let x = i as f32;
            ss_xy += (x - x_mean) * (y - y_mean);
            ss_xx += (x - x_mean).powi(2);
            ss_yy += (y - y_mean).powi(2);
        }

        if ss_xx < 0.001 || ss_yy < 0.001 {
            return (0.0, 0.0);
        }

        let slope = ss_xy / ss_xx;
        let r_squared = (ss_xy.powi(2) / (ss_xx * ss_yy)).min(1.0);

        (slope, r_squared)
    }

    /// Detect dominant period via autocorrelation
    fn detect_period(&self, samples: &[f32]) -> Option<f32> {
        if samples.len() < 10 {
            return None;
        }

        let n = samples.len();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let centered: Vec<f32> = samples.iter().map(|x| x - mean).collect();

        // Compute autocorrelation for lags 2 to n/2
        let mut max_corr = 0.0f32;
        let mut best_lag = 0usize;

        for lag in 2..n / 2 {
            let mut corr = 0.0f32;
            for i in 0..n - lag {
                corr += centered[i] * centered[i + lag];
            }
            corr /= (n - lag) as f32;

            if corr > max_corr {
                max_corr = corr;
                best_lag = lag;
            }
        }

        if max_corr > 0.3 * centered.iter().map(|x| x.powi(2)).sum::<f32>() / n as f32 {
            Some(best_lag as f32)
        } else {
            None
        }
    }

    /// Estimate entropy via binning
    fn estimate_entropy(&self, samples: &[f32]) -> f32 {
        let n_bins = 10;
        let min = samples.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max = samples.iter().fold(f32::MIN, |a, &b| a.max(b));
        let range = max - min;

        if range < 0.001 {
            return 0.0;
        }

        let mut bins = vec![0usize; n_bins];
        for &x in samples {
            let bin = (((x - min) / range) * (n_bins - 1) as f32) as usize;
            bins[bin.min(n_bins - 1)] += 1;
        }

        let n = samples.len() as f32;
        let mut entropy = 0.0f32;
        for count in bins {
            if count > 0 {
                let p = count as f32 / n;
                entropy -= p * p.ln();
            }
        }

        // Normalize to [0, 1]
        let max_entropy = (n_bins as f32).ln();
        entropy / max_entropy
    }
}

impl Default for DynamicsCharacterizer {
    fn default() -> Self {
        Self::new(10)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INVARIANT DETECTION — What never changes?
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks potential invariants (properties that remain constant)
#[derive(Debug, Clone)]
pub struct InvariantDetector {
    /// Property observations: name -> values
    observations: HashMap<String, Vec<f32>>,
    /// Detected invariants: name -> (value, confidence)
    invariants: HashMap<String, (f32, f32)>,
    /// Threshold for invariance (max coefficient of variation)
    cv_threshold: f32,
    /// Minimum observations for invariant detection
    min_observations: usize,
}

impl InvariantDetector {
    pub fn new(cv_threshold: f32, min_observations: usize) -> Self {
        Self {
            observations: HashMap::new(),
            invariants: HashMap::new(),
            cv_threshold,
            min_observations,
        }
    }

    /// Observe a property value
    pub fn observe(&mut self, property: &str, value: f32) {
        self.observations
            .entry(property.to_string())
            .or_default()
            .push(value);

        // Recompute invariant status for this property
        self.check_invariant(property);
    }

    fn check_invariant(&mut self, property: &str) {
        if let Some(values) = self.observations.get(property) {
            if values.len() < self.min_observations {
                return;
            }

            let n = values.len() as f32;
            let mean: f32 = values.iter().sum::<f32>() / n;

            if mean.abs() < 0.001 {
                // Near-zero mean, check absolute variance
                let max_dev = values
                    .iter()
                    .map(|x| (x - mean).abs())
                    .fold(0.0f32, f32::max);

                if max_dev < 0.01 {
                    let confidence = 1.0 - max_dev / 0.01;
                    self.invariants
                        .insert(property.to_string(), (mean, confidence));
                } else {
                    self.invariants.remove(property);
                }
            } else {
                // Use coefficient of variation
                let variance: f32 = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                let cv = variance.sqrt() / mean.abs();

                if cv < self.cv_threshold {
                    let confidence = 1.0 - cv / self.cv_threshold;
                    self.invariants
                        .insert(property.to_string(), (mean, confidence));
                } else {
                    self.invariants.remove(property);
                }
            }
        }
    }

    /// Get detected invariants
    pub fn invariants(&self) -> &HashMap<String, (f32, f32)> {
        &self.invariants
    }

    /// Count of invariant properties
    pub fn invariant_count(&self) -> usize {
        self.invariants.len()
    }

    /// Invariant strength: ratio of invariant properties to observed properties
    pub fn invariant_strength(&self) -> f32 {
        if self.observations.is_empty() {
            return 0.0;
        }
        self.invariants.len() as f32 / self.observations.len() as f32
    }

    /// Check if a property is invariant
    pub fn is_invariant(&self, property: &str) -> Option<(f32, f32)> {
        self.invariants.get(property).copied()
    }
}

impl Default for InvariantDetector {
    fn default() -> Self {
        Self::new(0.05, 20) // 5% CV threshold, 20 min observations
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL CHARACTERIZATION — Combines all components
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete dynamics characterization for an axis
#[derive(Debug, Clone)]
pub struct AxisCharacterization {
    /// Current value
    pub value: f32,
    /// Meta-stability of measurements
    pub meta_stability: f32,
    /// Current trend
    pub trend: Trend,
    /// Dynamics shape
    pub shape: DynamicsShape,
    /// Constraint tightness for this axis
    pub constraint_tightness: f32,
}

/// Full system characterization
#[derive(Debug)]
pub struct SystemCharacterization {
    pub p: AxisCharacterization,
    pub c: AxisCharacterization,
    pub a: AxisCharacterization,
    pub invariant_count: usize,
    pub invariant_strength: f32,
    pub envelope_tightness: f32,
    pub violation_rate: f32,
}

/// Characterization engine combining all components
pub struct CharacterizationEngine {
    /// Meta-stability trackers per axis
    pub p_stability: MetaStabilityTracker,
    pub c_stability: MetaStabilityTracker,
    pub a_stability: MetaStabilityTracker,
    /// Operating envelope
    pub envelope: OperatingEnvelope,
    /// Invariant detector
    pub invariants: InvariantDetector,
    /// Dynamics characterizer
    characterizer: DynamicsCharacterizer,
}

impl CharacterizationEngine {
    pub fn new() -> Self {
        Self {
            p_stability: MetaStabilityTracker::new(50),
            c_stability: MetaStabilityTracker::new(50),
            a_stability: MetaStabilityTracker::new(50),
            envelope: OperatingEnvelope::new(),
            invariants: InvariantDetector::default(),
            characterizer: DynamicsCharacterizer::default(),
        }
    }

    /// Record measurements from threshold detector
    pub fn record(&mut self, p: f32, c: f32, a: f32) {
        // Update stability trackers
        self.p_stability.record(p);
        self.c_stability.record(c);
        self.a_stability.record(a);

        // Update envelope
        self.envelope.observe("P", p);
        self.envelope.observe("C", c);
        self.envelope.observe("A", a);

        // Update invariants (using derived properties)
        self.invariants
            .observe("P_C_ratio", if c > 0.001 { p / c } else { 0.0 });
        self.invariants.observe("A_magnitude", a);
    }

    /// Get full system characterization
    pub fn characterize(&self) -> SystemCharacterization {
        SystemCharacterization {
            p: AxisCharacterization {
                value: self.p_stability.mean,
                meta_stability: self.p_stability.stability(),
                trend: self.p_stability.trend(),
                shape: self
                    .characterizer
                    .characterize(&self.p_stability.recent(50)),
                constraint_tightness: self.envelope.constraint_strength("P"),
            },
            c: AxisCharacterization {
                value: self.c_stability.mean,
                meta_stability: self.c_stability.stability(),
                trend: self.c_stability.trend(),
                shape: self
                    .characterizer
                    .characterize(&self.c_stability.recent(50)),
                constraint_tightness: self.envelope.constraint_strength("C"),
            },
            a: AxisCharacterization {
                value: self.a_stability.mean,
                meta_stability: self.a_stability.stability(),
                trend: self.a_stability.trend(),
                shape: self
                    .characterizer
                    .characterize(&self.a_stability.recent(50)),
                constraint_tightness: self.envelope.constraint_strength("A"),
            },
            invariant_count: self.invariants.invariant_count(),
            invariant_strength: self.invariants.invariant_strength(),
            envelope_tightness: self.envelope.tightness(),
            violation_rate: self.envelope.violation_rate(),
        }
    }
}

impl Default for CharacterizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_stability_constant() {
        let mut tracker = MetaStabilityTracker::new(20);
        for _ in 0..20 {
            tracker.record(0.5);
        }
        assert!(tracker.stability() > 0.95);
        assert_eq!(tracker.trend(), Trend::Stable);
    }

    #[test]
    fn test_meta_stability_increasing() {
        let mut tracker = MetaStabilityTracker::new(20);
        for i in 0..20 {
            tracker.record(i as f32 / 20.0);
        }
        assert_eq!(tracker.trend(), Trend::Increasing);
    }

    #[test]
    fn test_operating_envelope() {
        let mut envelope = OperatingEnvelope::new();
        for i in 0..100 {
            envelope.observe("test", 0.4 + 0.1 * (i as f32 / 100.0));
        }

        let bounds = envelope.bounds("test").unwrap();
        assert!(bounds.0 >= 0.4);
        assert!(bounds.1 <= 0.5);
        assert!(envelope.tightness() > 0.8);
    }

    #[test]
    fn test_dynamics_constant() {
        let characterizer = DynamicsCharacterizer::default();
        let samples: Vec<f32> = (0..20).map(|_| 0.5).collect();
        let shape = characterizer.characterize(&samples);

        assert!(matches!(shape, DynamicsShape::Constant { .. }));
    }

    #[test]
    fn test_dynamics_linear() {
        let characterizer = DynamicsCharacterizer::default();
        let samples: Vec<f32> = (0..20).map(|i| i as f32 * 0.05).collect();
        let shape = characterizer.characterize(&samples);

        assert!(matches!(shape, DynamicsShape::Linear { .. }));
    }

    #[test]
    fn test_invariant_detection() {
        let mut detector = InvariantDetector::default();
        for _ in 0..30 {
            detector.observe("constant", 0.5);
            detector.observe("varying", rand_simple());
        }

        assert!(detector.is_invariant("constant").is_some());
        assert!(detector.is_invariant("varying").is_none());
    }

    fn rand_simple() -> f32 {
        use std::time::SystemTime;
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        (t % 1000) as f32 / 1000.0
    }
}
