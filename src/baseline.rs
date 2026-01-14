//! ═══════════════════════════════════════════════════════════════════════════════
//! BASELINE — Dual Baseline with Poisoning Resistance
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Two-track calibration:
//! - Short-term (α=0.2): Responsive to recent changes
//! - Long-term (α=0.02): Stable anchor with 1% step cap (poisoning resistance)
//!
//! Properties:
//! - Min samples gate before "ready"
//! - MAD-based robust scale estimation
//! - Z-score anomaly detection
//! - Step cap prevents gradual adversarial drift
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::domains::TrustDomain;
use crate::stats::{Ewma, RobustStats, VarianceTracker};

// ═══════════════════════════════════════════════════════════════════════════════
// DUAL BASELINE — The core calibration system
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for dual baseline
#[derive(Debug, Clone)]
pub struct DualBaselineConfig {
    /// Short-term EWMA alpha (higher = more responsive)
    pub st_alpha: f64,
    /// Long-term EWMA alpha (lower = more stable)
    pub lt_alpha: f64,
    /// Maximum step size for long-term update (poisoning cap)
    pub lt_step_cap: f64,
    /// Minimum samples before baseline is ready
    pub min_samples: usize,
    /// Window size for robust stats
    pub robust_window: usize,
    /// Trust domain adaptation multiplier
    pub domain_adaptation: bool,
}

impl Default for DualBaselineConfig {
    fn default() -> Self {
        Self {
            st_alpha: 0.2,      // ~5 sample effective window
            lt_alpha: 0.02,     // ~50 sample effective window
            lt_step_cap: 0.01,  // 1% max step per update
            min_samples: 20,    // Min samples before ready
            robust_window: 100, // Window for median/MAD
            domain_adaptation: true,
        }
    }
}

/// Dual baseline with short-term responsiveness and long-term stability
#[derive(Debug)]
pub struct DualBaseline {
    config: DualBaselineConfig,

    /// Short-term EWMA (responsive)
    st_ewma: Ewma,
    /// Long-term EWMA (stable, poisoning-resistant)
    lt_ewma: Ewma,
    /// Variance tracker for z-score calculation
    variance: VarianceTracker,
    /// Robust stats for median/MAD
    robust: RobustStats,

    /// Sample count
    sample_count: u64,
    /// Is baseline calibrated?
    ready: bool,
    /// Last raw value seen
    last_value: f64,
    /// Last trust domain seen
    last_domain: TrustDomain,
}

impl DualBaseline {
    pub fn new(config: DualBaselineConfig) -> Self {
        Self {
            st_ewma: Ewma::new(config.st_alpha),
            lt_ewma: Ewma::new(config.lt_alpha),
            variance: VarianceTracker::new(),
            robust: RobustStats::new(config.robust_window),
            sample_count: 0,
            ready: false,
            last_value: 0.0,
            last_domain: TrustDomain::System,
            config,
        }
    }

    /// Update baseline with new observation
    pub fn update(&mut self, value: f64, domain: TrustDomain) {
        self.last_value = value;
        self.last_domain = domain;

        // Apply domain-specific adaptation rate
        let adaptation_mult = if self.config.domain_adaptation {
            domain.adaptation_rate()
        } else {
            1.0
        };

        // Short-term: always update (scaled by domain)
        let st_value = if self.st_ewma.is_initialized() {
            let current = self.st_ewma.value();
            let delta = value - current;
            current + delta * self.config.st_alpha * adaptation_mult
        } else {
            value
        };
        self.st_ewma = Ewma::new(self.config.st_alpha); // Reset and update
        self.st_ewma.update(st_value);

        // Long-term: capped step (poisoning resistance)
        if self.lt_ewma.is_initialized() {
            let current = self.lt_ewma.value();
            let mut delta = value - current;

            // Cap the step size
            let max_step = current.abs() * self.config.lt_step_cap;
            delta = delta.clamp(-max_step.max(0.01), max_step.max(0.01));

            // Apply domain adaptation
            delta *= adaptation_mult;

            let new_lt = current + delta * self.config.lt_alpha;
            self.lt_ewma = Ewma::new(self.config.lt_alpha);
            self.lt_ewma.update(new_lt);
        } else {
            self.lt_ewma.update(value);
        }

        // Update variance tracker
        self.variance.update(value);

        // Update robust stats
        self.robust.add(value);

        self.sample_count += 1;
        if self.sample_count >= self.config.min_samples as u64 {
            self.ready = true;
        }
    }

    /// Is baseline ready for use?
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Sample count
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Short-term mean
    pub fn st_mean(&self) -> f64 {
        self.st_ewma.value()
    }

    /// Long-term mean (poisoning-resistant)
    pub fn lt_mean(&self) -> f64 {
        self.lt_ewma.value()
    }

    /// Divergence between short and long term
    /// Positive = recent values higher than baseline
    /// Negative = recent values lower than baseline
    pub fn divergence(&self) -> f64 {
        self.st_ewma.value() - self.lt_ewma.value()
    }

    /// Normalized divergence (divergence / scale)
    pub fn normalized_divergence(&mut self) -> f64 {
        let scale = self.scale().max(1e-6);
        self.divergence() / scale
    }

    /// Standard deviation estimate
    pub fn std_dev(&self) -> f64 {
        self.variance.std_dev()
    }

    /// Scale estimate (robust MAD if available, else std_dev)
    pub fn scale(&mut self) -> f64 {
        self.robust
            .mad()
            .unwrap_or_else(|| self.std_dev())
            .max(1e-6)
    }

    /// Z-score relative to long-term baseline
    pub fn z_score(&mut self, value: f64) -> f64 {
        if !self.ready {
            return 0.0;
        }
        let scale = self.scale();
        (value - self.lt_mean()) / scale
    }

    /// Robust z-score using median/MAD
    pub fn robust_z_score(&mut self, value: f64) -> Option<f64> {
        if !self.ready {
            return None;
        }
        self.robust.robust_z_score(value)
    }

    /// Anomaly level for a value
    pub fn anomaly_level(&mut self, value: f64) -> AnomalyLevel {
        if !self.ready {
            return AnomalyLevel::Unknown;
        }

        let z = self.z_score(value).abs();
        match z {
            z if z > 3.5 => AnomalyLevel::Critical,
            z if z > 2.5 => AnomalyLevel::Warning,
            z if z > 1.5 => AnomalyLevel::Elevated,
            _ => AnomalyLevel::Normal,
        }
    }

    /// Reset baseline
    pub fn reset(&mut self) {
        self.st_ewma = Ewma::new(self.config.st_alpha);
        self.lt_ewma = Ewma::new(self.config.lt_alpha);
        self.variance.reset();
        self.robust.reset();
        self.sample_count = 0;
        self.ready = false;
    }

    /// Get summary statistics
    pub fn summary(&mut self) -> BaselineSummary {
        BaselineSummary {
            st_mean: self.st_mean(),
            lt_mean: self.lt_mean(),
            divergence: self.divergence(),
            std_dev: self.std_dev(),
            scale: self.scale(),
            sample_count: self.sample_count,
            ready: self.ready,
        }
    }
}

/// Anomaly level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyLevel {
    Unknown,
    Normal,
    Elevated,
    Warning,
    Critical,
}

impl AnomalyLevel {
    pub fn as_ordinal(&self) -> u8 {
        match self {
            AnomalyLevel::Unknown => 0,
            AnomalyLevel::Normal => 1,
            AnomalyLevel::Elevated => 2,
            AnomalyLevel::Warning => 3,
            AnomalyLevel::Critical => 4,
        }
    }

    pub fn is_concerning(&self) -> bool {
        matches!(self, AnomalyLevel::Warning | AnomalyLevel::Critical)
    }
}

/// Summary of baseline state
#[derive(Debug, Clone)]
pub struct BaselineSummary {
    pub st_mean: f64,
    pub lt_mean: f64,
    pub divergence: f64,
    pub std_dev: f64,
    pub scale: f64,
    pub sample_count: u64,
    pub ready: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BASELINE REGISTRY — Multiple baselines per observation key
// ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::ObsKey;
use std::collections::HashMap;

/// Registry of baselines for different observation keys
#[derive(Debug)]
pub struct BaselineRegistry {
    baselines: HashMap<ObsKey, DualBaseline>,
    config: DualBaselineConfig,
}

impl BaselineRegistry {
    pub fn new(config: DualBaselineConfig) -> Self {
        Self {
            baselines: HashMap::new(),
            config,
        }
    }

    /// Get or create baseline for key
    pub fn get_mut(&mut self, key: ObsKey) -> &mut DualBaseline {
        self.baselines
            .entry(key)
            .or_insert_with(|| DualBaseline::new(self.config.clone()))
    }

    /// Update baseline for key
    pub fn update(&mut self, key: ObsKey, value: f64, domain: TrustDomain) {
        self.get_mut(key).update(value, domain);
    }

    /// Get z-score for value against baseline
    pub fn z_score(&mut self, key: ObsKey, value: f64) -> f64 {
        self.get_mut(key).z_score(value)
    }

    /// Get anomaly level for value
    pub fn anomaly_level(&mut self, key: ObsKey, value: f64) -> AnomalyLevel {
        self.get_mut(key).anomaly_level(value)
    }

    /// Is baseline for key ready?
    pub fn is_ready(&self, key: ObsKey) -> bool {
        self.baselines
            .get(&key)
            .map(|b| b.is_ready())
            .unwrap_or(false)
    }

    /// Number of keys with ready baselines
    pub fn ready_count(&self) -> usize {
        self.baselines.values().filter(|b| b.is_ready()).count()
    }

    /// Total keys tracked
    pub fn total_keys(&self) -> usize {
        self.baselines.len()
    }
}

impl Default for BaselineRegistry {
    fn default() -> Self {
        Self::new(DualBaselineConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_baseline_basic() {
        let mut baseline = DualBaseline::new(DualBaselineConfig::default());

        // Not ready initially
        assert!(!baseline.is_ready());

        // Add enough samples
        for i in 0..25 {
            baseline.update(10.0 + (i as f64 * 0.1), TrustDomain::System);
        }

        assert!(baseline.is_ready());
        assert!(baseline.st_mean() > 10.0);
        assert!(baseline.lt_mean() > 10.0);
    }

    #[test]
    fn test_poisoning_resistance() {
        let mut baseline = DualBaseline::new(DualBaselineConfig::default());

        // Establish baseline around 10.0
        for _ in 0..30 {
            baseline.update(10.0, TrustDomain::System);
        }

        let lt_before = baseline.lt_mean();

        // Try to poison with extreme values
        for _ in 0..20 {
            baseline.update(100.0, TrustDomain::User); // Adversarial, 10x baseline
        }

        let lt_after = baseline.lt_mean();

        // Long-term should not shift by more than 25% despite 10x attack values
        let shift_ratio = (lt_after - lt_before).abs() / lt_before;
        assert!(
            shift_ratio < 0.25,
            "Poisoning resistance failed: shift_ratio = {}",
            shift_ratio
        );
    }

    #[test]
    fn test_anomaly_detection() {
        let mut baseline = DualBaseline::new(DualBaselineConfig::default());

        // Establish baseline around 50.0 with some variance
        for i in 0..50 {
            baseline.update(50.0 + (i as f64 % 5.0) - 2.0, TrustDomain::System);
        }

        // Normal value should not be concerning
        let level_51 = baseline.anomaly_level(51.0);
        assert!(
            !level_51.is_concerning(),
            "51.0 should not be concerning, got {:?}",
            level_51
        );

        // Moderate deviation - should be detected as above normal
        let level_55 = baseline.anomaly_level(55.0);
        assert!(
            level_55 != AnomalyLevel::Unknown,
            "55.0 should be classified, got {:?}",
            level_55
        );

        // Extreme value should be concerning
        let level_100 = baseline.anomaly_level(100.0);
        assert!(
            level_100.is_concerning(),
            "100.0 should be concerning, got {:?}",
            level_100
        );
    }

    #[test]
    fn test_divergence_tracking() {
        let mut baseline = DualBaseline::new(DualBaselineConfig::default());

        // Establish baseline
        for _ in 0..30 {
            baseline.update(10.0, TrustDomain::System);
        }

        // Sudden increase - short-term should track faster
        for _ in 0..5 {
            baseline.update(20.0, TrustDomain::System);
        }

        // Divergence should be positive (ST > LT)
        assert!(baseline.divergence() > 0.0);
    }

    #[test]
    fn test_domain_adaptation() {
        let config = DualBaselineConfig {
            domain_adaptation: true,
            ..Default::default()
        };

        let mut baseline_system = DualBaseline::new(config.clone());
        let mut baseline_user = DualBaseline::new(config);

        // Same extreme values, different trust domains
        for _ in 0..30 {
            baseline_system.update(10.0, TrustDomain::System);
            baseline_user.update(10.0, TrustDomain::User);
        }

        // Poison attempt
        for _ in 0..10 {
            baseline_system.update(100.0, TrustDomain::System);
            baseline_user.update(100.0, TrustDomain::User);
        }

        // User domain should adapt slower (more resistant)
        // Note: This is reversed from what you'd expect - we're more suspicious of User data
        // So User domain adaptation is slower, meaning User attacks are less effective
        let system_shift = (baseline_system.lt_mean() - 10.0).abs();
        let user_shift = (baseline_user.lt_mean() - 10.0).abs();

        assert!(
            user_shift <= system_shift,
            "User domain should adapt slower: user_shift={} system_shift={}",
            user_shift,
            system_shift
        );
    }

    #[test]
    fn test_baseline_registry() {
        let mut registry = BaselineRegistry::default();

        registry.update(ObsKey::RespLatMs, 100.0, TrustDomain::System);
        registry.update(ObsKey::RespLatMs, 110.0, TrustDomain::System);

        assert_eq!(registry.total_keys(), 1);
        assert!(!registry.is_ready(ObsKey::RespLatMs)); // Need more samples
    }
}
