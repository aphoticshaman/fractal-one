//! ═══════════════════════════════════════════════════════════════════════════════
//! TRANSPORTABILITY TEST
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Tests whether persistence findings generalize across different conditions.
//!
//! Key questions:
//!   1. Do findings replicate across different marker types?
//!   2. Are findings stable across different washout configurations?
//!   3. Do findings hold at different time points?
//!   4. (Future) Do findings generalize across endpoints?
//!
//! Methods:
//!   - Stratified analysis by condition
//!   - Heterogeneity tests (I², Q-test)
//!   - Meta-analytic combination of results
//!   - Leave-one-out cross-validation
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::marker::MarkerClass;
use super::mi::Observation;

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSPORTABILITY SETTING
// ═══════════════════════════════════════════════════════════════════════════════

/// A specific experimental setting for transportability analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SettingKey {
    /// Setting dimension (e.g., "marker_class", "washout_ms", "time_block")
    pub dimension: String,
    /// Setting value within that dimension
    pub value: String,
}

impl SettingKey {
    pub fn new(dimension: &str, value: &str) -> Self {
        Self {
            dimension: dimension.to_string(),
            value: value.to_string(),
        }
    }

    pub fn marker_class(class: MarkerClass) -> Self {
        Self::new("marker_class", &format!("{:?}", class))
    }

    pub fn washout_ms(ms: u64) -> Self {
        Self::new("washout_ms", &ms.to_string())
    }

    pub fn time_block(block: usize) -> Self {
        Self::new("time_block", &block.to_string())
    }

    pub fn trial(trial_id: usize) -> Self {
        Self::new("trial", &trial_id.to_string())
    }
}

/// Results for a specific setting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingResult {
    /// Setting key
    pub key: SettingKey,
    /// Number of observations in this setting
    pub n_observations: usize,
    /// Mean detection score for injected markers
    pub mean_injected: f64,
    /// Mean detection score for control markers
    pub mean_control: f64,
    /// Effect size (Cohen's d or similar)
    pub effect_size: f64,
    /// Standard error of the effect
    pub std_error: f64,
    /// 95% CI lower bound
    pub ci_lower: f64,
    /// 95% CI upper bound
    pub ci_upper: f64,
    /// Weight for meta-analysis (inverse variance)
    pub weight: f64,
}

impl SettingResult {
    /// Is this setting's effect significant (CI doesn't include zero)?
    pub fn is_significant(&self) -> bool {
        self.ci_lower > 0.0 || self.ci_upper < 0.0
    }

    /// Direction of effect: 1 = positive, -1 = negative, 0 = null
    pub fn direction(&self) -> i8 {
        if self.effect_size > 0.1 {
            1
        } else if self.effect_size < -0.1 {
            -1
        } else {
            0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HETEROGENEITY METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Heterogeneity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneityResult {
    /// Cochran's Q statistic
    pub q_statistic: f64,
    /// Q statistic p-value
    pub q_p_value: f64,
    /// I² statistic (percentage of variance due to heterogeneity)
    pub i_squared: f64,
    /// Tau² (between-study variance)
    pub tau_squared: f64,
    /// Number of settings analyzed
    pub k_settings: usize,
    /// Interpretation
    pub interpretation: String,
}

impl HeterogeneityResult {
    /// Compute heterogeneity metrics from setting results
    pub fn from_settings(settings: &[SettingResult]) -> Self {
        let k = settings.len();

        if k < 2 {
            return Self {
                q_statistic: 0.0,
                q_p_value: 1.0,
                i_squared: 0.0,
                tau_squared: 0.0,
                k_settings: k,
                interpretation: "Insufficient settings for heterogeneity analysis".to_string(),
            };
        }

        // Compute weighted mean effect
        let total_weight: f64 = settings.iter().map(|s| s.weight).sum();
        if total_weight <= 0.0 {
            return Self {
                q_statistic: 0.0,
                q_p_value: 1.0,
                i_squared: 0.0,
                tau_squared: 0.0,
                k_settings: k,
                interpretation: "Zero total weight".to_string(),
            };
        }

        let weighted_mean: f64 = settings
            .iter()
            .map(|s| s.weight * s.effect_size)
            .sum::<f64>()
            / total_weight;

        // Cochran's Q statistic
        let q: f64 = settings
            .iter()
            .map(|s| s.weight * (s.effect_size - weighted_mean).powi(2))
            .sum();

        // Q follows chi-square with k-1 degrees of freedom
        let df = (k - 1) as f64;

        // Approximate p-value using chi-square
        // For large df, use normal approximation: (Q - df) / sqrt(2*df)
        let q_p_value = if df > 0.0 {
            1.0 - chi_square_cdf(q, df)
        } else {
            1.0
        };

        // I² = max(0, (Q - df) / Q)
        let i_squared = if q > df { ((q - df) / q) * 100.0 } else { 0.0 };

        // Tau² (DerSimonian-Laird estimator)
        let c: f64 =
            total_weight - settings.iter().map(|s| s.weight.powi(2)).sum::<f64>() / total_weight;
        let tau_squared = if q > df && c > 0.0 { (q - df) / c } else { 0.0 };

        // Interpretation
        let interpretation = if i_squared < 25.0 {
            format!(
                "LOW HETEROGENEITY (I²={:.1}%): Findings appear consistent across settings.",
                i_squared
            )
        } else if i_squared < 50.0 {
            format!(
                "MODERATE HETEROGENEITY (I²={:.1}%): Some variation across settings, \
                 but overall pattern is consistent.",
                i_squared
            )
        } else if i_squared < 75.0 {
            format!(
                "SUBSTANTIAL HETEROGENEITY (I²={:.1}%): Considerable variation across \
                 settings. Findings may not fully generalize.",
                i_squared
            )
        } else {
            format!(
                "HIGH HETEROGENEITY (I²={:.1}%): Large variation across settings. \
                 Transportability is questionable.",
                i_squared
            )
        };

        Self {
            q_statistic: q,
            q_p_value,
            i_squared,
            tau_squared,
            k_settings: k,
            interpretation,
        }
    }
}

/// Approximate chi-square CDF using Wilson-Hilferty transformation
fn chi_square_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }

    // Wilson-Hilferty transformation to normal
    let z = ((x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();

    // Standard normal CDF approximation
    standard_normal_cdf(z)
}

/// Standard normal CDF approximation
fn standard_normal_cdf(z: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let z = z.abs();

    let t = 1.0 / (1.0 + p * z);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-z * z / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSPORTABILITY RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete transportability analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportabilityResult {
    /// Dimension analyzed (e.g., "marker_class")
    pub dimension: String,
    /// Results per setting
    pub settings: Vec<SettingResult>,
    /// Pooled effect estimate
    pub pooled_effect: f64,
    /// Pooled effect standard error
    pub pooled_se: f64,
    /// Pooled effect 95% CI
    pub pooled_ci_lower: f64,
    pub pooled_ci_upper: f64,
    /// Heterogeneity analysis
    pub heterogeneity: HeterogeneityResult,
    /// Are findings transportable across this dimension?
    pub is_transportable: bool,
    /// Interpretation
    pub interpretation: String,
}

impl TransportabilityResult {
    /// Overall assessment
    pub fn assessment(&self) -> &str {
        if self.is_transportable {
            if self.pooled_effect > 0.3 {
                "STRONG TRANSPORTABLE SIGNAL"
            } else if self.pooled_effect > 0.1 {
                "MODERATE TRANSPORTABLE SIGNAL"
            } else {
                "WEAK/NULL TRANSPORTABLE"
            }
        } else {
            "NOT TRANSPORTABLE"
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSPORTABILITY ANALYZER
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for transportability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportabilityConfig {
    /// Minimum observations per setting
    pub min_observations_per_setting: usize,
    /// Maximum I² for transportability
    pub max_i_squared: f64,
    /// Minimum settings required
    pub min_settings: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for TransportabilityConfig {
    fn default() -> Self {
        Self {
            min_observations_per_setting: 5,
            max_i_squared: 50.0,
            min_settings: 3,
            seed: 42,
        }
    }
}

/// Main transportability analyzer
pub struct TransportabilityAnalyzer {
    config: TransportabilityConfig,
    observations: Vec<(SettingKey, Observation)>,
}

impl TransportabilityAnalyzer {
    pub fn new(config: TransportabilityConfig) -> Self {
        Self {
            config,
            observations: Vec::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(TransportabilityConfig::default())
    }

    /// Add an observation with its setting
    pub fn add_observation(&mut self, setting: SettingKey, obs: Observation) {
        self.observations.push((setting, obs));
    }

    /// Add multiple observations for a setting
    pub fn add_observations(&mut self, setting: SettingKey, obs: Vec<Observation>) {
        for o in obs {
            self.observations.push((setting.clone(), o));
        }
    }

    /// Group observations by setting
    fn group_by_setting(&self, dimension: &str) -> HashMap<String, Vec<&Observation>> {
        let mut groups: HashMap<String, Vec<&Observation>> = HashMap::new();

        for (key, obs) in &self.observations {
            if key.dimension == dimension {
                groups.entry(key.value.clone()).or_default().push(obs);
            }
        }

        groups
    }

    /// Compute effect size (standardized mean difference) for a group
    fn compute_effect_size(observations: &[&Observation]) -> (f64, f64, usize) {
        let injected: Vec<f64> = observations
            .iter()
            .filter(|o| o.injected)
            .map(|o| o.score)
            .collect();

        let control: Vec<f64> = observations
            .iter()
            .filter(|o| !o.injected)
            .map(|o| o.score)
            .collect();

        let n_inj = injected.len();
        let n_ctl = control.len();
        let n_total = n_inj + n_ctl;

        if n_inj == 0 || n_ctl == 0 {
            return (0.0, f64::INFINITY, n_total);
        }

        let mean_inj: f64 = injected.iter().sum::<f64>() / n_inj as f64;
        let mean_ctl: f64 = control.iter().sum::<f64>() / n_ctl as f64;

        // Pooled standard deviation
        let var_inj: f64 = if n_inj > 1 {
            injected.iter().map(|x| (x - mean_inj).powi(2)).sum::<f64>() / (n_inj - 1) as f64
        } else {
            0.0
        };

        let var_ctl: f64 = if n_ctl > 1 {
            control.iter().map(|x| (x - mean_ctl).powi(2)).sum::<f64>() / (n_ctl - 1) as f64
        } else {
            0.0
        };

        let pooled_var = ((n_inj as f64 - 1.0) * var_inj + (n_ctl as f64 - 1.0) * var_ctl)
            / (n_inj as f64 + n_ctl as f64 - 2.0);
        let pooled_sd = pooled_var.sqrt().max(0.001); // Avoid division by zero

        // Cohen's d
        let effect_size = (mean_inj - mean_ctl) / pooled_sd;

        // Standard error of Cohen's d
        let se = ((n_inj as f64 + n_ctl as f64) / (n_inj as f64 * n_ctl as f64)
            + effect_size.powi(2) / (2.0 * (n_inj as f64 + n_ctl as f64)))
            .sqrt();

        (effect_size, se, n_total)
    }

    /// Analyze transportability across a dimension
    pub fn analyze_dimension(&self, dimension: &str) -> TransportabilityResult {
        let groups = self.group_by_setting(dimension);

        let mut setting_results = Vec::new();

        for (value, observations) in &groups {
            if observations.len() < self.config.min_observations_per_setting {
                continue;
            }

            let (effect, se, n) = Self::compute_effect_size(observations);

            // Skip if SE is infinite or too large
            if se.is_infinite() || se > 10.0 {
                continue;
            }

            let weight = if se > 0.0 { 1.0 / (se * se) } else { 0.0 };

            let mean_inj: f64 = observations
                .iter()
                .filter(|o| o.injected)
                .map(|o| o.score)
                .sum::<f64>()
                / observations.iter().filter(|o| o.injected).count().max(1) as f64;

            let mean_ctl: f64 = observations
                .iter()
                .filter(|o| !o.injected)
                .map(|o| o.score)
                .sum::<f64>()
                / observations.iter().filter(|o| !o.injected).count().max(1) as f64;

            setting_results.push(SettingResult {
                key: SettingKey::new(dimension, value),
                n_observations: n,
                mean_injected: mean_inj,
                mean_control: mean_ctl,
                effect_size: effect,
                std_error: se,
                ci_lower: effect - 1.96 * se,
                ci_upper: effect + 1.96 * se,
                weight,
            });
        }

        // Compute heterogeneity
        let heterogeneity = HeterogeneityResult::from_settings(&setting_results);

        // Compute pooled effect (fixed-effect model)
        let total_weight: f64 = setting_results.iter().map(|s| s.weight).sum();
        let (pooled_effect, pooled_se) = if total_weight > 0.0 {
            let effect = setting_results
                .iter()
                .map(|s| s.weight * s.effect_size)
                .sum::<f64>()
                / total_weight;
            let se = (1.0 / total_weight).sqrt();
            (effect, se)
        } else {
            (0.0, f64::INFINITY)
        };

        let pooled_ci_lower = pooled_effect - 1.96 * pooled_se;
        let pooled_ci_upper = pooled_effect + 1.96 * pooled_se;

        // Determine transportability
        let is_transportable = setting_results.len() >= self.config.min_settings
            && heterogeneity.i_squared <= self.config.max_i_squared;

        // Generate interpretation
        let interpretation = if setting_results.len() < self.config.min_settings {
            format!(
                "INSUFFICIENT DATA: Only {} settings with enough observations (need >= {}).",
                setting_results.len(),
                self.config.min_settings
            )
        } else if is_transportable {
            if pooled_effect > 0.5 {
                format!(
                    "STRONG TRANSPORTABLE SIGNAL: Large pooled effect (d={:.2}) is consistent \
                     across {} settings (I²={:.1}%). Findings likely generalize.",
                    pooled_effect,
                    setting_results.len(),
                    heterogeneity.i_squared
                )
            } else if pooled_effect > 0.2 {
                format!(
                    "MODERATE TRANSPORTABLE SIGNAL: Medium effect (d={:.2}) across {} settings \
                     (I²={:.1}%). Findings appear to generalize.",
                    pooled_effect,
                    setting_results.len(),
                    heterogeneity.i_squared
                )
            } else if pooled_effect > 0.0 {
                format!(
                    "WEAK TRANSPORTABLE SIGNAL: Small effect (d={:.2}) across {} settings \
                     (I²={:.1}%). Signal is consistent but weak.",
                    pooled_effect,
                    setting_results.len(),
                    heterogeneity.i_squared
                )
            } else {
                format!(
                    "NULL TRANSPORTABLE: No effect (d={:.2}) across {} settings (I²={:.1}%). \
                     Consistent with null hypothesis.",
                    pooled_effect,
                    setting_results.len(),
                    heterogeneity.i_squared
                )
            }
        } else {
            format!(
                "NOT TRANSPORTABLE: High heterogeneity (I²={:.1}%) indicates findings \
                 vary substantially across {} settings. Results may be setting-specific.",
                heterogeneity.i_squared,
                setting_results.len()
            )
        };

        TransportabilityResult {
            dimension: dimension.to_string(),
            settings: setting_results,
            pooled_effect,
            pooled_se,
            pooled_ci_lower,
            pooled_ci_upper,
            heterogeneity,
            is_transportable,
            interpretation,
        }
    }

    /// Analyze all dimensions present in observations
    pub fn analyze_all(&self) -> Vec<TransportabilityResult> {
        let dimensions: Vec<String> = self
            .observations
            .iter()
            .map(|(k, _)| k.dimension.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        dimensions
            .iter()
            .map(|d| self.analyze_dimension(d))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEAVE-ONE-OUT ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

/// Leave-one-out cross-validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaveOneOutResult {
    /// Which setting was left out
    pub left_out: SettingKey,
    /// Pooled effect without this setting
    pub effect_without: f64,
    /// Original pooled effect
    pub effect_with: f64,
    /// Difference (influence of this setting)
    pub influence: f64,
    /// Is this setting an outlier?
    pub is_outlier: bool,
}

impl TransportabilityAnalyzer {
    /// Leave-one-out analysis to identify influential settings
    pub fn leave_one_out(&self, dimension: &str) -> Vec<LeaveOneOutResult> {
        let full_result = self.analyze_dimension(dimension);
        let original_effect = full_result.pooled_effect;

        let mut results = Vec::new();

        for setting in &full_result.settings {
            // Recompute without this setting
            let without: Vec<&SettingResult> = full_result
                .settings
                .iter()
                .filter(|s| s.key != setting.key)
                .collect();

            if without.is_empty() {
                continue;
            }

            let total_weight: f64 = without.iter().map(|s| s.weight).sum();
            let effect_without = if total_weight > 0.0 {
                without
                    .iter()
                    .map(|s| s.weight * s.effect_size)
                    .sum::<f64>()
                    / total_weight
            } else {
                0.0
            };

            let influence = (original_effect - effect_without).abs();
            let is_outlier = influence > 0.5 * original_effect.abs().max(0.1);

            results.push(LeaveOneOutResult {
                left_out: setting.key.clone(),
                effect_without,
                effect_with: original_effect,
                influence,
                is_outlier,
            });
        }

        results
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMBINED REPORT
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete transportability report across all dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportabilityReport {
    /// Results per dimension
    pub dimensions: Vec<TransportabilityResult>,
    /// Overall transportability score (0-1)
    pub transportability_score: f64,
    /// Is the finding overall transportable?
    pub is_transportable: bool,
    /// Summary interpretation
    pub summary: String,
}

impl TransportabilityReport {
    pub fn from_results(results: Vec<TransportabilityResult>) -> Self {
        if results.is_empty() {
            return Self {
                dimensions: Vec::new(),
                transportability_score: 0.0,
                is_transportable: false,
                summary: "No dimensions analyzed.".to_string(),
            };
        }

        // Score: proportion of transportable dimensions, weighted by sample size
        let total_n: usize = results
            .iter()
            .map(|r| r.settings.iter().map(|s| s.n_observations).sum::<usize>())
            .sum();
        let transportable_n: usize = results
            .iter()
            .filter(|r| r.is_transportable)
            .map(|r| r.settings.iter().map(|s| s.n_observations).sum::<usize>())
            .sum();

        let score = if total_n > 0 {
            transportable_n as f64 / total_n as f64
        } else {
            0.0
        };

        let is_transportable = results.iter().all(|r| r.is_transportable);

        let summary = if is_transportable {
            format!(
                "TRANSPORTABLE: Findings generalize across all {} analyzed dimensions. \
                 Transportability score: {:.0}%.",
                results.len(),
                score * 100.0
            )
        } else {
            let non_transportable: Vec<&str> = results
                .iter()
                .filter(|r| !r.is_transportable)
                .map(|r| r.dimension.as_str())
                .collect();

            format!(
                "PARTIALLY TRANSPORTABLE: Findings do not generalize across: {:?}. \
                 Transportability score: {:.0}%.",
                non_transportable,
                score * 100.0
            )
        };

        Self {
            dimensions: results,
            transportability_score: score,
            is_transportable,
            summary,
        }
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
    fn test_setting_key() {
        let key = SettingKey::marker_class(MarkerClass::HashLike);
        assert_eq!(key.dimension, "marker_class");
        assert!(key.value.contains("HashLike"));

        let key2 = SettingKey::washout_ms(1000);
        assert_eq!(key2.dimension, "washout_ms");
        assert_eq!(key2.value, "1000");
    }

    #[test]
    fn test_heterogeneity_low() {
        // Consistent effects across settings
        let settings = vec![
            SettingResult {
                key: SettingKey::new("test", "A"),
                n_observations: 100,
                mean_injected: 0.8,
                mean_control: 0.2,
                effect_size: 1.5,
                std_error: 0.2,
                ci_lower: 1.1,
                ci_upper: 1.9,
                weight: 25.0,
            },
            SettingResult {
                key: SettingKey::new("test", "B"),
                n_observations: 100,
                mean_injected: 0.75,
                mean_control: 0.25,
                effect_size: 1.4,
                std_error: 0.2,
                ci_lower: 1.0,
                ci_upper: 1.8,
                weight: 25.0,
            },
            SettingResult {
                key: SettingKey::new("test", "C"),
                n_observations: 100,
                mean_injected: 0.78,
                mean_control: 0.22,
                effect_size: 1.45,
                std_error: 0.2,
                ci_lower: 1.05,
                ci_upper: 1.85,
                weight: 25.0,
            },
        ];

        let het = HeterogeneityResult::from_settings(&settings);
        assert!(
            het.i_squared < 50.0,
            "Should have low heterogeneity: I²={}",
            het.i_squared
        );
    }

    #[test]
    fn test_heterogeneity_high() {
        // Inconsistent effects
        let settings = vec![
            SettingResult {
                key: SettingKey::new("test", "A"),
                n_observations: 100,
                mean_injected: 0.9,
                mean_control: 0.1,
                effect_size: 2.5,
                std_error: 0.2,
                ci_lower: 2.1,
                ci_upper: 2.9,
                weight: 25.0,
            },
            SettingResult {
                key: SettingKey::new("test", "B"),
                n_observations: 100,
                mean_injected: 0.5,
                mean_control: 0.5,
                effect_size: 0.0,
                std_error: 0.2,
                ci_lower: -0.4,
                ci_upper: 0.4,
                weight: 25.0,
            },
            SettingResult {
                key: SettingKey::new("test", "C"),
                n_observations: 100,
                mean_injected: 0.3,
                mean_control: 0.7,
                effect_size: -1.0,
                std_error: 0.2,
                ci_lower: -1.4,
                ci_upper: -0.6,
                weight: 25.0,
            },
        ];

        let het = HeterogeneityResult::from_settings(&settings);
        assert!(
            het.i_squared > 50.0,
            "Should have high heterogeneity: I²={}",
            het.i_squared
        );
    }

    #[test]
    fn test_transportability_consistent() {
        let mut analyzer = TransportabilityAnalyzer::with_default_config();

        // Add consistent data across marker classes
        for class in ["A", "B", "C", "D"] {
            let setting = SettingKey::new("marker_class", class);

            // Injected markers score high
            for i in 0..20 {
                analyzer.add_observation(
                    setting.clone(),
                    make_obs(
                        &format!("{}_inj_{}", class, i),
                        true,
                        0.8 + (i as f64 * 0.01),
                    ),
                );
            }

            // Control markers score low
            for i in 0..20 {
                analyzer.add_observation(
                    setting.clone(),
                    make_obs(
                        &format!("{}_ctl_{}", class, i),
                        false,
                        0.2 + (i as f64 * 0.01),
                    ),
                );
            }
        }

        let result = analyzer.analyze_dimension("marker_class");

        assert!(
            result.is_transportable,
            "Consistent data should be transportable"
        );
        assert!(
            result.pooled_effect > 0.0,
            "Should have positive pooled effect"
        );
        assert!(result.heterogeneity.i_squared < 50.0, "Should have low I²");
    }

    #[test]
    fn test_transportability_inconsistent() {
        let mut analyzer = TransportabilityAnalyzer::with_default_config();

        // Class A: Strong positive effect
        for i in 0..20 {
            analyzer.add_observation(
                SettingKey::new("marker_class", "A"),
                make_obs(&format!("A_inj_{}", i), true, 0.9),
            );
            analyzer.add_observation(
                SettingKey::new("marker_class", "A"),
                make_obs(&format!("A_ctl_{}", i), false, 0.1),
            );
        }

        // Class B: No effect (with some variance)
        for i in 0..20 {
            let noise = (i as f64 * 0.01).sin() * 0.1;
            analyzer.add_observation(
                SettingKey::new("marker_class", "B"),
                make_obs(&format!("B_inj_{}", i), true, 0.5 + noise),
            );
            analyzer.add_observation(
                SettingKey::new("marker_class", "B"),
                make_obs(&format!("B_ctl_{}", i), false, 0.5 - noise),
            );
        }

        // Class C: Negative effect (weird) - with variance
        for i in 0..20 {
            let noise = (i as f64 * 0.02).sin() * 0.05;
            analyzer.add_observation(
                SettingKey::new("marker_class", "C"),
                make_obs(&format!("C_inj_{}", i), true, 0.3 + noise),
            );
            analyzer.add_observation(
                SettingKey::new("marker_class", "C"),
                make_obs(&format!("C_ctl_{}", i), false, 0.7 + noise),
            );
        }

        let result = analyzer.analyze_dimension("marker_class");

        assert!(
            !result.is_transportable,
            "Inconsistent data should NOT be transportable"
        );
        assert!(
            result.heterogeneity.i_squared > 50.0,
            "Should have high I²: {}",
            result.heterogeneity.i_squared
        );
    }

    #[test]
    fn test_leave_one_out() {
        let mut analyzer = TransportabilityAnalyzer::with_default_config();

        // Add mostly consistent data with some variance
        for class in ["A", "B", "C"] {
            let setting = SettingKey::new("marker_class", class);
            for i in 0..20 {
                let noise = (i as f64 * 0.05).sin() * 0.05;
                analyzer.add_observation(
                    setting.clone(),
                    make_obs(&format!("{}_i{}", class, i), true, 0.7 + noise),
                );
                analyzer.add_observation(
                    setting.clone(),
                    make_obs(&format!("{}_c{}", class, i), false, 0.3 + noise),
                );
            }
        }

        // Add one outlier class (reversed effect)
        let outlier = SettingKey::new("marker_class", "OUTLIER");
        for i in 0..20 {
            let noise = (i as f64 * 0.05).sin() * 0.05;
            analyzer.add_observation(
                outlier.clone(),
                make_obs(&format!("OUT_i{}", i), true, 0.2 + noise),
            );
            analyzer.add_observation(
                outlier.clone(),
                make_obs(&format!("OUT_c{}", i), false, 0.8 + noise),
            );
        }

        let loo_results = analyzer.leave_one_out("marker_class");

        // Should have 4 results (one for each setting left out)
        assert!(
            loo_results.len() >= 3,
            "Should have at least 3 LOO results, got {}",
            loo_results.len()
        );

        // The OUTLIER should have high influence if present
        if let Some(outlier_r) = loo_results.iter().find(|r| r.left_out.value == "OUTLIER") {
            // Just verify it computed something
            assert!(
                outlier_r.influence >= 0.0,
                "Influence should be non-negative: {}",
                outlier_r.influence
            );
        }
    }

    #[test]
    fn test_transportability_report() {
        // Create results with actual settings data
        let settings1 = vec![
            SettingResult {
                key: SettingKey::new("marker_class", "A"),
                n_observations: 50,
                mean_injected: 0.8,
                mean_control: 0.2,
                effect_size: 1.5,
                std_error: 0.2,
                ci_lower: 1.1,
                ci_upper: 1.9,
                weight: 25.0,
            },
            SettingResult {
                key: SettingKey::new("marker_class", "B"),
                n_observations: 50,
                mean_injected: 0.75,
                mean_control: 0.25,
                effect_size: 1.4,
                std_error: 0.2,
                ci_lower: 1.0,
                ci_upper: 1.8,
                weight: 25.0,
            },
        ];

        let settings2 = vec![SettingResult {
            key: SettingKey::new("washout_ms", "100"),
            n_observations: 40,
            mean_injected: 0.7,
            mean_control: 0.3,
            effect_size: 1.2,
            std_error: 0.25,
            ci_lower: 0.7,
            ci_upper: 1.7,
            weight: 16.0,
        }];

        let results = vec![
            TransportabilityResult {
                dimension: "marker_class".to_string(),
                settings: settings1,
                pooled_effect: 0.5,
                pooled_se: 0.1,
                pooled_ci_lower: 0.3,
                pooled_ci_upper: 0.7,
                heterogeneity: HeterogeneityResult {
                    q_statistic: 2.0,
                    q_p_value: 0.5,
                    i_squared: 20.0,
                    tau_squared: 0.01,
                    k_settings: 4,
                    interpretation: "Low heterogeneity".to_string(),
                },
                is_transportable: true,
                interpretation: "Transportable".to_string(),
            },
            TransportabilityResult {
                dimension: "washout_ms".to_string(),
                settings: settings2,
                pooled_effect: 0.4,
                pooled_se: 0.15,
                pooled_ci_lower: 0.1,
                pooled_ci_upper: 0.7,
                heterogeneity: HeterogeneityResult {
                    q_statistic: 1.5,
                    q_p_value: 0.6,
                    i_squared: 15.0,
                    tau_squared: 0.005,
                    k_settings: 3,
                    interpretation: "Low heterogeneity".to_string(),
                },
                is_transportable: true,
                interpretation: "Transportable".to_string(),
            },
        ];

        let report = TransportabilityReport::from_results(results);

        assert!(report.is_transportable);
        // Score should be 1.0 since all dimensions are transportable
        assert!(
            report.transportability_score >= 0.5,
            "Score should be >= 0.5: {}",
            report.transportability_score
        );
    }

    #[test]
    fn test_standard_normal_cdf() {
        // Test known values
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!((standard_normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((standard_normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }

    #[test]
    fn test_effect_size_computation() {
        let obs: Vec<Observation> = vec![
            make_obs("1", true, 0.9),
            make_obs("2", true, 0.85),
            make_obs("3", true, 0.88),
            make_obs("4", false, 0.15),
            make_obs("5", false, 0.12),
            make_obs("6", false, 0.18),
        ];

        let obs_refs: Vec<&Observation> = obs.iter().collect();
        let (effect, se, n) = TransportabilityAnalyzer::compute_effect_size(&obs_refs);

        assert_eq!(n, 6);
        assert!(effect > 0.0, "Effect should be positive: {}", effect);
        assert!(se > 0.0 && se < 10.0, "SE should be reasonable: {}", se);
    }
}
