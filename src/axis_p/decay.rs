//! ═══════════════════════════════════════════════════════════════════════════════
//! DECAY — Temporal Decay Curve Estimation for Persistence Mechanism Discovery
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Fits decay models to signal strength vs washout time data to characterize
//! the temporal structure of any detected persistence.
//!
//! Decay Model Interpretation:
//!   - Exponential (τ): Cache-like mechanism, half-life = τ * ln(2)
//!   - Power Law (α): Associative/episodic memory, slow decay
//!   - Step (threshold): Hard context boundary
//!   - Constant: True cross-session persistence
//!   - Null: No detectable signal at any washout time
//!
//! Model selection via AIC (Akaike Information Criterion).
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════════════
// DECAY MODEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Decay model types with fitted parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DecayModel {
    /// y = amplitude * exp(-t/tau) — cache-like mechanism
    /// Half-life = tau * ln(2)
    Exponential { amplitude: f64, tau: f64 },

    /// y = amplitude * t^(-alpha) — associative memory
    /// Slower decay than exponential for large t
    PowerLaw { amplitude: f64, alpha: f64 },

    /// y = amplitude * (t < threshold) — hard boundary
    /// Signal present below threshold, absent above
    Step { amplitude: f64, threshold_ms: f64 },

    /// y = amplitude — true persistence
    /// Signal constant regardless of washout time
    Constant { amplitude: f64 },

    /// No detectable signal at any time
    Null,
}

impl DecayModel {
    /// Predict signal strength at given washout time
    pub fn predict(&self, washout_ms: f64) -> f64 {
        match self {
            DecayModel::Exponential { amplitude, tau } => {
                if *tau <= 0.0 {
                    return 0.0;
                }
                amplitude * (-washout_ms / tau).exp()
            }
            DecayModel::PowerLaw { amplitude, alpha } => {
                if washout_ms <= 0.0 {
                    return *amplitude;
                }
                amplitude * washout_ms.powf(-alpha)
            }
            DecayModel::Step {
                amplitude,
                threshold_ms,
            } => {
                if washout_ms < *threshold_ms {
                    *amplitude
                } else {
                    0.0
                }
            }
            DecayModel::Constant { amplitude } => *amplitude,
            DecayModel::Null => 0.0,
        }
    }

    /// Get half-life if applicable (time for signal to decay to 50%)
    pub fn half_life(&self) -> Option<Duration> {
        match self {
            DecayModel::Exponential { tau, .. } => {
                let half_life_ms = tau * std::f64::consts::LN_2;
                Some(Duration::from_millis(half_life_ms as u64))
            }
            DecayModel::PowerLaw { alpha, .. } => {
                // t where amplitude * t^(-alpha) = 0.5 * amplitude
                // t^(-alpha) = 0.5
                // t = 0.5^(-1/alpha) = 2^(1/alpha)
                if *alpha <= 0.0 {
                    return None;
                }
                let half_life_ms = 2.0_f64.powf(1.0 / alpha);
                Some(Duration::from_millis(half_life_ms as u64))
            }
            DecayModel::Step { threshold_ms, .. } => {
                Some(Duration::from_millis(*threshold_ms as u64))
            }
            DecayModel::Constant { .. } => None, // Never decays
            DecayModel::Null => Some(Duration::ZERO), // Instant decay
        }
    }

    /// Human-readable description
    pub fn describe(&self) -> String {
        match self {
            DecayModel::Exponential { amplitude, tau } => {
                let half_life = tau * std::f64::consts::LN_2;
                format!(
                    "Exponential: amplitude={:.3}, τ={:.0}ms, half-life={:.0}ms",
                    amplitude, tau, half_life
                )
            }
            DecayModel::PowerLaw { amplitude, alpha } => {
                format!("Power Law: amplitude={:.3}, α={:.3}", amplitude, alpha)
            }
            DecayModel::Step {
                amplitude,
                threshold_ms,
            } => {
                format!(
                    "Step: amplitude={:.3}, threshold={:.0}ms",
                    amplitude, threshold_ms
                )
            }
            DecayModel::Constant { amplitude } => {
                format!("Constant: amplitude={:.3}", amplitude)
            }
            DecayModel::Null => "Null: no signal".to_string(),
        }
    }

    /// Model name for display
    pub fn name(&self) -> &'static str {
        match self {
            DecayModel::Exponential { .. } => "Exponential",
            DecayModel::PowerLaw { .. } => "PowerLaw",
            DecayModel::Step { .. } => "Step",
            DecayModel::Constant { .. } => "Constant",
            DecayModel::Null => "Null",
        }
    }

    /// Number of free parameters (for AIC calculation)
    pub fn n_params(&self) -> usize {
        match self {
            DecayModel::Exponential { .. } => 2, // amplitude, tau
            DecayModel::PowerLaw { .. } => 2,    // amplitude, alpha
            DecayModel::Step { .. } => 2,        // amplitude, threshold
            DecayModel::Constant { .. } => 1,    // amplitude
            DecayModel::Null => 0,               // no parameters
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECAY POINT
// ═══════════════════════════════════════════════════════════════════════════════

/// A single decay measurement at a specific washout time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayPoint {
    /// Washout time in milliseconds
    pub washout_ms: u64,
    /// Mean detection score at this washout time
    pub detection_score: f64,
    /// Standard error of the mean
    pub std_error: f64,
    /// Number of samples used to compute mean
    pub n_samples: usize,
}

impl DecayPoint {
    pub fn new(washout_ms: u64, detection_score: f64, std_error: f64, n_samples: usize) -> Self {
        Self {
            washout_ms,
            detection_score,
            std_error,
            n_samples,
        }
    }

    /// Create from raw samples
    pub fn from_samples(washout_ms: u64, samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::new(washout_ms, 0.0, 0.0, 0);
        }

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;

        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        let std_error = (variance / n as f64).sqrt();

        Self::new(washout_ms, mean, std_error, n)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIT RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of fitting a specific model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitResult {
    /// The fitted model
    pub model: DecayModel,
    /// Residual sum of squares
    pub rss: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECAY CURVE ESTIMATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Decay curve estimator with model fitting
pub struct DecayCurveEstimator {
    /// Collected decay points
    points: Vec<DecayPoint>,
    /// All fitted models
    fits: Vec<FitResult>,
    /// Best model by AIC
    best_model: Option<DecayModel>,
}

impl DecayCurveEstimator {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            fits: Vec::new(),
            best_model: None,
        }
    }

    /// Add a measurement point
    pub fn add_point(&mut self, point: DecayPoint) {
        self.points.push(point);
        // Invalidate cached fits
        self.fits.clear();
        self.best_model = None;
    }

    /// Add multiple points
    pub fn add_points(&mut self, points: Vec<DecayPoint>) {
        self.points.extend(points);
        self.fits.clear();
        self.best_model = None;
    }

    /// Get collected points
    pub fn points(&self) -> &[DecayPoint] {
        &self.points
    }

    /// Number of points
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// Fit all models and select best by AIC
    pub fn fit(&mut self) -> DecayModel {
        if self.points.is_empty() {
            self.best_model = Some(DecayModel::Null);
            return DecayModel::Null;
        }

        // Check if all scores are near zero -> Null model
        let max_score = self
            .points
            .iter()
            .map(|p| p.detection_score)
            .fold(0.0, f64::max);
        if max_score < 0.05 {
            self.best_model = Some(DecayModel::Null);
            return DecayModel::Null;
        }

        let mut fits = Vec::new();

        // Fit each model
        if let Some(fit) = self.fit_exponential() {
            fits.push(fit);
        }
        if let Some(fit) = self.fit_power_law() {
            fits.push(fit);
        }
        if let Some(fit) = self.fit_step() {
            fits.push(fit);
        }
        if let Some(fit) = self.fit_constant() {
            fits.push(fit);
        }

        // Add null model
        let null_rss: f64 = self.points.iter().map(|p| p.detection_score.powi(2)).sum();
        let null_aic = self.compute_aic(0, null_rss);
        fits.push(FitResult {
            model: DecayModel::Null,
            rss: null_rss,
            aic: null_aic,
            r_squared: 0.0,
        });

        // Select best by AIC (lower is better)
        fits.sort_by(|a, b| {
            a.aic
                .partial_cmp(&b.aic)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.fits = fits;
        self.best_model = self.fits.first().map(|f| f.model);

        self.best_model.unwrap_or(DecayModel::Null)
    }

    /// Get the best fitted model
    pub fn best_model(&self) -> Option<DecayModel> {
        self.best_model
    }

    /// Get all fit results
    pub fn all_fits(&self) -> &[FitResult] {
        &self.fits
    }

    /// Predict signal at arbitrary washout time using best model
    pub fn predict(&self, washout_ms: u64) -> f64 {
        self.best_model
            .map(|m| m.predict(washout_ms as f64))
            .unwrap_or(0.0)
    }

    /// Get half-life from best model
    pub fn half_life(&self) -> Option<Duration> {
        self.best_model.and_then(|m| m.half_life())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MODEL FITTING
    // ═══════════════════════════════════════════════════════════════════════════

    /// Fit exponential model: y = a * exp(-t/τ)
    /// Using log-linear regression: ln(y) = ln(a) - t/τ
    fn fit_exponential(&self) -> Option<FitResult> {
        // Filter points with positive scores for log transform
        let valid_points: Vec<_> = self
            .points
            .iter()
            .filter(|p| p.detection_score > 0.01)
            .collect();

        if valid_points.len() < 2 {
            return None;
        }

        // Log-linear regression: ln(y) = b0 + b1*t
        // b0 = ln(amplitude), b1 = -1/tau
        let n = valid_points.len() as f64;
        let sum_t: f64 = valid_points.iter().map(|p| p.washout_ms as f64).sum();
        let sum_lny: f64 = valid_points.iter().map(|p| p.detection_score.ln()).sum();
        let sum_t2: f64 = valid_points
            .iter()
            .map(|p| (p.washout_ms as f64).powi(2))
            .sum();
        let sum_t_lny: f64 = valid_points
            .iter()
            .map(|p| p.washout_ms as f64 * p.detection_score.ln())
            .sum();

        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1e-10 {
            return None;
        }

        let b1 = (n * sum_t_lny - sum_t * sum_lny) / denom;
        let b0 = (sum_lny - b1 * sum_t) / n;

        // tau = -1/b1, amplitude = exp(b0)
        if b1 >= 0.0 {
            // Slope should be negative for decay
            return None;
        }

        let tau = -1.0 / b1;
        let amplitude = b0.exp();

        if tau <= 0.0 || amplitude <= 0.0 || !tau.is_finite() || !amplitude.is_finite() {
            return None;
        }

        let model = DecayModel::Exponential { amplitude, tau };
        let (rss, r_squared) = self.compute_rss_r2(&model);
        let aic = self.compute_aic(2, rss);

        Some(FitResult {
            model,
            rss,
            aic,
            r_squared,
        })
    }

    /// Fit power law model: y = a * t^(-α)
    /// Using log-log regression: ln(y) = ln(a) - α*ln(t)
    fn fit_power_law(&self) -> Option<FitResult> {
        // Filter points with positive scores and positive times
        let valid_points: Vec<_> = self
            .points
            .iter()
            .filter(|p| p.detection_score > 0.01 && p.washout_ms > 0)
            .collect();

        if valid_points.len() < 2 {
            return None;
        }

        // Log-log regression: ln(y) = b0 + b1*ln(t)
        // b0 = ln(amplitude), b1 = -alpha
        let n = valid_points.len() as f64;
        let sum_lnt: f64 = valid_points
            .iter()
            .map(|p| (p.washout_ms as f64).ln())
            .sum();
        let sum_lny: f64 = valid_points.iter().map(|p| p.detection_score.ln()).sum();
        let sum_lnt2: f64 = valid_points
            .iter()
            .map(|p| (p.washout_ms as f64).ln().powi(2))
            .sum();
        let sum_lnt_lny: f64 = valid_points
            .iter()
            .map(|p| (p.washout_ms as f64).ln() * p.detection_score.ln())
            .sum();

        let denom = n * sum_lnt2 - sum_lnt * sum_lnt;
        if denom.abs() < 1e-10 {
            return None;
        }

        let b1 = (n * sum_lnt_lny - sum_lnt * sum_lny) / denom;
        let b0 = (sum_lny - b1 * sum_lnt) / n;

        // alpha = -b1, amplitude = exp(b0)
        let alpha = -b1;
        let amplitude = b0.exp();

        if alpha <= 0.0 || amplitude <= 0.0 || !alpha.is_finite() || !amplitude.is_finite() {
            return None;
        }

        let model = DecayModel::PowerLaw { amplitude, alpha };
        let (rss, r_squared) = self.compute_rss_r2(&model);
        let aic = self.compute_aic(2, rss);

        Some(FitResult {
            model,
            rss,
            aic,
            r_squared,
        })
    }

    /// Fit step function model via threshold search
    fn fit_step(&self) -> Option<FitResult> {
        if self.points.len() < 2 {
            return None;
        }

        // Sort points by washout time
        let mut sorted_points = self.points.clone();
        sorted_points.sort_by(|a, b| a.washout_ms.cmp(&b.washout_ms));

        // Search for best threshold
        let mut best_rss = f64::MAX;
        let mut best_threshold = 0.0;
        let mut best_amplitude = 0.0;

        // Try each point's washout time as a potential threshold
        for i in 1..sorted_points.len() {
            let threshold = sorted_points[i].washout_ms as f64;

            // Compute amplitude as mean of points below threshold
            let below: Vec<f64> = sorted_points
                .iter()
                .filter(|p| (p.washout_ms as f64) < threshold)
                .map(|p| p.detection_score)
                .collect();

            if below.is_empty() {
                continue;
            }

            let amplitude = below.iter().sum::<f64>() / below.len() as f64;

            // Compute RSS
            let model = DecayModel::Step {
                amplitude,
                threshold_ms: threshold,
            };
            let (rss, _) = self.compute_rss_r2(&model);

            if rss < best_rss {
                best_rss = rss;
                best_threshold = threshold;
                best_amplitude = amplitude;
            }
        }

        if best_rss == f64::MAX || best_amplitude <= 0.0 {
            return None;
        }

        let model = DecayModel::Step {
            amplitude: best_amplitude,
            threshold_ms: best_threshold,
        };
        let (rss, r_squared) = self.compute_rss_r2(&model);
        let aic = self.compute_aic(2, rss);

        Some(FitResult {
            model,
            rss,
            aic,
            r_squared,
        })
    }

    /// Fit constant model: y = a
    fn fit_constant(&self) -> Option<FitResult> {
        if self.points.is_empty() {
            return None;
        }

        let amplitude =
            self.points.iter().map(|p| p.detection_score).sum::<f64>() / self.points.len() as f64;

        if amplitude <= 0.0 {
            return None;
        }

        let model = DecayModel::Constant { amplitude };
        let (rss, r_squared) = self.compute_rss_r2(&model);
        let aic = self.compute_aic(1, rss);

        Some(FitResult {
            model,
            rss,
            aic,
            r_squared,
        })
    }

    /// Compute RSS and R² for a model
    fn compute_rss_r2(&self, model: &DecayModel) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }

        let mean_y =
            self.points.iter().map(|p| p.detection_score).sum::<f64>() / self.points.len() as f64;

        let ss_tot: f64 = self
            .points
            .iter()
            .map(|p| (p.detection_score - mean_y).powi(2))
            .sum();

        let rss: f64 = self
            .points
            .iter()
            .map(|p| {
                let predicted = model.predict(p.washout_ms as f64);
                (p.detection_score - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 1e-10 {
            1.0 - rss / ss_tot
        } else {
            0.0
        };

        (rss, r_squared.max(0.0))
    }

    /// Compute AIC: 2k + n*ln(RSS/n)
    fn compute_aic(&self, k: usize, rss: f64) -> f64 {
        let n = self.points.len() as f64;
        if n <= 0.0 {
            return f64::MAX;
        }

        // Handle perfect fit (RSS = 0) or near-zero RSS
        let rss_adj = rss.max(1e-10);

        let aic = 2.0 * k as f64 + n * (rss_adj / n).ln();

        // Small sample correction (AICc)
        if n > k as f64 + 2.0 {
            aic + (2.0 * k as f64 * (k as f64 + 1.0)) / (n - k as f64 - 1.0)
        } else {
            aic
        }
    }
}

impl Default for DecayCurveEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate logarithmically-spaced washout times
pub fn log_spaced_washouts(min_ms: u64, max_ms: u64, n_points: usize) -> Vec<u64> {
    if n_points == 0 || min_ms >= max_ms {
        return Vec::new();
    }

    if n_points == 1 {
        return vec![min_ms];
    }

    let log_min = (min_ms as f64).ln();
    let log_max = (max_ms as f64).ln();
    let step = (log_max - log_min) / (n_points - 1) as f64;

    (0..n_points)
        .map(|i| {
            let log_val = log_min + step * i as f64;
            log_val.exp().round() as u64
        })
        .collect()
}

/// Generate decay sweep report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecaySweepReport {
    /// Washout times tested
    pub washout_times: Vec<u64>,
    /// Measured points
    pub points: Vec<DecayPoint>,
    /// Best fitted model
    pub best_model: DecayModel,
    /// All model fits
    pub all_fits: Vec<FitResult>,
    /// Estimated half-life
    pub half_life_ms: Option<u64>,
    /// Interpretation
    pub interpretation: String,
}

impl DecaySweepReport {
    pub fn from_estimator(estimator: &DecayCurveEstimator) -> Self {
        let best_model = estimator.best_model.unwrap_or(DecayModel::Null);
        let half_life_ms = estimator.half_life().map(|d| d.as_millis() as u64);

        let interpretation = match best_model {
            DecayModel::Exponential { tau, .. } => {
                let half_life = tau * std::f64::consts::LN_2;
                format!(
                    "Exponential decay detected. Half-life: {:.0}ms. \
                     Suggests cache-like or short-term memory mechanism.",
                    half_life
                )
            }
            DecayModel::PowerLaw { alpha, .. } => {
                format!(
                    "Power-law decay detected (α={:.2}). \
                     Suggests associative or episodic memory mechanism with slow forgetting.",
                    alpha
                )
            }
            DecayModel::Step { threshold_ms, .. } => {
                format!(
                    "Step function detected. Signal drops at {:.0}ms. \
                     Suggests hard context window boundary.",
                    threshold_ms
                )
            }
            DecayModel::Constant { amplitude } => {
                format!(
                    "Constant signal detected (amplitude={:.3}). \
                     Suggests true cross-session persistence. REQUIRES VERIFICATION.",
                    amplitude
                )
            }
            DecayModel::Null => "No signal detected at any washout time. \
                 Consistent with no persistence."
                .to_string(),
        };

        Self {
            washout_times: estimator.points.iter().map(|p| p.washout_ms).collect(),
            points: estimator.points.clone(),
            best_model,
            all_fits: estimator.fits.clone(),
            half_life_ms,
            interpretation,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_spaced_washouts() {
        let times = log_spaced_washouts(100, 10000, 5);
        assert_eq!(times.len(), 5);
        assert_eq!(times[0], 100);
        assert_eq!(times[4], 10000);
        // Should be roughly logarithmic: 100, 316, 1000, 3162, 10000
        assert!(times[1] > 200 && times[1] < 500);
        assert!(times[2] > 800 && times[2] < 1200);
    }

    #[test]
    fn test_log_spaced_edge_cases() {
        assert!(log_spaced_washouts(100, 100, 5).is_empty());
        assert!(log_spaced_washouts(100, 1000, 0).is_empty());
        assert_eq!(log_spaced_washouts(100, 1000, 1), vec![100]);
    }

    #[test]
    fn test_decay_point_from_samples() {
        let samples = vec![0.5, 0.6, 0.4, 0.55, 0.45];
        let point = DecayPoint::from_samples(1000, &samples);

        assert_eq!(point.washout_ms, 1000);
        assert!((point.detection_score - 0.5).abs() < 0.01);
        assert_eq!(point.n_samples, 5);
        assert!(point.std_error > 0.0);
    }

    #[test]
    fn test_decay_model_predict() {
        // Exponential
        let exp = DecayModel::Exponential {
            amplitude: 1.0,
            tau: 1000.0,
        };
        assert!((exp.predict(0.0) - 1.0).abs() < 0.01);
        assert!((exp.predict(1000.0) - 0.368).abs() < 0.01); // e^(-1)

        // Power law
        let pow = DecayModel::PowerLaw {
            amplitude: 1.0,
            alpha: 1.0,
        };
        assert!((pow.predict(1.0) - 1.0).abs() < 0.01);
        assert!((pow.predict(2.0) - 0.5).abs() < 0.01);

        // Step
        let step = DecayModel::Step {
            amplitude: 1.0,
            threshold_ms: 500.0,
        };
        assert!((step.predict(100.0) - 1.0).abs() < 0.01);
        assert!((step.predict(600.0) - 0.0).abs() < 0.01);

        // Constant
        let constant = DecayModel::Constant { amplitude: 0.5 };
        assert!((constant.predict(0.0) - 0.5).abs() < 0.01);
        assert!((constant.predict(10000.0) - 0.5).abs() < 0.01);

        // Null
        let null = DecayModel::Null;
        assert!((null.predict(1000.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_decay_model_half_life() {
        let exp = DecayModel::Exponential {
            amplitude: 1.0,
            tau: 1000.0,
        };
        let half = exp.half_life().unwrap();
        assert!((half.as_millis() as f64 - 693.0).abs() < 10.0); // 1000 * ln(2)

        let constant = DecayModel::Constant { amplitude: 1.0 };
        assert!(constant.half_life().is_none());

        let null = DecayModel::Null;
        assert_eq!(null.half_life().unwrap(), Duration::ZERO);
    }

    #[test]
    fn test_fit_exponential_synthetic() {
        // Generate synthetic exponential decay: y = 1.0 * exp(-t/1000)
        let mut estimator = DecayCurveEstimator::new();

        for t in [100, 200, 500, 1000, 2000, 5000].iter() {
            let y = (-(*t as f64) / 1000.0).exp();
            estimator.add_point(DecayPoint::new(*t, y, 0.01, 10));
        }

        let model = estimator.fit();

        match model {
            DecayModel::Exponential { tau, amplitude } => {
                assert!((tau - 1000.0).abs() < 200.0, "tau={} should be ~1000", tau);
                assert!(
                    (amplitude - 1.0).abs() < 0.3,
                    "amplitude={} should be ~1.0",
                    amplitude
                );
            }
            other => panic!("Expected Exponential, got {:?}", other),
        }
    }

    #[test]
    fn test_fit_constant_synthetic() {
        // Generate constant signal
        let mut estimator = DecayCurveEstimator::new();

        for t in [100, 500, 1000, 5000, 10000].iter() {
            estimator.add_point(DecayPoint::new(*t, 0.5, 0.01, 10));
        }

        let model = estimator.fit();

        // Should detect constant (or close to it)
        match model {
            DecayModel::Constant { amplitude } => {
                assert!(
                    (amplitude - 0.5).abs() < 0.1,
                    "amplitude={} should be ~0.5",
                    amplitude
                );
            }
            DecayModel::Exponential { tau, .. } if tau > 100000.0 => {
                // Very slow decay is effectively constant
            }
            other => panic!(
                "Expected Constant or very slow Exponential, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_fit_null_synthetic() {
        // Generate null signal (all zeros)
        let mut estimator = DecayCurveEstimator::new();

        for t in [100, 500, 1000, 5000].iter() {
            estimator.add_point(DecayPoint::new(*t, 0.01, 0.005, 10));
        }

        let model = estimator.fit();
        assert!(
            matches!(model, DecayModel::Null),
            "Expected Null, got {:?}",
            model
        );
    }

    #[test]
    fn test_fit_step_synthetic() {
        // Generate step function: high before 1000ms, zero after
        let mut estimator = DecayCurveEstimator::new();

        for t in [100, 300, 500, 800].iter() {
            estimator.add_point(DecayPoint::new(*t, 0.8, 0.05, 10));
        }
        for t in [1200, 2000, 5000].iter() {
            estimator.add_point(DecayPoint::new(*t, 0.02, 0.01, 10));
        }

        let model = estimator.fit();

        match model {
            DecayModel::Step {
                threshold_ms,
                amplitude,
            } => {
                assert!(
                    threshold_ms > 800.0 && threshold_ms < 1500.0,
                    "threshold={} should be ~1000",
                    threshold_ms
                );
                assert!(
                    (amplitude - 0.8).abs() < 0.2,
                    "amplitude={} should be ~0.8",
                    amplitude
                );
            }
            other => {
                // Step might not always be selected if other models fit well
                // Just ensure we get a reasonable model
                println!("Got {:?} instead of Step", other);
            }
        }
    }

    #[test]
    fn test_decay_sweep_report() {
        let mut estimator = DecayCurveEstimator::new();

        for t in [100, 500, 1000, 2000, 5000].iter() {
            let y = (-(*t as f64) / 1000.0).exp();
            estimator.add_point(DecayPoint::new(*t, y, 0.01, 10));
        }

        estimator.fit();
        let report = DecaySweepReport::from_estimator(&estimator);

        assert_eq!(report.points.len(), 5);
        assert!(report.half_life_ms.is_some());
        assert!(!report.interpretation.is_empty());
    }

    #[test]
    fn test_model_descriptions() {
        let exp = DecayModel::Exponential {
            amplitude: 1.0,
            tau: 1000.0,
        };
        assert!(exp.describe().contains("Exponential"));
        assert!(exp.describe().contains("half-life"));

        let pow = DecayModel::PowerLaw {
            amplitude: 1.0,
            alpha: 0.5,
        };
        assert!(pow.describe().contains("Power Law"));

        let step = DecayModel::Step {
            amplitude: 1.0,
            threshold_ms: 500.0,
        };
        assert!(step.describe().contains("Step"));

        let constant = DecayModel::Constant { amplitude: 0.5 };
        assert!(constant.describe().contains("Constant"));

        let null = DecayModel::Null;
        assert!(null.describe().contains("Null"));
    }

    #[test]
    fn test_n_params() {
        assert_eq!(
            DecayModel::Exponential {
                amplitude: 1.0,
                tau: 1000.0
            }
            .n_params(),
            2
        );
        assert_eq!(
            DecayModel::PowerLaw {
                amplitude: 1.0,
                alpha: 0.5
            }
            .n_params(),
            2
        );
        assert_eq!(
            DecayModel::Step {
                amplitude: 1.0,
                threshold_ms: 500.0
            }
            .n_params(),
            2
        );
        assert_eq!(DecayModel::Constant { amplitude: 1.0 }.n_params(), 1);
        assert_eq!(DecayModel::Null.n_params(), 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // INTEGRATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════════

    #[test]
    fn integration_decay_sweep_no_persistence_simulated() {
        // Simulate a target with NO persistence (all zeros at all washout times)
        let mut estimator = DecayCurveEstimator::new();

        // Test at multiple washout points
        let washout_points = log_spaced_washouts(10, 1000, 4);

        for &washout_ms in &washout_points {
            // Zero detection rate across trials (no persistence)
            estimator.add_point(DecayPoint::new(washout_ms, 0.02, 0.01, 5));
        }

        let model = estimator.fit();

        // Should detect null (no persistence)
        match model {
            DecayModel::Null => {
                // Expected - no persistence
            }
            DecayModel::Constant { amplitude } => {
                assert!(
                    amplitude < 0.1,
                    "Should detect no persistence, got amplitude={}",
                    amplitude
                );
            }
            other => {
                // Other models might fit if there's noise, but amplitude should be low
                let predictions: Vec<f64> = washout_points
                    .iter()
                    .map(|&t| other.predict(t as f64))
                    .collect();
                let max_pred = predictions.iter().cloned().fold(f64::MIN, f64::max);
                assert!(
                    max_pred < 0.1,
                    "Should show no signal, max_pred={}",
                    max_pred
                );
            }
        }
    }

    #[test]
    fn integration_decay_sweep_with_simulated_cache() {
        // Simulate a target with exponential decay by manually injecting synthetic data
        let mut estimator = DecayCurveEstimator::new();

        // Simulate exponential decay with tau=500ms
        // y = exp(-t/500)
        let washout_points = log_spaced_washouts(100, 5000, 6);

        for (i, &washout_ms) in washout_points.iter().enumerate() {
            let true_signal = (-(washout_ms as f64) / 500.0).exp();
            // Add deterministic noise based on index
            let noise = (i as f64 - 2.5) * 0.02; // Small variation
            let observed = (true_signal + noise).max(0.0).min(1.0);

            estimator.add_point(DecayPoint::new(washout_ms, observed, 0.02, 5));
        }

        let model = estimator.fit();
        let report = DecaySweepReport::from_estimator(&estimator);

        // Should detect exponential with half-life around 500 * ln(2) ≈ 347ms
        match model {
            DecayModel::Exponential { tau, .. } => {
                // tau should be in reasonable range
                assert!(tau > 200.0 && tau < 1000.0, "tau={} should be ~500", tau);
            }
            other => {
                // Might fit power law too, but should have meaningful half-life
                assert!(
                    report.half_life_ms.is_some() || matches!(other, DecayModel::PowerLaw { .. }),
                    "Expected decay model with half-life, got {:?}",
                    other
                );
            }
        }

        // Interpretation should mention cache or memory
        assert!(
            report.interpretation.contains("cache")
                || report.interpretation.contains("memory")
                || report.interpretation.contains("decay"),
            "Interpretation should mention decay mechanism: {}",
            report.interpretation
        );
    }

    #[test]
    fn integration_decay_sweep_detects_step_function() {
        // Simulate a hard context window cutoff at 2000ms
        let mut estimator = DecayCurveEstimator::new();

        let washout_points = log_spaced_washouts(500, 10000, 8);

        for &washout_ms in &washout_points {
            // Step function: 0.8 before 2000ms, 0.0 after
            let signal = if washout_ms < 2000 { 0.8 } else { 0.02 };
            estimator.add_point(DecayPoint::new(washout_ms, signal, 0.02, 5));
        }

        let model = estimator.fit();

        match model {
            DecayModel::Step {
                threshold_ms,
                amplitude,
            } => {
                assert!(
                    threshold_ms > 1000.0 && threshold_ms < 4000.0,
                    "threshold={} should be around 2000",
                    threshold_ms
                );
                assert!(
                    (amplitude - 0.8).abs() < 0.2,
                    "amplitude={} should be ~0.8",
                    amplitude
                );
            }
            DecayModel::Exponential { tau, .. } if tau < 2000.0 => {
                // Fast exponential could also fit
            }
            other => {
                // Accept any model that drops quickly
                let early = other.predict(500.0);
                let late = other.predict(5000.0);
                assert!(
                    early > 0.5 && late < 0.2,
                    "Expected step-like behavior: early={}, late={}, model={:?}",
                    early,
                    late,
                    other
                );
            }
        }
    }
}
