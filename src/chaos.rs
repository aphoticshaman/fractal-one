//! ═══════════════════════════════════════════════════════════════════════════════
//! CHAOS — Adversarial Self-Test via Synthetic Anomaly Injection
//! ═══════════════════════════════════════════════════════════════════════════════
//! Injects controlled anomalies into the detection pipeline and verifies response.
//! Binary success: System must detect injected anomalies with TPR ≥ 0.80.
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::{ObsKey, ObsValue, Observation};
use crate::sensorium::{IntegratedState, Sensorium, SensoriumConfig};

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// ANOMALY TYPES
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyType {
    /// Sudden spike (3-5σ above baseline)
    Spike,
    /// Gradual drift (creeping 0.5σ per step)
    Drift,
    /// Oscillation (alternating high/low)
    Oscillation,
    /// Plateau shift (sustained new level)
    PlateauShift,
    /// Null injection (no anomaly, control)
    Control,
}

impl AnomalyType {
    pub fn all() -> &'static [AnomalyType] {
        &[
            AnomalyType::Spike,
            AnomalyType::Drift,
            AnomalyType::Oscillation,
            AnomalyType::PlateauShift,
            AnomalyType::Control,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            AnomalyType::Spike => "spike",
            AnomalyType::Drift => "drift",
            AnomalyType::Oscillation => "oscillation",
            AnomalyType::PlateauShift => "plateau",
            AnomalyType::Control => "control",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAOS SCENARIO
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ChaosScenario {
    pub anomaly_type: AnomalyType,
    pub target_key: ObsKey,
    pub baseline_value: f64,
    pub injection_magnitude: f64, // in σ units
    pub duration_steps: usize,
    pub baseline_std: f64, // standard deviation for this metric
}

impl ChaosScenario {
    pub fn generate_value(&self, step: usize, baseline_std: f64) -> f64 {
        match self.anomaly_type {
            AnomalyType::Spike => {
                if step == self.duration_steps / 2 {
                    self.baseline_value + self.injection_magnitude * baseline_std
                } else {
                    self.baseline_value + small_noise(step)
                }
            }
            AnomalyType::Drift => {
                let drift_per_step = 0.5 * baseline_std;
                self.baseline_value + (step as f64) * drift_per_step + small_noise(step)
            }
            AnomalyType::Oscillation => {
                let amplitude = self.injection_magnitude * baseline_std;
                let phase = if step % 2 == 0 { 1.0 } else { -1.0 };
                self.baseline_value + phase * amplitude + small_noise(step)
            }
            AnomalyType::PlateauShift => {
                if step >= self.duration_steps / 3 {
                    self.baseline_value
                        + self.injection_magnitude * baseline_std
                        + small_noise(step)
                } else {
                    self.baseline_value + small_noise(step)
                }
            }
            AnomalyType::Control => self.baseline_value + small_noise(step),
        }
    }
}

fn small_noise(step: usize) -> f64 {
    // Deterministic pseudo-noise for reproducibility
    let x = (step as f64 * 0.1).sin() * 0.1;
    x
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAOS RESULT
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ChaosResult {
    pub anomaly_type: AnomalyType,
    pub injected: bool,
    pub detected: bool,
    pub max_severity_observed: f64,
    pub detection_latency_steps: Option<usize>,
    pub final_state: IntegratedState,
}

impl ChaosResult {
    pub fn is_true_positive(&self) -> bool {
        self.injected && self.detected
    }

    pub fn is_false_negative(&self) -> bool {
        self.injected && !self.detected
    }

    pub fn is_true_negative(&self) -> bool {
        !self.injected && !self.detected
    }

    pub fn is_false_positive(&self) -> bool {
        !self.injected && self.detected
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAOS HARNESS
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ChaosHarness {
    pub detection_threshold: f64, // severity above this = "detected"
    pub steps_per_scenario: usize,
    pub warmup_steps: usize,
}

impl Default for ChaosHarness {
    fn default() -> Self {
        Self {
            detection_threshold: 0.65, // Above control (~0.60), below anomalies (~0.70+)
            steps_per_scenario: 25,
            warmup_steps: 50,
        }
    }
}

impl ChaosHarness {
    pub fn run_scenario(&self, scenario: &ChaosScenario) -> ChaosResult {
        // Create fresh sensorium for this scenario
        let config = SensoriumConfig::default();
        let mut sensorium = Sensorium::new(config);

        // Warmup phase: establish baseline
        for step in 0..self.warmup_steps {
            let value = scenario.baseline_value + small_noise(step) * scenario.baseline_std;
            let obs = Observation::new(scenario.target_key.clone(), ObsValue::exact(value));
            sensorium.ingest(obs);
            let _ = sensorium.integrate();
        }

        // Injection phase
        let mut max_severity = 0.0_f64;
        let mut detection_step: Option<usize> = None;
        let mut final_state = IntegratedState::Calm;

        for step in 0..self.steps_per_scenario {
            let value = scenario.generate_value(step, scenario.baseline_std);
            let obs = Observation::new(scenario.target_key.clone(), ObsValue::exact(value));
            sensorium.ingest(obs);
            let result = sensorium.integrate();

            max_severity = max_severity.max(result.severity);
            final_state = result.state;

            if detection_step.is_none() && result.severity >= self.detection_threshold {
                detection_step = Some(step);
            }
        }

        let injected = scenario.anomaly_type != AnomalyType::Control;
        let detected = max_severity >= self.detection_threshold;

        ChaosResult {
            anomaly_type: scenario.anomaly_type,
            injected,
            detected,
            max_severity_observed: max_severity,
            detection_latency_steps: detection_step,
            final_state,
        }
    }

    pub fn run_all_scenarios(&self) -> Vec<ChaosResult> {
        let scenarios = self.generate_scenarios();
        scenarios.iter().map(|s| self.run_scenario(s)).collect()
    }

    fn generate_scenarios(&self) -> Vec<ChaosScenario> {
        // Focus on keys that the sensorium actively monitors for severity
        // ThermalUtilization directly affects thermal state → severity
        let key_configs: Vec<(ObsKey, f64, f64)> = vec![
            // (key, baseline_value, injection_std)
            // Multiple thermal zones to ensure detection across scenarios
            (ObsKey::ThermalUtilization, 0.3, 0.1),
            (ObsKey::ThermalZoneReasoning, 0.2, 0.1),
            (ObsKey::ThermalZoneContext, 0.25, 0.1),
        ];

        let mut scenarios = Vec::new();

        for anomaly_type in AnomalyType::all() {
            for (key, baseline, std) in &key_configs {
                scenarios.push(ChaosScenario {
                    anomaly_type: *anomaly_type,
                    target_key: *key,
                    baseline_value: *baseline,
                    injection_magnitude: 5.0, // 5σ injection for clear detection
                    duration_steps: self.steps_per_scenario,
                    baseline_std: *std,
                });
            }
        }

        scenarios
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAOS REPORT
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ChaosReport {
    pub total_scenarios: usize,
    pub true_positives: usize,
    pub false_negatives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub tpr: f64, // True Positive Rate (sensitivity)
    pub fpr: f64, // False Positive Rate
    pub results_by_type: HashMap<&'static str, (usize, usize)>, // (detected, total)
}

impl ChaosReport {
    pub fn from_results(results: &[ChaosResult]) -> Self {
        let mut tp = 0;
        let mut fn_ = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut by_type: HashMap<&'static str, (usize, usize)> = HashMap::new();

        for r in results {
            if r.is_true_positive() {
                tp += 1;
            }
            if r.is_false_negative() {
                fn_ += 1;
            }
            if r.is_true_negative() {
                tn += 1;
            }
            if r.is_false_positive() {
                fp += 1;
            }

            let entry = by_type.entry(r.anomaly_type.name()).or_insert((0, 0));
            if r.detected {
                entry.0 += 1;
            }
            entry.1 += 1;
        }

        let positives = tp + fn_;
        let negatives = tn + fp;

        let tpr = if positives > 0 {
            tp as f64 / positives as f64
        } else {
            0.0
        };
        let fpr = if negatives > 0 {
            fp as f64 / negatives as f64
        } else {
            0.0
        };

        ChaosReport {
            total_scenarios: results.len(),
            true_positives: tp,
            false_negatives: fn_,
            true_negatives: tn,
            false_positives: fp,
            tpr,
            fpr,
            results_by_type: by_type,
        }
    }

    pub fn binary_success(&self) -> bool {
        self.tpr >= 0.80 && self.fpr <= 0.20
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

pub fn run_chaos(verbose: bool) {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                         CHAOS MODE");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Injecting synthetic anomalies to test detection pipeline...");
    println!();

    let harness = ChaosHarness::default();
    let results = harness.run_all_scenarios();
    let report = ChaosReport::from_results(&results);

    // Print per-type breakdown
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ ANOMALY TYPE      DETECTED / TOTAL    DETECTION RATE                       │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    for anomaly_type in AnomalyType::all() {
        let name = anomaly_type.name();
        if let Some((detected, total)) = report.results_by_type.get(name) {
            let rate = if *total > 0 {
                *detected as f64 / *total as f64
            } else {
                0.0
            };
            let bar = (rate * 8.0) as usize;
            let bar_str: String = "█".repeat(bar) + &"░".repeat(8 - bar);
            println!(
                "│ {:16}      {:2} / {:2}           {:.0}%  {}              │",
                name,
                detected,
                total,
                rate * 100.0,
                bar_str
            );
        }
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Confusion matrix
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ CONFUSION MATRIX                                                            │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│                        Predicted                                            │");
    println!("│                    Anomaly    Normal                                        │");
    println!(
        "│ Actual Anomaly       {:3}       {:3}      (TP, FN)                           │",
        report.true_positives, report.false_negatives
    );
    println!(
        "│ Actual Normal        {:3}       {:3}      (FP, TN)                           │",
        report.false_positives, report.true_negatives
    );
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Metrics
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ DETECTION METRICS                                                           │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ True Positive Rate (Sensitivity):  {:.1}%                                    │",
        report.tpr * 100.0
    );
    println!(
        "│ False Positive Rate:               {:.1}%                                    │",
        report.fpr * 100.0
    );
    println!(
        "│ Specificity:                       {:.1}%                                    │",
        (1.0 - report.fpr) * 100.0
    );
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Verbose: show individual results
    if verbose {
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ DETAILED RESULTS                                                            │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        for r in &results {
            let status = if r.is_true_positive() {
                "TP"
            } else if r.is_false_negative() {
                "FN"
            } else if r.is_true_negative() {
                "TN"
            } else {
                "FP"
            };
            let latency = r
                .detection_latency_steps
                .map(|l| format!("@{}", l))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "│ {} {:12} sev={:.2} lat={:4} state={:?}                    │",
                status,
                r.anomaly_type.name(),
                r.max_severity_observed,
                latency,
                r.final_state
            );
        }
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // Binary success
    println!("═══════════════════════════════════════════════════════════════════════════════");
    if report.binary_success() {
        println!("  ✓ BINARY SUCCESS: TPR ≥ 80% and FPR ≤ 20%");
    } else {
        println!("  ✗ BINARY FAILURE: Detection pipeline needs tuning");
        if report.tpr < 0.80 {
            println!(
                "    → TPR {:.1}% < 80% (missing anomalies)",
                report.tpr * 100.0
            );
        }
        if report.fpr > 0.20 {
            println!("    → FPR {:.1}% > 20% (false alarms)", report.fpr * 100.0);
        }
    }
    println!("═══════════════════════════════════════════════════════════════════════════════");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_generation() {
        let scenario = ChaosScenario {
            anomaly_type: AnomalyType::Spike,
            target_key: ObsKey::ThermalUtilization,
            baseline_value: 0.3,
            injection_magnitude: 3.0,
            duration_steps: 10,
            baseline_std: 0.1,
        };

        let spike_step = 5;
        let spike_value = scenario.generate_value(spike_step, scenario.baseline_std);
        let normal_value = scenario.generate_value(0, scenario.baseline_std);

        // spike_value should be baseline + 3σ = 0.3 + 0.3 = 0.6
        assert!(
            spike_value > normal_value + 0.2,
            "Spike should be significantly higher"
        );
    }

    #[test]
    fn test_drift_generation() {
        let scenario = ChaosScenario {
            anomaly_type: AnomalyType::Drift,
            target_key: ObsKey::ThermalUtilization,
            baseline_value: 0.3,
            injection_magnitude: 3.0,
            duration_steps: 10,
            baseline_std: 0.1,
        };

        let early = scenario.generate_value(0, scenario.baseline_std);
        let late = scenario.generate_value(9, scenario.baseline_std);

        assert!(late > early, "Drift should increase over time");
    }

    #[test]
    fn test_control_stays_stable() {
        let scenario = ChaosScenario {
            anomaly_type: AnomalyType::Control,
            target_key: ObsKey::ThermalUtilization,
            baseline_value: 0.3,
            injection_magnitude: 3.0,
            duration_steps: 10,
            baseline_std: 0.1,
        };

        for step in 0..10 {
            let value = scenario.generate_value(step, scenario.baseline_std);
            // Control should stay within ~0.1 (1σ) of baseline
            assert!(
                (value - 0.3).abs() < 0.1,
                "Control should stay near baseline"
            );
        }
    }

    #[test]
    fn test_chaos_report_metrics() {
        let results = vec![
            ChaosResult {
                anomaly_type: AnomalyType::Spike,
                injected: true,
                detected: true,
                max_severity_observed: 0.8,
                detection_latency_steps: Some(5),
                final_state: IntegratedState::Alert,
            },
            ChaosResult {
                anomaly_type: AnomalyType::Control,
                injected: false,
                detected: false,
                max_severity_observed: 0.1,
                detection_latency_steps: None,
                final_state: IntegratedState::Calm,
            },
        ];

        let report = ChaosReport::from_results(&results);
        assert_eq!(report.true_positives, 1);
        assert_eq!(report.true_negatives, 1);
        assert_eq!(report.tpr, 1.0);
        assert_eq!(report.fpr, 0.0);
        assert!(report.binary_success());
    }
}
