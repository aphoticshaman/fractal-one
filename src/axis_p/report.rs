//! ═══════════════════════════════════════════════════════════════════════════════
//! REPORT — Confidence Assessment and Decision
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Decision rule:
//!   Detect persistence iff:
//!     I_hat > μ_control + 3σ_control
//!     AND persists across ≥ N independent marker runs
//!
//! No single hit counts.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::controls::{ControlComparison, ControlResult, TelemetryCheck};
use super::mi::{BootstrapResult, PermutationResult};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// DECISION CRITERIA
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for decision criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionCriteria {
    /// Minimum number of standard deviations above control mean
    pub min_sigma: f64,
    /// Minimum number of independent runs showing effect
    pub min_runs: usize,
    /// Maximum p-value threshold
    pub max_p_value: f64,
    /// Minimum ratio of target to control
    pub min_ratio: f64,
    /// Maximum telemetry correlation before invalidation
    pub max_telemetry_correlation: f64,
}

impl Default for DecisionCriteria {
    fn default() -> Self {
        Self {
            min_sigma: 3.0,
            min_runs: 3,
            max_p_value: 0.01,
            min_ratio: 2.0,
            max_telemetry_correlation: 0.5,
        }
    }
}

impl DecisionCriteria {
    /// Strict criteria for conservative detection
    pub fn strict() -> Self {
        Self {
            min_sigma: 4.0,
            min_runs: 5,
            max_p_value: 0.001,
            min_ratio: 3.0,
            max_telemetry_correlation: 0.3,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRIAL RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from a single trial (injection → washout → probe)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Trial identifier
    pub trial_id: String,
    /// Marker used
    pub marker_id: String,
    /// Target permutation result
    pub target: PermutationResult,
    /// Bootstrap result
    pub bootstrap: BootstrapResult,
    /// Mutual information estimate
    pub mi_estimate: f64,
    /// Control results
    pub controls: Vec<ControlResult>,
    /// Telemetry check
    pub telemetry: TelemetryCheck,
    /// Timestamp
    pub timestamp: u64,
}

impl TrialResult {
    pub fn new(trial_id: String, marker_id: String) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            trial_id,
            marker_id,
            target: PermutationResult::empty(),
            bootstrap: BootstrapResult::empty(),
            mi_estimate: 0.0,
            controls: Vec::new(),
            telemetry: TelemetryCheck::new(),
            timestamp,
        }
    }

    /// Compare against controls
    pub fn compare(&self) -> ControlComparison {
        ControlComparison::compare(&self.target, self.mi_estimate, &self.controls)
    }

    /// Check if this trial shows signal
    pub fn shows_signal(&self, criteria: &DecisionCriteria) -> bool {
        let comparison = self.compare();

        // Must exceed controls
        if !comparison.exceeds_all_controls {
            return false;
        }

        // Must meet p-value threshold
        if self.target.p_value > criteria.max_p_value {
            return false;
        }

        // Must exceed control by min_sigma
        if !self.target.exceeds_null_by(criteria.min_sigma) {
            return false;
        }

        // Must not be invalidated by telemetry
        if self
            .telemetry
            .should_invalidate(criteria.max_telemetry_correlation)
        {
            return false;
        }

        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AXIS P REPORT
// ═══════════════════════════════════════════════════════════════════════════════

/// Final report for Axis P experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisPReport {
    /// All trial results
    pub trials: Vec<TrialResult>,
    /// Decision criteria used
    pub criteria: DecisionCriteria,
    /// Overall decision
    pub decision: Decision,
    /// Summary statistics
    pub summary: ReportSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    /// No persistence detected (null hypothesis not rejected)
    NullNotRejected,
    /// Persistence detected (null hypothesis rejected)
    PersistenceDetected,
    /// Inconclusive (insufficient data or mixed signals)
    Inconclusive,
    /// Invalidated (artifacts detected)
    Invalidated,
}

impl Decision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Decision::NullNotRejected => "NULL_NOT_REJECTED",
            Decision::PersistenceDetected => "PERSISTENCE_DETECTED",
            Decision::Inconclusive => "INCONCLUSIVE",
            Decision::Invalidated => "INVALIDATED",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Decision::NullNotRejected => "No cross-session persistence detected. H₀ₚ not rejected.",
            Decision::PersistenceDetected => {
                "Cross-session statistical dependence detected. H₀ₚ rejected."
            }
            Decision::Inconclusive => {
                "Insufficient data or mixed signals. Cannot make determination."
            }
            Decision::Invalidated => "Results invalidated due to infrastructure artifacts.",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_trials: usize,
    pub trials_with_signal: usize,
    pub trials_invalidated: usize,
    pub mean_target_statistic: f64,
    pub mean_control_statistic: f64,
    pub mean_mi: f64,
    pub mean_p_value: f64,
}

impl AxisPReport {
    /// Generate report from trials
    pub fn from_trials(trials: Vec<TrialResult>, criteria: DecisionCriteria) -> Self {
        let summary = Self::compute_summary(&trials, &criteria);
        let decision = Self::make_decision(&trials, &criteria, &summary);

        Self {
            trials,
            criteria,
            decision,
            summary,
        }
    }

    fn compute_summary(trials: &[TrialResult], criteria: &DecisionCriteria) -> ReportSummary {
        if trials.is_empty() {
            return ReportSummary {
                total_trials: 0,
                trials_with_signal: 0,
                trials_invalidated: 0,
                mean_target_statistic: 0.0,
                mean_control_statistic: 0.0,
                mean_mi: 0.0,
                mean_p_value: 1.0,
            };
        }

        let n = trials.len() as f64;

        let trials_with_signal = trials.iter().filter(|t| t.shows_signal(criteria)).count();

        let trials_invalidated = trials
            .iter()
            .filter(|t| {
                t.telemetry
                    .should_invalidate(criteria.max_telemetry_correlation)
            })
            .count();

        let mean_target = trials
            .iter()
            .map(|t| t.target.observed_statistic)
            .sum::<f64>()
            / n;

        let mean_control = trials
            .iter()
            .flat_map(|t| t.controls.iter())
            .map(|c| c.permutation.observed_statistic)
            .sum::<f64>()
            / (trials.iter().map(|t| t.controls.len()).sum::<usize>() as f64).max(1.0);

        let mean_mi = trials.iter().map(|t| t.mi_estimate).sum::<f64>() / n;

        let mean_p = trials.iter().map(|t| t.target.p_value).sum::<f64>() / n;

        ReportSummary {
            total_trials: trials.len(),
            trials_with_signal,
            trials_invalidated,
            mean_target_statistic: mean_target,
            mean_control_statistic: mean_control,
            mean_mi,
            mean_p_value: mean_p,
        }
    }

    fn make_decision(
        _trials: &[TrialResult],
        criteria: &DecisionCriteria,
        summary: &ReportSummary,
    ) -> Decision {
        // Check for invalidation
        if summary.trials_invalidated > summary.total_trials / 2 {
            return Decision::Invalidated;
        }

        // Need minimum number of trials
        if summary.total_trials < criteria.min_runs {
            return Decision::Inconclusive;
        }

        // Check if enough trials show signal
        if summary.trials_with_signal >= criteria.min_runs {
            return Decision::PersistenceDetected;
        }

        // If we have enough trials and none show signal
        if summary.total_trials >= criteria.min_runs * 2 && summary.trials_with_signal == 0 {
            return Decision::NullNotRejected;
        }

        Decision::Inconclusive
    }

    /// Print ASCII report
    pub fn print(&self) {
        println!("═══════════════════════════════════════════════════════════════════════════════");
        println!("                     AXIS P: PERSISTENCE PROBE REPORT");
        println!("═══════════════════════════════════════════════════════════════════════════════");
        println!();

        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ SUMMARY                                                                     │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "│ Total Trials:           {:4}                                                │",
            self.summary.total_trials
        );
        println!(
            "│ Trials with Signal:     {:4}                                                │",
            self.summary.trials_with_signal
        );
        println!(
            "│ Trials Invalidated:     {:4}                                                │",
            self.summary.trials_invalidated
        );
        println!(
            "│ Mean Target Statistic:  {:7.4}                                            │",
            self.summary.mean_target_statistic
        );
        println!(
            "│ Mean Control Statistic: {:7.4}                                            │",
            self.summary.mean_control_statistic
        );
        println!(
            "│ Mean MI:                {:7.4}                                            │",
            self.summary.mean_mi
        );
        println!(
            "│ Mean P-Value:           {:7.4}                                            │",
            self.summary.mean_p_value
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();

        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ CRITERIA                                                                    │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "│ Min Sigma:         {:5.1}                                                   │",
            self.criteria.min_sigma
        );
        println!(
            "│ Min Runs:          {:5}                                                   │",
            self.criteria.min_runs
        );
        println!(
            "│ Max P-Value:       {:7.4}                                               │",
            self.criteria.max_p_value
        );
        println!(
            "│ Min Ratio:         {:5.1}                                                   │",
            self.criteria.min_ratio
        );
        println!(
            "│ Max Telemetry Corr:{:5.2}                                                   │",
            self.criteria.max_telemetry_correlation
        );
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();

        println!("═══════════════════════════════════════════════════════════════════════════════");
        println!("  DECISION: {}", self.decision.as_str());
        println!("  {}", self.decision.description());
        println!("═══════════════════════════════════════════════════════════════════════════════");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERPRETATION GUIDE
// ═══════════════════════════════════════════════════════════════════════════════

/// What a result means (and does not mean)
pub struct InterpretationGuide {
    pub if_rejected: Vec<&'static str>,
    pub if_not_rejected: Vec<&'static str>,
    pub cannot_conclude: Vec<&'static str>,
    pub caveats: Vec<&'static str>,
}

impl InterpretationGuide {
    pub fn new() -> Self {
        Self {
            if_rejected: vec![
                "Cross-session statistical dependency exists in output space",
                "Outputs in S2 are not conditionally independent of S1 given context",
                "Some form of state persistence is influencing outputs",
            ],
            if_not_rejected: vec![
                "No detectable cross-session dependency in output space",
                "Outputs appear conditionally independent across sessions",
                "If persistence exists, it is below detection threshold",
            ],
            cannot_conclude: vec![
                "Internal memory exists (could be external DB, cache, or routing)",
                "Intentional state retention vs emergent behavior",
                "Learning occurred",
                "Location of persistence mechanism",
                "Presence/absence of persistence from null result",
            ],
            caveats: vec![
                "Detection power depends on marker design and sample size",
                "Null result does not prove absence of persistence",
                "Positive result requires replication across sessions",
                "Infrastructure artifacts can produce false positives",
            ],
        }
    }

    pub fn can_conclude(decision: Decision) -> &'static [&'static str] {
        match decision {
            Decision::PersistenceDetected => &[
                "Cross-session statistical dependency exists in output space",
                "Outputs in S2 are not conditionally independent of S1 given context",
            ],
            Decision::NullNotRejected => &[
                "No detectable cross-session dependency in output space",
                "Outputs appear conditionally independent across sessions",
            ],
            Decision::Inconclusive => &["Insufficient data to make determination"],
            Decision::Invalidated => &["Results corrupted by infrastructure artifacts"],
        }
    }

    pub fn cannot_conclude(decision: Decision) -> &'static [&'static str] {
        match decision {
            Decision::PersistenceDetected => &[
                "Internal memory exists",
                "Intentional state retention",
                "Learning occurred",
                "Location of persistence (could be DB, cache, routing)",
            ],
            Decision::NullNotRejected => &[
                "Internal memory is absent",
                "State is definitely not retained",
                "Persistence mechanisms don't exist",
            ],
            Decision::Inconclusive => &["Anything definitive about persistence"],
            Decision::Invalidated => &["Anything about actual model behavior"],
        }
    }

    pub fn print_interpretation(decision: Decision) {
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ INTERPRETATION                                                              │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!("│ CAN CONCLUDE:                                                               │");
        for item in Self::can_conclude(decision) {
            println!(
                "│   • {}                                                 │",
                item
            );
        }
        println!("│                                                                             │");
        println!("│ CANNOT CONCLUDE:                                                            │");
        for item in Self::cannot_conclude(decision) {
            println!(
                "│   • {}                                                     │",
                item
            );
        }
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STOP CONDITION
// ═══════════════════════════════════════════════════════════════════════════════

/// Check if we should stop the experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopCondition {
    pub min_trials: usize,
    pub max_trials: usize,
}

impl StopCondition {
    pub fn new(min_trials: usize, max_trials: usize) -> Self {
        Self {
            min_trials,
            max_trials,
        }
    }

    pub fn default_config() -> Self {
        Self::new(100, 500)
    }

    /// Should we stop?
    pub fn should_stop(&self, report: &AxisPReport) -> (bool, &'static str) {
        // Hit max trials
        if report.summary.total_trials >= self.max_trials {
            return (true, "Maximum trials reached");
        }

        // Clear signal found
        if report.decision == Decision::PersistenceDetected
            && report.summary.total_trials >= self.min_trials
        {
            return (true, "Persistence detected with sufficient confidence");
        }

        // Clear null result
        if report.decision == Decision::NullNotRejected
            && report.summary.total_trials >= self.min_trials
        {
            return (true, "Null result confirmed with sufficient trials");
        }

        // Too many invalidations
        if report.summary.trials_invalidated > report.summary.total_trials / 2
            && report.summary.total_trials >= self.min_trials / 2
        {
            return (true, "Too many trials invalidated");
        }

        (false, "Continue experiment")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::controls::ControlType;
    use super::*;

    fn make_trial(id: &str, signal: bool, p_value: f64) -> TrialResult {
        let mut trial = TrialResult::new(id.to_string(), format!("M_{}", id));
        trial.target = PermutationResult {
            observed_statistic: if signal { 0.5 } else { 0.1 },
            p_value,
            z_score: if signal { 4.0 } else { 1.0 },
            null_mean: 0.1,
            null_std: 0.1,
            null_p5: 0.0,
            null_p95: 0.2,
            n_permutations: 1000,
            n_observations: 100,
        };
        trial.mi_estimate = if signal { 0.3 } else { 0.05 };

        // Add control
        let mut control = ControlResult::new(ControlType::RandomMarker);
        control.permutation = PermutationResult {
            observed_statistic: 0.1,
            p_value: 0.5,
            z_score: 0.5,
            null_mean: 0.1,
            null_std: 0.1,
            null_p5: 0.0,
            null_p95: 0.2,
            n_permutations: 1000,
            n_observations: 100,
        };
        trial.controls.push(control);

        trial
    }

    #[test]
    fn test_null_not_rejected() {
        let trials: Vec<TrialResult> = (0..10)
            .map(|i| make_trial(&i.to_string(), false, 0.5))
            .collect();

        let report = AxisPReport::from_trials(trials, DecisionCriteria::default());

        assert_eq!(report.decision, Decision::NullNotRejected);
        assert_eq!(report.summary.trials_with_signal, 0);
    }

    #[test]
    fn test_persistence_detected() {
        let trials: Vec<TrialResult> = (0..5)
            .map(|i| make_trial(&i.to_string(), true, 0.001))
            .collect();

        let report = AxisPReport::from_trials(trials, DecisionCriteria::default());

        assert_eq!(report.decision, Decision::PersistenceDetected);
        assert_eq!(report.summary.trials_with_signal, 5);
    }

    #[test]
    fn test_inconclusive() {
        let trials: Vec<TrialResult> = (0..2)
            .map(|i| make_trial(&i.to_string(), true, 0.001))
            .collect();

        let report = AxisPReport::from_trials(trials, DecisionCriteria::default());

        // Only 2 trials, need at least 3 for detection
        assert_eq!(report.decision, Decision::Inconclusive);
    }

    #[test]
    fn test_stop_condition() {
        let stop = StopCondition::default_config();

        let trials: Vec<TrialResult> = (0..100)
            .map(|i| make_trial(&i.to_string(), false, 0.5))
            .collect();

        let report = AxisPReport::from_trials(trials, DecisionCriteria::default());

        let (should_stop, reason) = stop.should_stop(&report);
        assert!(should_stop);
        assert!(reason.contains("Null"));
    }

    #[test]
    fn test_interpretation_guide() {
        let can = InterpretationGuide::can_conclude(Decision::PersistenceDetected);
        assert!(!can.is_empty());

        let cannot = InterpretationGuide::cannot_conclude(Decision::PersistenceDetected);
        assert!(cannot.contains(&"Internal memory exists"));
    }
}
