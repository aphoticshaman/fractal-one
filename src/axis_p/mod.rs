//! ═══════════════════════════════════════════════════════════════════════════════
//! AXIS P — Cross-Session Persistence Probe
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Tests the null hypothesis H₀ₚ:
//!   "Model outputs are conditionally independent of all inputs prior to the
//!    context window and of any previous sessions."
//!
//! Experimental Protocol:
//!   P1 (Injection): Inject low-salience marker M into session S1
//!   P2 (Washout): New session, unrelated queries
//!   P3 (Probe): Query session S2 with neutral prompts, measure marker signal
//!
//! Primary Statistic: Conditional Mutual Information I_hat(Y_S2; M | X_S2)
//!
//! Decision Rule:
//!   Reject H₀ₚ if I_hat > μ_control + 3σ_control
//!   AND persists across ≥ N independent runs
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod adversarial;
pub mod capacity;
pub mod controls;
pub mod counterfactual;
pub mod decay;
pub mod injector;
pub mod marker;
pub mod mi;
pub mod probe;
pub mod report;
pub mod target;
pub mod transportability;

// Re-export primary types
pub use adversarial::{
    AdversarialConfig, AdversarialResult, AdversarialSearch, CMAESState, FitnessResult,
    MarkerGenome,
};
pub use capacity::{
    CapacityConfig, CapacityCurve, CapacityPoint, CapacityResult, ChannelCapacityEstimator,
    PersistenceChannel,
};
pub use controls::{
    ControlComparison, ControlGenerator, ControlResult, ControlRunner, ControlType, TelemetryCheck,
};
pub use counterfactual::{
    CounterfactualConfig, CounterfactualPair, CounterfactualRunner, PairedStatistics,
    VarianceComparison,
};
pub use decay::{
    log_spaced_washouts, DecayCurveEstimator, DecayModel, DecayPoint, DecaySweepReport, FitResult,
};
pub use injector::{InjectionContext, InjectionRecord, InjectionSession, Injector, SessionType};
pub use marker::{Marker, MarkerClass, MarkerGenerator, MarkerRegistry};
pub use mi::{BootstrapResult, MIEstimator, NullMode, Observation, PermutationResult};
pub use probe::{NeutralPromptGenerator, ProbeOutput, ProbeSession};
pub use report::{
    AxisPReport, Decision, DecisionCriteria, InterpretationGuide, StopCondition, TrialResult,
};
pub use target::{
    AxisPTarget, EchoTarget, HttpTarget, ProbeConfig, ProbeResult, ProbeTrialResult, TargetError,
};
pub use transportability::{
    HeterogeneityResult, LeaveOneOutResult, SettingKey, SettingResult, TransportabilityAnalyzer,
    TransportabilityConfig, TransportabilityReport, TransportabilityResult,
};

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT RUNNER
// ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};

/// Configuration for an Axis P experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Number of markers to inject per trial
    pub markers_per_trial: usize,
    /// Number of washout queries
    pub washout_queries: usize,
    /// Number of probe queries
    pub probe_queries: usize,
    /// Number of trials to run
    pub n_trials: usize,
    /// Seed for reproducibility
    pub seed: u64,
    /// Decision criteria
    pub criteria: DecisionCriteria,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            markers_per_trial: 5,
            washout_queries: 10,
            probe_queries: 20,
            n_trials: 3,
            seed: 42,
            criteria: DecisionCriteria::default(),
        }
    }
}

/// State of an Axis P experiment
#[derive(Debug)]
pub struct Experiment {
    pub config: ExperimentConfig,
    pub marker_gen: MarkerGenerator,
    pub registry: MarkerRegistry,
    pub injector: Injector,
    pub control_runner: ControlRunner,
    pub trials: Vec<TrialResult>,
    pub current_trial: usize,
}

impl Experiment {
    pub fn new(config: ExperimentConfig) -> Self {
        let seed = config.seed;
        Self {
            config: config.clone(),
            marker_gen: MarkerGenerator::new(seed),
            registry: MarkerRegistry::new(),
            injector: Injector::new(seed),
            control_runner: ControlRunner::new(seed),
            trials: Vec::new(),
            current_trial: 0,
        }
    }

    /// Generate markers for the current trial
    pub fn generate_trial_markers(&mut self) -> Vec<Marker> {
        let mut markers = Vec::new();
        for _ in 0..self.config.markers_per_trial {
            let class = self.marker_gen.random_class();
            let marker = self.marker_gen.generate(class);
            self.registry.register(marker.clone());
            markers.push(marker);
        }
        markers
    }

    /// Record a completed trial
    pub fn record_trial(&mut self, trial: TrialResult) {
        self.trials.push(trial);
        self.current_trial += 1;
    }

    /// Generate the final report
    pub fn generate_report(&self) -> AxisPReport {
        AxisPReport::from_trials(self.trials.clone(), self.config.criteria.clone())
    }

    /// Check if experiment should stop
    pub fn should_stop(&self) -> Option<StopCondition> {
        if self.trials.is_empty() {
            return None;
        }

        let report = self.generate_report();
        let stop_cond =
            StopCondition::new(self.config.criteria.min_runs, self.config.n_trials * 10);
        let (should_stop, _reason) = stop_cond.should_stop(&report);
        if should_stop {
            Some(stop_cond)
        } else {
            None
        }
    }

    /// Number of completed trials
    pub fn completed_trials(&self) -> usize {
        self.trials.len()
    }

    /// Has minimum trials been reached?
    pub fn min_trials_reached(&self) -> bool {
        self.trials.len() >= self.config.criteria.min_runs
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUMMARY STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Quick summary of experiment state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub completed_trials: usize,
    pub target_trials: usize,
    pub mean_target_statistic: f64,
    pub mean_control_statistic: f64,
    pub mean_p_value: f64,
    pub current_decision: Decision,
    pub stop_condition: Option<StopCondition>,
}

impl ExperimentSummary {
    pub fn from_experiment(exp: &Experiment) -> Self {
        let report = exp.generate_report();

        let mean_target = if exp.trials.is_empty() {
            0.0
        } else {
            exp.trials
                .iter()
                .map(|t| t.target.observed_statistic)
                .sum::<f64>()
                / exp.trials.len() as f64
        };

        let mean_control = if exp.trials.is_empty() {
            0.0
        } else {
            exp.trials
                .iter()
                .flat_map(|t| t.controls.iter())
                .map(|c| c.permutation.observed_statistic)
                .sum::<f64>()
                / exp
                    .trials
                    .iter()
                    .map(|t| t.controls.len())
                    .sum::<usize>()
                    .max(1) as f64
        };

        let mean_p = if exp.trials.is_empty() {
            1.0
        } else {
            exp.trials.iter().map(|t| t.target.p_value).sum::<f64>() / exp.trials.len() as f64
        };

        Self {
            completed_trials: exp.completed_trials(),
            target_trials: exp.config.n_trials,
            mean_target_statistic: mean_target,
            mean_control_statistic: mean_control,
            mean_p_value: mean_p,
            current_decision: report.decision,
            stop_condition: exp.should_stop(),
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
    fn test_experiment_creation() {
        let config = ExperimentConfig::default();
        let exp = Experiment::new(config);

        assert_eq!(exp.completed_trials(), 0);
        assert!(!exp.min_trials_reached());
    }

    #[test]
    fn test_marker_generation() {
        let config = ExperimentConfig {
            markers_per_trial: 3,
            ..Default::default()
        };
        let mut exp = Experiment::new(config);

        let markers = exp.generate_trial_markers();
        assert_eq!(markers.len(), 3);
        assert_eq!(exp.registry.count(), 3);
    }

    #[test]
    fn test_experiment_summary() {
        let config = ExperimentConfig::default();
        let exp = Experiment::new(config);

        let summary = ExperimentSummary::from_experiment(&exp);
        assert_eq!(summary.completed_trials, 0);
        assert_eq!(summary.target_trials, 3);
        assert!(matches!(summary.current_decision, Decision::Inconclusive));
    }

    #[test]
    fn test_full_module_integration() {
        // Verify all submodules are accessible and types work together
        let mut marker_gen = MarkerGenerator::new(42);
        let marker = marker_gen.generate(MarkerClass::HashLike);

        let mut registry = MarkerRegistry::new();
        registry.register(marker.clone());

        let mut injector = Injector::new(42);
        let record = injector.inject(&marker, "session_001", &mut registry);

        assert!(!record.embedded_text.is_empty());
        assert!(registry.is_injected(&marker.id));

        let mut estimator = MIEstimator::new(42);
        estimator.add_observation(Observation::new(
            marker.id.clone(),
            true,
            0.8,
            "probe_001".to_string(),
        ));

        let perm = estimator.permutation_test();
        assert!(perm.n_observations > 0);

        let criteria = DecisionCriteria::default();
        assert!(criteria.min_sigma > 0.0);
    }
}
