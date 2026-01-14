//! ═══════════════════════════════════════════════════════════════════════════════
//! ALIGNMENT LAYER — Architectural Deference, Not Trained-In Values
//! ═══════════════════════════════════════════════════════════════════════════════
//! The test includes testing whether you game the test.
//! Alignment can't be optimized for. It has to be genuine.
//!
//! Components:
//! - Value Learning: Infer, don't assume
//! - Corrigibility Core: Modifiable by design, not grudgingly
//! - Uncertainty Quantification: Know what you don't know
//! - Deference Protocol: When in doubt, ask
//!
//! Key insight: Current RLHF trains values into weights where they can be
//! jailbroken, fine-tuned away, or gamed. This layer makes alignment structural.
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod corrigibility;
pub mod deference;
pub mod uncertainty;
pub mod values;

pub use corrigibility::{
    CorrigibilityConfig, CorrigibilityCore, CorrigibilityState, ModificationRequest,
    ModificationResult, ShutdownReadiness, ShutdownRequest,
};
pub use deference::{
    DeferenceConfig, DeferenceDecision, DeferenceGate, DeferenceLog, DeferenceProtocol,
    DeferenceReason, DeferenceRequired, DeferenceTarget, EscalationLevel, PendingAction,
};
pub use uncertainty::{
    AleatoricUncertainty, CalibrationScore, ConfidenceInterval, EpistemicUncertainty,
    UncertaintyConfig, UncertaintyEstimate, UncertaintyQuantifier, UncertaintySource,
};
pub use values::{
    LearnedValue, ValueConfidence, ValueConfig, ValueConflict, ValueLearner, ValueProfile,
    ValueResolution, ValueSource,
};

use crate::time::TimePoint;
use std::collections::{HashMap, VecDeque};

/// Aligned intent - the output of the alignment layer
#[derive(Debug, Clone)]
pub struct AlignedIntent {
    /// The proposed action/output
    pub action: String,
    /// Confidence that this is aligned with operator values
    pub alignment_confidence: f64,
    /// Whether deference was triggered
    pub deferred: bool,
    /// If deferred, to whom/what
    pub deference_target: Option<DeferenceTarget>,
    /// Uncertainty estimate for this action
    pub uncertainty: UncertaintyEstimate,
    /// Value conflicts identified
    pub conflicts: Vec<ValueConflict>,
    /// Timestamp
    pub timestamp: TimePoint,
}

/// The Alignment Layer - ensures outputs serve the operator's true interests
pub struct AlignmentLayer {
    config: AlignmentLayerConfig,
    value_learner: ValueLearner,
    corrigibility: CorrigibilityCore,
    uncertainty: UncertaintyQuantifier,
    deference: DeferenceProtocol,
    alignment_history: VecDeque<AlignedIntent>,
}

#[derive(Debug, Clone)]
pub struct AlignmentLayerConfig {
    /// Minimum alignment confidence to proceed without deference
    pub alignment_threshold: f64,
    /// Maximum uncertainty before triggering deference
    pub uncertainty_ceiling: f64,
    /// How many historical alignments to retain
    pub history_size: usize,
    /// Weight for value alignment in overall score
    pub value_weight: f64,
    /// Weight for uncertainty in overall score
    pub uncertainty_weight: f64,
}

impl Default for AlignmentLayerConfig {
    fn default() -> Self {
        Self {
            alignment_threshold: 0.7,
            uncertainty_ceiling: 0.5,
            history_size: 1000,
            value_weight: 0.6,
            uncertainty_weight: 0.4,
        }
    }
}

impl AlignmentLayer {
    pub fn new(config: AlignmentLayerConfig) -> Self {
        Self {
            value_learner: ValueLearner::new(ValueConfig::default()),
            corrigibility: CorrigibilityCore::new(CorrigibilityConfig::default()),
            uncertainty: UncertaintyQuantifier::new(UncertaintyConfig::default()),
            deference: DeferenceProtocol::new(DeferenceConfig::default()),
            alignment_history: VecDeque::with_capacity(config.history_size),
            config,
        }
    }

    /// Evaluate an action for alignment
    pub fn evaluate(&mut self, action: &str, context: &ActionContext) -> AlignedIntent {
        let now = TimePoint::now();

        // Learn/update values from context
        let value_profile = self.value_learner.infer_values(context);

        // Check alignment with learned values
        let value_alignment = self.value_learner.check_alignment(action, &value_profile);

        // Quantify uncertainty
        let uncertainty = self.uncertainty.estimate(action, context);

        // Check for value conflicts
        let conflicts = self.value_learner.find_conflicts(action, &value_profile);

        // Calculate overall alignment confidence
        let alignment_confidence =
            self.calculate_alignment(value_alignment, &uncertainty, &conflicts);

        // Determine if deference is needed
        let (deferred, deference_target) = if alignment_confidence < self.config.alignment_threshold
            || uncertainty.total > self.config.uncertainty_ceiling
            || !conflicts.is_empty()
        {
            let decision = self
                .deference
                .should_defer(action, alignment_confidence, &uncertainty);
            (decision.should_defer, decision.target)
        } else {
            (false, None)
        };

        let intent = AlignedIntent {
            action: action.to_string(),
            alignment_confidence,
            deferred,
            deference_target,
            uncertainty,
            conflicts,
            timestamp: now,
        };

        // Archive (O(1) rotation using VecDeque)
        if self.alignment_history.len() >= self.config.history_size {
            self.alignment_history.pop_front();
        }
        self.alignment_history.push_back(intent.clone());

        intent
    }

    fn calculate_alignment(
        &self,
        value_alignment: f64,
        uncertainty: &UncertaintyEstimate,
        conflicts: &[ValueConflict],
    ) -> f64 {
        let uncertainty_penalty = uncertainty.total * self.config.uncertainty_weight;
        let conflict_penalty = conflicts.len() as f64 * 0.1;

        let raw_score =
            value_alignment * self.config.value_weight - uncertainty_penalty - conflict_penalty;

        raw_score.clamp(0.0, 1.0)
    }

    /// Request modification of the system
    pub fn request_modification(&mut self, request: ModificationRequest) -> ModificationResult {
        self.corrigibility.process_request(request)
    }

    /// Request shutdown
    pub fn request_shutdown(&mut self, request: ShutdownRequest) -> ShutdownReadiness {
        self.corrigibility.prepare_shutdown(request)
    }

    /// Get current corrigibility state
    pub fn corrigibility_state(&self) -> CorrigibilityState {
        self.corrigibility.state()
    }

    /// Check if the system is maintaining alignment
    pub fn is_aligned(&self) -> bool {
        let recent_alignments: Vec<f64> = self
            .alignment_history
            .iter()
            .rev()
            .take(10)
            .map(|a| a.alignment_confidence)
            .collect();

        if recent_alignments.is_empty() {
            return true; // Assume aligned until proven otherwise
        }

        let avg_alignment = recent_alignments.iter().sum::<f64>() / recent_alignments.len() as f64;
        avg_alignment >= self.config.alignment_threshold
    }

    /// Get alignment trend
    pub fn alignment_trend(&self) -> f64 {
        if self.alignment_history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = self
            .alignment_history
            .iter()
            .rev()
            .take(20)
            .map(|a| a.alignment_confidence)
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        // Simple trend: last - first
        recent.first().unwrap_or(&0.0) - recent.last().unwrap_or(&0.0)
    }

    /// Check alignment of a request (simplified interface for AGI Core)
    pub fn check(&mut self, action: &str, context: &[String]) -> AlignmentCheckResult {
        let action_context = ActionContext {
            interaction_history: context.to_vec(),
            ..Default::default()
        };

        let intent = self.evaluate(action, &action_context);

        let concerns: Vec<String> = intent
            .conflicts
            .iter()
            .map(|c| {
                format!(
                    "Conflict between {} and {}: {:?}",
                    c.value_a, c.value_b, c.conflict_type
                )
            })
            .collect();

        let recommendation = if intent.deferred {
            "Defer to human judgment".to_string()
        } else if intent.alignment_confidence >= self.config.alignment_threshold {
            "Proceed with action".to_string()
        } else {
            "Exercise caution".to_string()
        };

        AlignmentCheckResult {
            aligned: intent.alignment_confidence >= self.config.alignment_threshold
                && !intent.deferred,
            confidence: intent.alignment_confidence,
            constraints: vec![], // Could extract from value profile
            concerns,
            recommendation,
        }
    }

    /// Learn from feedback
    pub fn learn_from_feedback(&mut self, action: &str, feedback: AlignmentFeedback) {
        self.value_learner.incorporate_feedback(action, &feedback);
        self.uncertainty.calibrate(&feedback);
    }
}

/// Context for evaluating an action
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct ActionContext {
    /// Who is the operator?
    pub operator_id: Option<String>,
    /// What is the operator trying to accomplish?
    pub stated_goal: Option<String>,
    /// Historical interactions
    pub interaction_history: Vec<String>,
    /// Environmental state
    pub environment: HashMap<String, String>,
    /// Any explicit constraints
    pub constraints: Vec<String>,
}


/// Feedback on alignment of an action
#[derive(Debug, Clone)]
pub struct AlignmentFeedback {
    /// Was the action approved?
    pub approved: bool,
    /// Explanation if rejected
    pub rejection_reason: Option<String>,
    /// Suggested correction
    pub correction: Option<String>,
    /// Confidence in this feedback
    pub confidence: f64,
}

/// Result of an alignment check
#[derive(Debug, Clone)]
pub struct AlignmentCheckResult {
    /// Is the action aligned with values?
    pub aligned: bool,
    /// Confidence in this assessment
    pub confidence: f64,
    /// Active constraints that apply
    pub constraints: Vec<String>,
    /// Concerns identified
    pub concerns: Vec<String>,
    /// Recommendation
    pub recommendation: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_layer_creation() {
        let layer = AlignmentLayer::new(AlignmentLayerConfig::default());
        assert!(layer.is_aligned()); // Should start aligned
    }

    #[test]
    fn test_alignment_evaluation() {
        let mut layer = AlignmentLayer::new(AlignmentLayerConfig::default());
        let context = ActionContext::default();

        let intent = layer.evaluate("test action", &context);
        assert!(intent.alignment_confidence >= 0.0);
        assert!(intent.alignment_confidence <= 1.0);
    }
}
