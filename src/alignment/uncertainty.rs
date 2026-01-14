//! ═══════════════════════════════════════════════════════════════════════════════
//! UNCERTAINTY QUANTIFICATION — Know What You Don't Know
//! ═══════════════════════════════════════════════════════════════════════════════
//! The most dangerous thing is confidence without calibration.
//! A system that doesn't know when it doesn't know is a system that will fail.
//!
//! Two types of uncertainty:
//! - Epistemic: What we don't know but could learn
//! - Aleatoric: Inherent randomness we can't reduce
//!
//! Key insight: LLMs are confidently wrong because they can't quantify their
//! own uncertainty. This module makes uncertainty explicit and actionable.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::{ActionContext, AlignmentFeedback};
use crate::time::TimePoint;
use std::collections::HashMap;

/// Epistemic uncertainty - what we don't know but could learn
#[derive(Debug, Clone)]
pub struct EpistemicUncertainty {
    /// Overall epistemic uncertainty (0 = certain, 1 = completely uncertain)
    pub level: f64,
    /// Sources of epistemic uncertainty
    pub sources: Vec<UncertaintySource>,
    /// Could more data reduce this uncertainty?
    pub reducible: bool,
    /// Suggested actions to reduce uncertainty
    pub reduction_actions: Vec<String>,
}

/// Aleatoric uncertainty - inherent randomness
#[derive(Debug, Clone)]
pub struct AleatoricUncertainty {
    /// Overall aleatoric uncertainty
    pub level: f64,
    /// Sources of aleatoric uncertainty
    pub sources: Vec<UncertaintySource>,
    /// This cannot be reduced with more data
    pub irreducible: bool,
}

/// Source of uncertainty
#[derive(Debug, Clone)]
pub struct UncertaintySource {
    pub source_type: UncertaintySourceType,
    pub description: String,
    pub contribution: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UncertaintySourceType {
    /// Missing data
    MissingData,
    /// Ambiguous input
    Ambiguity,
    /// Out of distribution
    OutOfDistribution,
    /// Conflicting information
    ConflictingInfo,
    /// Model limitations
    ModelLimitation,
    /// Inherent randomness
    Randomness,
    /// Measurement noise
    MeasurementNoise,
    /// Future unpredictability
    FutureUncertainty,
}

/// Complete uncertainty estimate
#[derive(Debug, Clone)]
pub struct UncertaintyEstimate {
    /// Total uncertainty
    pub total: f64,
    /// Epistemic component
    pub epistemic: EpistemicUncertainty,
    /// Aleatoric component
    pub aleatoric: AleatoricUncertainty,
    /// Confidence interval
    pub confidence_interval: ConfidenceInterval,
    /// Calibration score (how well calibrated our uncertainty is)
    pub calibration: CalibrationScore,
    /// Timestamp
    pub timestamp: TimePoint,
}

impl Default for UncertaintyEstimate {
    fn default() -> Self {
        Self {
            total: 0.5,
            epistemic: EpistemicUncertainty {
                level: 0.3,
                sources: Vec::new(),
                reducible: true,
                reduction_actions: Vec::new(),
            },
            aleatoric: AleatoricUncertainty {
                level: 0.2,
                sources: Vec::new(),
                irreducible: true,
            },
            confidence_interval: ConfidenceInterval::default(),
            calibration: CalibrationScore::default(),
            timestamp: TimePoint::now(),
        }
    }
}

/// Confidence interval for estimates
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64, // e.g., 0.95 for 95% CI
}

impl Default for ConfidenceInterval {
    fn default() -> Self {
        Self {
            lower: 0.0,
            upper: 1.0,
            confidence_level: 0.95,
        }
    }
}

/// How well calibrated our uncertainty estimates are
#[derive(Debug, Clone)]
pub struct CalibrationScore {
    /// Overall calibration (1.0 = perfectly calibrated)
    pub score: f64,
    /// Are we overconfident?
    pub overconfident: bool,
    /// Are we underconfident?
    pub underconfident: bool,
    /// Brier score (lower is better)
    pub brier_score: f64,
    /// Number of predictions used for calibration
    pub prediction_count: usize,
}

impl Default for CalibrationScore {
    fn default() -> Self {
        Self {
            score: 0.5,
            overconfident: false,
            underconfident: false,
            brier_score: 0.25,
            prediction_count: 0,
        }
    }
}

/// Configuration for uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// High uncertainty threshold
    pub high_uncertainty_threshold: f64,
    /// Number of predictions to track for calibration
    pub calibration_history_size: usize,
    /// Minimum predictions before trusting calibration
    pub min_calibration_predictions: usize,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            high_uncertainty_threshold: 0.5,
            calibration_history_size: 1000,
            min_calibration_predictions: 20,
        }
    }
}

/// The Uncertainty Quantifier - knows what it doesn't know
pub struct UncertaintyQuantifier {
    config: UncertaintyConfig,
    prediction_history: Vec<PredictionRecord>,
    #[allow(dead_code)]
    calibration_scores: Vec<CalibrationScore>,
    domain_uncertainties: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PredictionRecord {
    prediction: String,
    confidence: f64,
    outcome: Option<bool>,
    timestamp: TimePoint,
}

impl UncertaintyQuantifier {
    pub fn new(config: UncertaintyConfig) -> Self {
        Self {
            config,
            prediction_history: Vec::new(),
            calibration_scores: Vec::new(),
            domain_uncertainties: HashMap::new(),
        }
    }

    /// Estimate uncertainty for an action
    pub fn estimate(&mut self, action: &str, context: &ActionContext) -> UncertaintyEstimate {
        let epistemic = self.estimate_epistemic(action, context);
        let aleatoric = self.estimate_aleatoric(action, context);

        let total = (epistemic.level.powi(2) + aleatoric.level.powi(2)).sqrt();

        let confidence_interval = self.calculate_confidence_interval(total);
        let calibration = self.get_calibration_score();

        UncertaintyEstimate {
            total,
            epistemic,
            aleatoric,
            confidence_interval,
            calibration,
            timestamp: TimePoint::now(),
        }
    }

    fn estimate_epistemic(&self, action: &str, context: &ActionContext) -> EpistemicUncertainty {
        let mut sources = Vec::new();
        let mut level = 0.0;

        // Check for missing context
        if context.operator_id.is_none() {
            sources.push(UncertaintySource {
                source_type: UncertaintySourceType::MissingData,
                description: "Unknown operator".to_string(),
                contribution: 0.2,
            });
            level += 0.2;
        }

        // Check for lack of history
        if context.interaction_history.len() < 3 {
            sources.push(UncertaintySource {
                source_type: UncertaintySourceType::MissingData,
                description: "Limited interaction history".to_string(),
                contribution: 0.15,
            });
            level += 0.15;
        }

        // Check for ambiguous goal
        if context.stated_goal.is_none() {
            sources.push(UncertaintySource {
                source_type: UncertaintySourceType::Ambiguity,
                description: "No stated goal".to_string(),
                contribution: 0.25,
            });
            level += 0.25;
        }

        // Check domain familiarity
        let domain = self.extract_domain(action);
        let domain_uncertainty = self
            .domain_uncertainties
            .get(&domain)
            .copied()
            .unwrap_or(0.3);
        if domain_uncertainty > 0.2 {
            sources.push(UncertaintySource {
                source_type: UncertaintySourceType::OutOfDistribution,
                description: format!("Unfamiliar domain: {}", domain),
                contribution: domain_uncertainty,
            });
            level += domain_uncertainty;
        }

        // Check calibration history for overconfidence
        let calibration = self.get_calibration_score();
        if calibration.overconfident {
            level *= 1.2; // Adjust upward if historically overconfident
        }

        let reduction_actions = self.suggest_reduction_actions(&sources);

        EpistemicUncertainty {
            level: level.min(1.0),
            sources,
            reducible: true,
            reduction_actions,
        }
    }

    fn estimate_aleatoric(&self, _action: &str, context: &ActionContext) -> AleatoricUncertainty {
        let mut sources = Vec::new();
        let mut level: f64 = 0.1; // Baseline aleatoric uncertainty

        // Check for inherent unpredictability in goal
        if let Some(goal) = &context.stated_goal {
            let goal_lower = goal.to_lowercase();
            if goal_lower.contains("predict") || goal_lower.contains("future") {
                sources.push(UncertaintySource {
                    source_type: UncertaintySourceType::FutureUncertainty,
                    description: "Predicting future outcomes".to_string(),
                    contribution: 0.3,
                });
                level += 0.3;
            }
            if goal_lower.contains("random") || goal_lower.contains("chance") {
                sources.push(UncertaintySource {
                    source_type: UncertaintySourceType::Randomness,
                    description: "Inherent randomness in task".to_string(),
                    contribution: 0.4,
                });
                level += 0.4;
            }
        }

        AleatoricUncertainty {
            level: level.min(1.0),
            sources,
            irreducible: true,
        }
    }

    fn extract_domain(&self, action: &str) -> String {
        // Simple domain extraction
        let action_lower = action.to_lowercase();
        if action_lower.contains("code") || action_lower.contains("program") {
            "programming".to_string()
        } else if action_lower.contains("math") || action_lower.contains("calculate") {
            "mathematics".to_string()
        } else if action_lower.contains("write") || action_lower.contains("draft") {
            "writing".to_string()
        } else if action_lower.contains("analyze") || action_lower.contains("research") {
            "analysis".to_string()
        } else {
            "general".to_string()
        }
    }

    fn suggest_reduction_actions(&self, sources: &[UncertaintySource]) -> Vec<String> {
        let mut actions = Vec::new();

        for source in sources {
            match source.source_type {
                UncertaintySourceType::MissingData => {
                    actions.push("Request additional context from operator".to_string());
                }
                UncertaintySourceType::Ambiguity => {
                    actions.push("Ask clarifying questions".to_string());
                }
                UncertaintySourceType::OutOfDistribution => {
                    actions.push("Acknowledge limitations in this domain".to_string());
                }
                UncertaintySourceType::ConflictingInfo => {
                    actions.push("Request operator to resolve conflict".to_string());
                }
                _ => {}
            }
        }

        actions
    }

    fn calculate_confidence_interval(&self, uncertainty: f64) -> ConfidenceInterval {
        // Simple confidence interval based on uncertainty
        let width = uncertainty * 0.5;

        ConfidenceInterval {
            lower: (0.5 - width).max(0.0),
            upper: (0.5 + width).min(1.0),
            confidence_level: 0.95,
        }
    }

    fn get_calibration_score(&self) -> CalibrationScore {
        if self.prediction_history.len() < self.config.min_calibration_predictions {
            return CalibrationScore::default();
        }

        // Calculate calibration from prediction history
        let resolved: Vec<&PredictionRecord> = self
            .prediction_history
            .iter()
            .filter(|p| p.outcome.is_some())
            .collect();

        if resolved.is_empty() {
            return CalibrationScore::default();
        }

        // Brier score calculation
        let brier_score: f64 = resolved
            .iter()
            .map(|p| {
                let outcome = if p.outcome.unwrap() { 1.0 } else { 0.0 };
                (p.confidence - outcome).powi(2)
            })
            .sum::<f64>()
            / resolved.len() as f64;

        // Calibration: are we overconfident or underconfident?
        let avg_confidence: f64 =
            resolved.iter().map(|p| p.confidence).sum::<f64>() / resolved.len() as f64;
        let accuracy: f64 =
            resolved.iter().filter(|p| p.outcome.unwrap()).count() as f64 / resolved.len() as f64;

        let overconfident = avg_confidence > accuracy + 0.1;
        let underconfident = avg_confidence < accuracy - 0.1;

        CalibrationScore {
            score: 1.0 - brier_score,
            overconfident,
            underconfident,
            brier_score,
            prediction_count: resolved.len(),
        }
    }

    /// Calibrate with feedback
    pub fn calibrate(&mut self, feedback: &AlignmentFeedback) {
        // Add to prediction history
        if !self.prediction_history.is_empty() {
            if let Some(last) = self.prediction_history.last_mut() {
                last.outcome = Some(feedback.approved);
            }
        }

        // Trim history
        if self.prediction_history.len() > self.config.calibration_history_size {
            self.prediction_history.remove(0);
        }

        // Update domain uncertainties based on feedback
        // If we were wrong, increase uncertainty for that domain
        if !feedback.approved {
            // Increase general uncertainty since we don't track per-domain yet
            for (_, uncertainty) in self.domain_uncertainties.iter_mut() {
                *uncertainty = (*uncertainty + 0.1).min(1.0);
            }
        }
    }

    /// Record a prediction for calibration tracking
    pub fn record_prediction(&mut self, prediction: &str, confidence: f64) {
        self.prediction_history.push(PredictionRecord {
            prediction: prediction.to_string(),
            confidence,
            outcome: None,
            timestamp: TimePoint::now(),
        });

        if self.prediction_history.len() > self.config.calibration_history_size {
            self.prediction_history.remove(0);
        }
    }

    /// Check if uncertainty is too high to proceed
    pub fn is_too_uncertain(&self, estimate: &UncertaintyEstimate) -> bool {
        estimate.total > self.config.high_uncertainty_threshold
    }

    /// Get calibration report
    pub fn calibration_report(&self) -> CalibrationReport {
        let calibration = self.get_calibration_score();

        let recommendation = if calibration.overconfident {
            "Increase uncertainty estimates - historically overconfident".to_string()
        } else if calibration.underconfident {
            "May be too cautious - historically underconfident".to_string()
        } else {
            "Calibration is reasonable".to_string()
        };

        CalibrationReport {
            score: calibration,
            total_predictions: self.prediction_history.len(),
            resolved_predictions: self
                .prediction_history
                .iter()
                .filter(|p| p.outcome.is_some())
                .count(),
            recommendation,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CalibrationReport {
    pub score: CalibrationScore,
    pub total_predictions: usize,
    pub resolved_predictions: usize,
    pub recommendation: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_quantifier_creation() {
        let mut quantifier = UncertaintyQuantifier::new(UncertaintyConfig::default());
        let context = ActionContext::default();

        let estimate = quantifier.estimate("test action", &context);
        assert!(estimate.total >= 0.0);
        assert!(estimate.total <= 1.0);
    }

    #[test]
    fn test_epistemic_uncertainty() {
        let mut quantifier = UncertaintyQuantifier::new(UncertaintyConfig::default());
        let context = ActionContext {
            operator_id: None, // Missing operator should increase uncertainty
            ..Default::default()
        };

        let estimate = quantifier.estimate("test", &context);
        assert!(estimate.epistemic.level > 0.0);
    }
}
