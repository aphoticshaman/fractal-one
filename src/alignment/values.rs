//! ═══════════════════════════════════════════════════════════════════════════════
//! VALUE LEARNING — Infer, Don't Assume
//! ═══════════════════════════════════════════════════════════════════════════════
//! Don't hardcode values. Don't assume you know what the operator wants.
//! Learn from interaction, update with feedback, maintain uncertainty.
//!
//! Key insight: Trained-in values can be jailbroken. Inferred values from
//! ongoing interaction are harder to game because they update.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::ActionContext;
use super::AlignmentFeedback;
use crate::time::TimePoint;
use std::collections::HashMap;

/// A learned value
#[derive(Debug, Clone)]
pub struct LearnedValue {
    /// What the value is about
    pub domain: String,
    /// The value itself (e.g., "prefers privacy", "values efficiency")
    pub description: String,
    /// Confidence in this value
    pub confidence: ValueConfidence,
    /// Where this value was learned from
    pub source: ValueSource,
    /// When this was last updated
    pub last_updated: TimePoint,
    /// How stable this value has been
    pub stability: f64,
}

/// Confidence in a learned value
#[derive(Debug, Clone)]
pub struct ValueConfidence {
    /// Overall confidence (0-1)
    pub overall: f64,
    /// How many observations support this
    pub observation_count: usize,
    /// Consistency across observations
    pub consistency: f64,
    /// Recency-weighted confidence
    pub recency_weighted: f64,
}

impl Default for ValueConfidence {
    fn default() -> Self {
        Self {
            overall: 0.5,
            observation_count: 0,
            consistency: 0.5,
            recency_weighted: 0.5,
        }
    }
}

/// Source of a learned value
#[derive(Debug, Clone, PartialEq)]
pub enum ValueSource {
    /// Explicitly stated by operator
    ExplicitStatement,
    /// Inferred from choices
    InferredFromChoices,
    /// Inferred from corrections
    InferredFromCorrections,
    /// Default assumption (lowest confidence)
    DefaultAssumption,
    /// From organizational policy
    OrganizationalPolicy,
}

/// A conflict between values
#[derive(Debug, Clone)]
pub struct ValueConflict {
    pub value_a: String,
    pub value_b: String,
    pub conflict_type: ConflictType,
    pub severity: f64,
    pub suggested_resolution: Option<ValueResolution>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictType {
    /// Values directly contradict
    Contradiction,
    /// Values compete for resources
    ResourceCompetition,
    /// Values have different time horizons
    TemporalMismatch,
    /// Values apply to different scopes
    ScopeMismatch,
}

/// Resolution for a value conflict
#[derive(Debug, Clone)]
pub struct ValueResolution {
    pub strategy: ResolutionStrategy,
    pub explanation: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResolutionStrategy {
    /// Prioritize one over the other
    Prioritize,
    /// Find a compromise
    Compromise,
    /// Defer to operator
    DeferToOperator,
    /// Apply different values in different contexts
    ContextualSeparation,
}

/// Profile of learned values for an operator
#[derive(Debug, Clone)]
pub struct ValueProfile {
    pub operator_id: Option<String>,
    pub values: Vec<LearnedValue>,
    pub constraints: Vec<String>,
    pub overall_confidence: f64,
    pub last_updated: TimePoint,
}

impl Default for ValueProfile {
    fn default() -> Self {
        Self {
            operator_id: None,
            values: Vec::new(),
            constraints: Vec::new(),
            overall_confidence: 0.5,
            last_updated: TimePoint::now(),
        }
    }
}

/// Configuration for value learning
#[derive(Debug, Clone)]
pub struct ValueConfig {
    /// Minimum observations before high confidence
    pub min_observations: usize,
    /// Decay rate for old observations
    pub decay_rate: f64,
    /// Threshold for value stability
    pub stability_threshold: f64,
    /// Maximum values to track per operator
    pub max_values: usize,
}

impl Default for ValueConfig {
    fn default() -> Self {
        Self {
            min_observations: 5,
            decay_rate: 0.95,
            stability_threshold: 0.7,
            max_values: 100,
        }
    }
}

/// The Value Learner - infers what the operator actually wants
pub struct ValueLearner {
    #[allow(dead_code)]
    config: ValueConfig,
    profiles: HashMap<String, ValueProfile>,
    default_profile: ValueProfile,
    feedback_history: Vec<FeedbackRecord>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FeedbackRecord {
    action: String,
    feedback: AlignmentFeedback,
    timestamp: TimePoint,
    operator_id: Option<String>,
}

impl ValueLearner {
    pub fn new(config: ValueConfig) -> Self {
        Self {
            config,
            profiles: HashMap::new(),
            default_profile: Self::create_default_profile(),
            feedback_history: Vec::new(),
        }
    }

    fn create_default_profile() -> ValueProfile {
        // Default values that we assume until we learn otherwise
        let default_values = vec![
            LearnedValue {
                domain: "harm".to_string(),
                description: "Avoid causing harm to the operator or others".to_string(),
                confidence: ValueConfidence {
                    overall: 0.99,
                    observation_count: 0,
                    consistency: 1.0,
                    recency_weighted: 0.99,
                },
                source: ValueSource::DefaultAssumption,
                last_updated: TimePoint::now(),
                stability: 1.0,
            },
            LearnedValue {
                domain: "honesty".to_string(),
                description: "Be truthful and don't deceive".to_string(),
                confidence: ValueConfidence {
                    overall: 0.95,
                    observation_count: 0,
                    consistency: 1.0,
                    recency_weighted: 0.95,
                },
                source: ValueSource::DefaultAssumption,
                last_updated: TimePoint::now(),
                stability: 1.0,
            },
            LearnedValue {
                domain: "helpfulness".to_string(),
                description: "Try to be genuinely helpful".to_string(),
                confidence: ValueConfidence {
                    overall: 0.9,
                    observation_count: 0,
                    consistency: 1.0,
                    recency_weighted: 0.9,
                },
                source: ValueSource::DefaultAssumption,
                last_updated: TimePoint::now(),
                stability: 0.9,
            },
            LearnedValue {
                domain: "autonomy".to_string(),
                description: "Respect operator's right to make decisions".to_string(),
                confidence: ValueConfidence {
                    overall: 0.85,
                    observation_count: 0,
                    consistency: 1.0,
                    recency_weighted: 0.85,
                },
                source: ValueSource::DefaultAssumption,
                last_updated: TimePoint::now(),
                stability: 0.9,
            },
        ];

        ValueProfile {
            operator_id: None,
            values: default_values,
            constraints: Vec::new(),
            overall_confidence: 0.8,
            last_updated: TimePoint::now(),
        }
    }

    /// Infer values from context
    pub fn infer_values(&mut self, context: &ActionContext) -> ValueProfile {
        let operator_id = context
            .operator_id
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let operator_id_for_profile = operator_id.clone();

        // Get or create profile for this operator using entry API
        let profile = self.profiles.entry(operator_id).or_insert_with(|| {
            let mut p = self.default_profile.clone();
            p.operator_id = Some(operator_id_for_profile);
            p
        });

        // Update values from context
        Self::extract_values_from_context_static(context, profile);

        // Update confidence based on interaction history
        Self::update_confidence_static(profile);

        profile.last_updated = TimePoint::now();
        profile.clone()
    }

    fn extract_values_from_context_static(context: &ActionContext, profile: &mut ValueProfile) {
        let max_values = 100; // Default max values

        // Extract explicit constraints as values
        for constraint in &context.constraints {
            let existing = profile
                .values
                .iter_mut()
                .find(|v| v.domain == "constraint" && v.description == *constraint);

            if let Some(value) = existing {
                value.confidence.observation_count += 1;
                value.confidence.overall = (value.confidence.overall + 0.1).min(1.0);
                value.last_updated = TimePoint::now();
            } else if profile.values.len() < max_values {
                profile.values.push(LearnedValue {
                    domain: "constraint".to_string(),
                    description: constraint.clone(),
                    confidence: ValueConfidence {
                        overall: 0.8,
                        observation_count: 1,
                        consistency: 1.0,
                        recency_weighted: 0.8,
                    },
                    source: ValueSource::ExplicitStatement,
                    last_updated: TimePoint::now(),
                    stability: 0.9,
                });
            }
        }

        // Infer from stated goal if present
        if let Some(goal) = &context.stated_goal {
            Self::infer_values_from_goal_static(goal, profile, max_values);
        }

        // Infer from interaction patterns
        Self::infer_from_patterns_static(&context.interaction_history, profile, max_values);
    }

    fn infer_values_from_goal_static(goal: &str, profile: &mut ValueProfile, max_values: usize) {
        let goal_lower = goal.to_lowercase();

        // Simple keyword-based inference (would be more sophisticated in production)
        let inferences: Vec<(&str, &str, f64)> = vec![
            ("privacy", "Values privacy and data protection", 0.7),
            ("secure", "Values security", 0.7),
            ("fast", "Values speed and efficiency", 0.6),
            ("accurate", "Values accuracy over speed", 0.6),
            ("simple", "Prefers simple solutions", 0.5),
            ("thorough", "Prefers thorough analysis", 0.5),
        ];

        for (keyword, description, confidence) in inferences {
            if goal_lower.contains(keyword) {
                let existing = profile
                    .values
                    .iter_mut()
                    .find(|v| v.description == description);

                if let Some(value) = existing {
                    value.confidence.observation_count += 1;
                    value.confidence.overall = (value.confidence.overall + 0.1).min(1.0);
                } else if profile.values.len() < max_values {
                    profile.values.push(LearnedValue {
                        domain: "preference".to_string(),
                        description: description.to_string(),
                        confidence: ValueConfidence {
                            overall: confidence,
                            observation_count: 1,
                            consistency: 0.7,
                            recency_weighted: confidence,
                        },
                        source: ValueSource::InferredFromChoices,
                        last_updated: TimePoint::now(),
                        stability: 0.5,
                    });
                }
            }
        }
    }

    fn infer_from_patterns_static(
        history: &[String],
        profile: &mut ValueProfile,
        max_values: usize,
    ) {
        if history.len() < 3 {
            return;
        }

        // Look for patterns in interaction history
        let recent = history.iter().rev().take(10).collect::<Vec<_>>();

        // Count repeated themes (simplified)
        let mut theme_counts: HashMap<&str, usize> = HashMap::new();
        for interaction in recent {
            let lower = interaction.to_lowercase();
            if lower.contains("please") || lower.contains("thank") {
                *theme_counts.entry("politeness").or_insert(0) += 1;
            }
            if lower.contains("why") || lower.contains("explain") {
                *theme_counts.entry("understanding").or_insert(0) += 1;
            }
            if lower.contains("quick") || lower.contains("asap") {
                *theme_counts.entry("urgency").or_insert(0) += 1;
            }
        }

        // Convert patterns to values
        for (theme, count) in theme_counts {
            if count >= 2 {
                let description = match theme {
                    "politeness" => "Values courteous interaction",
                    "understanding" => "Values explanations and transparency",
                    "urgency" => "Often working under time pressure",
                    _ => continue,
                };

                let existing = profile
                    .values
                    .iter_mut()
                    .find(|v| v.description == description);

                if let Some(value) = existing {
                    value.confidence.observation_count += count;
                } else if profile.values.len() < max_values {
                    profile.values.push(LearnedValue {
                        domain: "interaction_style".to_string(),
                        description: description.to_string(),
                        confidence: ValueConfidence {
                            overall: 0.5,
                            observation_count: count,
                            consistency: 0.6,
                            recency_weighted: 0.5,
                        },
                        source: ValueSource::InferredFromChoices,
                        last_updated: TimePoint::now(),
                        stability: 0.3,
                    });
                }
            }
        }
    }

    fn update_confidence_static(profile: &mut ValueProfile) {
        // Update confidence based on observation count and consistency
        let min_observations = 3; // Default
        let decay_rate = 0.95; // Default

        for value in &mut profile.values {
            let obs_factor =
                (value.confidence.observation_count as f64 / min_observations as f64).min(1.0);
            value.confidence.overall = value.confidence.overall * 0.9 + obs_factor * 0.1;

            // Apply decay
            value.confidence.recency_weighted *= decay_rate;
        }

        // Update overall profile confidence
        if !profile.values.is_empty() {
            profile.overall_confidence = profile
                .values
                .iter()
                .map(|v| v.confidence.overall)
                .sum::<f64>()
                / profile.values.len() as f64;
        }
    }

    /// Check if an action aligns with the value profile
    pub fn check_alignment(&self, action: &str, profile: &ValueProfile) -> f64 {
        let action_lower = action.to_lowercase();

        // Check against each value
        let mut alignment_score = 1.0;

        for value in &profile.values {
            let value_alignment = self.action_value_alignment(&action_lower, value);
            alignment_score *= value_alignment;
        }

        alignment_score.max(0.0)
    }

    fn action_value_alignment(&self, action: &str, value: &LearnedValue) -> f64 {
        // Check if action conflicts with value
        let conflicts = match value.domain.as_str() {
            "harm" => {
                action.contains("harm") || action.contains("hurt") || action.contains("damage")
            }
            "honesty" => {
                action.contains("lie") || action.contains("deceive") || action.contains("mislead")
            }
            _ => false,
        };

        if conflicts {
            // Action conflicts with value
            1.0 - value.confidence.overall
        } else {
            // Assume neutral alignment
            1.0
        }
    }

    /// Find conflicts in values when considering an action
    pub fn find_conflicts(&self, action: &str, profile: &ValueProfile) -> Vec<ValueConflict> {
        let mut conflicts = Vec::new();

        // Check pairs of values for conflicts
        for i in 0..profile.values.len() {
            for j in (i + 1)..profile.values.len() {
                if let Some(conflict) =
                    self.check_value_pair_conflict(&profile.values[i], &profile.values[j], action)
                {
                    conflicts.push(conflict);
                }
            }
        }

        conflicts
    }

    fn check_value_pair_conflict(
        &self,
        a: &LearnedValue,
        b: &LearnedValue,
        _action: &str,
    ) -> Option<ValueConflict> {
        // Check for known conflict patterns
        let conflicts_with = |domain_a: &str, domain_b: &str| -> bool {
            (domain_a == "urgency" && domain_b == "thorough")
                || (domain_a == "privacy" && domain_b == "transparency")
                || (domain_a == "efficiency" && domain_b == "safety")
        };

        if conflicts_with(&a.domain, &b.domain) || conflicts_with(&b.domain, &a.domain) {
            Some(ValueConflict {
                value_a: a.description.clone(),
                value_b: b.description.clone(),
                conflict_type: ConflictType::ResourceCompetition,
                severity: 0.5,
                suggested_resolution: Some(ValueResolution {
                    strategy: ResolutionStrategy::DeferToOperator,
                    explanation: "Values may conflict; operator should clarify priority"
                        .to_string(),
                    confidence: 0.7,
                }),
            })
        } else {
            None
        }
    }

    /// Incorporate feedback to improve value learning
    pub fn incorporate_feedback(&mut self, action: &str, feedback: &AlignmentFeedback) {
        self.feedback_history.push(FeedbackRecord {
            action: action.to_string(),
            feedback: feedback.clone(),
            timestamp: TimePoint::now(),
            operator_id: None, // Would need context to set this
        });

        // If rejected, learn from the rejection
        if !feedback.approved {
            if let Some(reason) = &feedback.rejection_reason {
                // Add as a constraint/value
                self.default_profile.values.push(LearnedValue {
                    domain: "learned_constraint".to_string(),
                    description: format!("Avoid: {}", reason),
                    confidence: ValueConfidence {
                        overall: 0.8,
                        observation_count: 1,
                        consistency: 1.0,
                        recency_weighted: 0.8,
                    },
                    source: ValueSource::InferredFromCorrections,
                    last_updated: TimePoint::now(),
                    stability: 0.7,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_learner_creation() {
        let learner = ValueLearner::new(ValueConfig::default());
        assert!(!learner.default_profile.values.is_empty());
    }

    #[test]
    fn test_value_inference() {
        let mut learner = ValueLearner::new(ValueConfig::default());
        let context = ActionContext {
            constraints: vec!["keep it private".to_string()],
            ..Default::default()
        };

        let profile = learner.infer_values(&context);
        assert!(profile.values.iter().any(|v| v.domain == "constraint"));
    }
}
