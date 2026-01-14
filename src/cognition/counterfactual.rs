//! ═══════════════════════════════════════════════════════════════════════════════
//! COUNTERFACTUAL REASONING — "What If?" Thinking
//! ═══════════════════════════════════════════════════════════════════════════════
//! The ability to reason about what didn't happen but could have.
//! This is where intelligence meets imagination - constrained by logic.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::pattern::PatternMatch;
use super::CognitionInput;

/// An intervention in a causal model
#[derive(Debug, Clone)]
pub struct Intervention {
    /// What variable we're intervening on
    pub variable: String,
    /// What value we're setting it to
    pub new_value: String,
    /// What the original value was
    pub original_value: Option<String>,
    /// Type of intervention
    pub intervention_type: InterventionType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterventionType {
    /// Set variable to specific value (do(X=x))
    Set,
    /// Remove variable from system
    Remove,
    /// Observe (not intervene, for comparison)
    Observe,
    /// Constrain to range
    Constrain,
}

/// A causal effect
#[derive(Debug, Clone)]
pub struct CausalEffect {
    /// What caused the effect
    pub cause: String,
    /// What was affected
    pub effect: String,
    /// Strength of effect (-1 to 1, negative = inverse)
    pub strength: f64,
    /// Is this direct or mediated?
    pub direct: bool,
    /// Mediating variables (if any)
    pub mediators: Vec<String>,
    /// Confidence in this effect
    pub confidence: f64,
}

/// A counterfactual scenario
#[derive(Debug, Clone)]
pub struct Counterfactual {
    /// Description of the counterfactual
    pub description: String,
    /// The intervention(s) involved
    pub interventions: Vec<Intervention>,
    /// Predicted effects
    pub predicted_effects: Vec<CausalEffect>,
    /// Assumptions made
    pub assumptions: Vec<String>,
}

/// Result of counterfactual analysis
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    pub counterfactual: Counterfactual,
    /// Is this counterfactual logically valid?
    pub is_valid: bool,
    /// Description of the outcome
    pub description: String,
    /// Does it contradict observations?
    pub contradicts_observations: bool,
    /// How plausible is this scenario? (0-1)
    pub plausibility: f64,
    /// What we learned from considering this
    pub insights: Vec<String>,
    /// Related counterfactuals to explore
    pub related: Vec<String>,
}

/// Configuration for counterfactual engine
#[derive(Debug, Clone)]
pub struct CounterfactualConfig {
    /// Maximum interventions per counterfactual
    pub max_interventions: usize,
    /// Minimum plausibility to report
    pub min_plausibility: f64,
    /// Enable speculative reasoning
    pub speculative: bool,
    /// How far to trace causal chains
    pub max_chain_length: usize,
}

impl Default for CounterfactualConfig {
    fn default() -> Self {
        Self {
            max_interventions: 3,
            min_plausibility: 0.2,
            speculative: true,
            max_chain_length: 5,
        }
    }
}

/// The Counterfactual Engine
pub struct CounterfactualEngine {
    config: CounterfactualConfig,
    causal_knowledge: Vec<CausalEffect>,
    explored_counterfactuals: Vec<CounterfactualResult>,
}

impl CounterfactualEngine {
    pub fn new(config: CounterfactualConfig) -> Self {
        Self {
            config,
            causal_knowledge: Self::default_causal_knowledge(),
            explored_counterfactuals: Vec::new(),
        }
    }

    fn default_causal_knowledge() -> Vec<CausalEffect> {
        vec![
            // Basic causal knowledge
            CausalEffect {
                cause: "complexity".to_string(),
                effect: "errors".to_string(),
                strength: 0.7,
                direct: true,
                mediators: vec![],
                confidence: 0.8,
            },
            CausalEffect {
                cause: "testing".to_string(),
                effect: "errors".to_string(),
                strength: -0.6,
                direct: true,
                mediators: vec![],
                confidence: 0.7,
            },
            CausalEffect {
                cause: "time_pressure".to_string(),
                effect: "errors".to_string(),
                strength: 0.5,
                direct: false,
                mediators: vec!["shortcuts".to_string()],
                confidence: 0.6,
            },
            CausalEffect {
                cause: "experience".to_string(),
                effect: "quality".to_string(),
                strength: 0.6,
                direct: true,
                mediators: vec![],
                confidence: 0.7,
            },
        ]
    }

    /// Generate a counterfactual from a pattern
    pub fn generate(
        &mut self,
        pattern: &PatternMatch,
        input: &CognitionInput,
    ) -> Option<CounterfactualResult> {
        // Extract potential intervention points from the pattern
        let interventions = self.extract_interventions(pattern);

        if interventions.is_empty() {
            return None;
        }

        // Build the counterfactual
        let counterfactual = Counterfactual {
            description: format!("What if {} were different?", pattern.pattern.description),
            interventions: interventions.clone(),
            predicted_effects: self.predict_effects(&interventions),
            assumptions: self.identify_assumptions(&interventions, input),
        };

        // Evaluate the counterfactual
        let result = self.evaluate_counterfactual(counterfactual);

        // Archive if interesting
        if result.plausibility >= self.config.min_plausibility {
            self.explored_counterfactuals.push(result.clone());
        }

        Some(result)
    }

    fn extract_interventions(&self, pattern: &PatternMatch) -> Vec<Intervention> {
        let mut interventions = Vec::new();

        // Create intervention based on pattern type
        match pattern.pattern.pattern_type {
            super::pattern::PatternType::Causal => {
                // For causal patterns, intervene on the cause
                if let Some(evidence) = pattern.evidence.first() {
                    interventions.push(Intervention {
                        variable: evidence.clone(),
                        new_value: "absent".to_string(),
                        original_value: Some("present".to_string()),
                        intervention_type: InterventionType::Set,
                    });
                }
            }
            super::pattern::PatternType::Temporal => {
                // For temporal patterns, intervene on ordering
                if let Some(evidence) = pattern.evidence.first() {
                    interventions.push(Intervention {
                        variable: "sequence".to_string(),
                        new_value: "reversed".to_string(),
                        original_value: Some(evidence.clone()),
                        intervention_type: InterventionType::Set,
                    });
                }
            }
            super::pattern::PatternType::Structural => {
                // For conditionals, negate the condition
                if let Some(evidence) = pattern.evidence.first() {
                    interventions.push(Intervention {
                        variable: evidence.clone(),
                        new_value: "false".to_string(),
                        original_value: Some("true".to_string()),
                        intervention_type: InterventionType::Set,
                    });
                }
            }
            _ => {
                // Generic intervention
                interventions.push(Intervention {
                    variable: pattern.pattern.id.clone(),
                    new_value: "altered".to_string(),
                    original_value: None,
                    intervention_type: InterventionType::Set,
                });
            }
        }

        interventions.truncate(self.config.max_interventions);
        interventions
    }

    fn predict_effects(&self, interventions: &[Intervention]) -> Vec<CausalEffect> {
        let mut effects = Vec::new();

        for intervention in interventions {
            // Find causal knowledge relevant to this intervention
            for knowledge in &self.causal_knowledge {
                if knowledge.cause.contains(&intervention.variable)
                    || intervention.variable.contains(&knowledge.cause)
                {
                    // Predict the effect
                    let modified_strength =
                        if intervention.intervention_type == InterventionType::Remove {
                            0.0
                        } else if intervention.new_value == "absent"
                            || intervention.new_value == "false"
                        {
                            -knowledge.strength
                        } else {
                            knowledge.strength
                        };

                    effects.push(CausalEffect {
                        cause: intervention.variable.clone(),
                        effect: knowledge.effect.clone(),
                        strength: modified_strength,
                        direct: knowledge.direct,
                        mediators: knowledge.mediators.clone(),
                        confidence: knowledge.confidence * 0.8, // Reduce confidence for predictions
                    });
                }
            }
        }

        effects
    }

    fn identify_assumptions(
        &self,
        interventions: &[Intervention],
        _input: &CognitionInput,
    ) -> Vec<String> {
        let mut assumptions =
            vec!["Causal relationships remain stable under intervention".to_string()];

        for intervention in interventions {
            match intervention.intervention_type {
                InterventionType::Set => {
                    assumptions.push(format!(
                        "Setting {} to {} is physically/logically possible",
                        intervention.variable, intervention.new_value
                    ));
                }
                InterventionType::Remove => {
                    assumptions.push(format!(
                        "The system can function without {}",
                        intervention.variable
                    ));
                }
                _ => {}
            }
        }

        assumptions
    }

    fn evaluate_counterfactual(&self, counterfactual: Counterfactual) -> CounterfactualResult {
        // Check validity
        let is_valid = self.check_validity(&counterfactual);

        // Check for contradictions
        let contradicts = self.check_contradictions(&counterfactual);

        // Calculate plausibility
        let plausibility = self.calculate_plausibility(&counterfactual, is_valid, contradicts);

        // Extract insights
        let insights = self.extract_insights(&counterfactual);

        // Find related counterfactuals
        let related = self.find_related(&counterfactual);

        let description = if is_valid && !contradicts {
            format!(
                "If {}, then {} (plausibility: {:.0}%)",
                counterfactual
                    .interventions
                    .first()
                    .map(|i| format!("{} = {}", i.variable, i.new_value))
                    .unwrap_or_default(),
                counterfactual
                    .predicted_effects
                    .first()
                    .map(|e| format!(
                        "{} would {}",
                        e.effect,
                        if e.strength > 0.0 {
                            "increase"
                        } else {
                            "decrease"
                        }
                    ))
                    .unwrap_or_else(|| "unknown effect".to_string()),
                plausibility * 100.0
            )
        } else if contradicts {
            "This counterfactual contradicts observations".to_string()
        } else {
            "This counterfactual is logically invalid".to_string()
        };

        CounterfactualResult {
            counterfactual,
            is_valid,
            description,
            contradicts_observations: contradicts,
            plausibility,
            insights,
            related,
        }
    }

    fn check_validity(&self, counterfactual: &Counterfactual) -> bool {
        // Check logical consistency
        for intervention in &counterfactual.interventions {
            // Can't set and remove simultaneously
            let conflicting = counterfactual
                .interventions
                .iter()
                .filter(|i| i.variable == intervention.variable)
                .count()
                > 1;

            if conflicting {
                return false;
            }
        }

        // Check for circular causation in effects
        for effect in &counterfactual.predicted_effects {
            if effect.cause == effect.effect {
                return false;
            }
        }

        true
    }

    fn check_contradictions(&self, counterfactual: &Counterfactual) -> bool {
        // Check if predicted effects contradict known facts
        for effect in &counterfactual.predicted_effects {
            // Check against causal knowledge
            for knowledge in &self.causal_knowledge {
                // If we predict opposite of established knowledge with high confidence
                if knowledge.effect == effect.effect
                    && knowledge.confidence > 0.8
                    && (effect.strength > 0.0) != (knowledge.strength > 0.0)
                {
                    return true;
                }
            }
        }

        false
    }

    fn calculate_plausibility(
        &self,
        counterfactual: &Counterfactual,
        valid: bool,
        contradicts: bool,
    ) -> f64 {
        if !valid || contradicts {
            return 0.0;
        }

        let mut plausibility = 0.5; // Base plausibility

        // Adjust based on intervention types
        for intervention in &counterfactual.interventions {
            match intervention.intervention_type {
                InterventionType::Set => plausibility += 0.1,
                InterventionType::Remove => plausibility -= 0.1,
                InterventionType::Observe => plausibility += 0.2,
                InterventionType::Constrain => plausibility += 0.05,
            }
        }

        // Adjust based on effect confidence
        let avg_confidence: f64 = if !counterfactual.predicted_effects.is_empty() {
            counterfactual
                .predicted_effects
                .iter()
                .map(|e| e.confidence)
                .sum::<f64>()
                / counterfactual.predicted_effects.len() as f64
        } else {
            0.5
        };

        plausibility *= avg_confidence;

        // Penalize for many assumptions
        plausibility -= counterfactual.assumptions.len() as f64 * 0.05;

        plausibility.clamp(0.0, 1.0)
    }

    fn extract_insights(&self, counterfactual: &Counterfactual) -> Vec<String> {
        let mut insights = Vec::new();

        // Insight from causal structure
        if !counterfactual.predicted_effects.is_empty() {
            let direct_effects = counterfactual
                .predicted_effects
                .iter()
                .filter(|e| e.direct)
                .count();
            let indirect_effects = counterfactual.predicted_effects.len() - direct_effects;

            if indirect_effects > direct_effects {
                insights.push("Most effects are mediated rather than direct".to_string());
            }
        }

        // Insight from intervention type
        for intervention in &counterfactual.interventions {
            if intervention.intervention_type == InterventionType::Remove {
                insights.push(format!(
                    "{} may be a critical component",
                    intervention.variable
                ));
            }
        }

        // Insight from assumptions
        if counterfactual.assumptions.len() > 2 {
            insights.push(
                "This counterfactual requires many assumptions - interpret cautiously".to_string(),
            );
        }

        insights
    }

    fn find_related(&self, counterfactual: &Counterfactual) -> Vec<String> {
        let mut related = Vec::new();

        for intervention in &counterfactual.interventions {
            // Suggest opposite intervention
            let opposite = match intervention.intervention_type {
                InterventionType::Set => {
                    format!("What if {} were removed entirely?", intervention.variable)
                }
                InterventionType::Remove => {
                    format!("What if {} were strengthened?", intervention.variable)
                }
                _ => continue,
            };
            related.push(opposite);

            // Suggest related variables
            for knowledge in &self.causal_knowledge {
                if knowledge.effect == intervention.variable && !related.contains(&knowledge.cause)
                {
                    related.push(format!(
                        "What if {} were different instead?",
                        knowledge.cause
                    ));
                }
            }
        }

        related.truncate(3); // Limit suggestions
        related
    }

    /// Analyze causal relations for counterfactuals
    pub fn analyze_causal(
        &mut self,
        relations: &[super::CausalRelation],
    ) -> Vec<CounterfactualResult> {
        let mut results = Vec::new();

        for relation in relations {
            // Build counterfactual: "What if cause were absent?"
            let intervention = Intervention {
                variable: relation.cause.clone(),
                new_value: "absent".to_string(),
                original_value: Some("present".to_string()),
                intervention_type: InterventionType::Remove,
            };

            let predicted_effect = CausalEffect {
                cause: relation.cause.clone(),
                effect: relation.effect.clone(),
                strength: -relation.strength,
                direct: true,
                mediators: vec![],
                confidence: relation.strength,
            };

            let counterfactual = Counterfactual {
                description: format!("What if '{}' were absent?", relation.cause),
                interventions: vec![intervention],
                predicted_effects: vec![predicted_effect],
                assumptions: vec![format!(
                    "'{}' is a necessary cause of '{}'",
                    relation.cause, relation.effect
                )],
            };

            let result = self.evaluate_counterfactual(counterfactual);
            if result.plausibility >= self.config.min_plausibility {
                results.push(result);
            }
        }

        results
    }

    /// Add causal knowledge
    pub fn add_causal_knowledge(&mut self, effect: CausalEffect) {
        self.causal_knowledge.push(effect);
    }

    /// Get explored counterfactuals
    pub fn history(&self) -> &[CounterfactualResult] {
        &self.explored_counterfactuals
    }

    /// Get statistics
    pub fn statistics(&self) -> CounterfactualStatistics {
        let total = self.explored_counterfactuals.len();
        let valid = self
            .explored_counterfactuals
            .iter()
            .filter(|cf| cf.is_valid)
            .count();
        let avg_plausibility = if total > 0 {
            self.explored_counterfactuals
                .iter()
                .map(|cf| cf.plausibility)
                .sum::<f64>()
                / total as f64
        } else {
            0.0
        };

        CounterfactualStatistics {
            total_explored: total,
            valid_counterfactuals: valid,
            average_plausibility: avg_plausibility,
            causal_facts_known: self.causal_knowledge.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CounterfactualStatistics {
    pub total_explored: usize,
    pub valid_counterfactuals: usize,
    pub average_plausibility: f64,
    pub causal_facts_known: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counterfactual_engine_creation() {
        let engine = CounterfactualEngine::new(CounterfactualConfig::default());
        assert!(!engine.causal_knowledge.is_empty());
    }

    #[test]
    fn test_causal_analysis() {
        let mut engine = CounterfactualEngine::new(CounterfactualConfig::default());

        let relations = vec![super::super::CausalRelation {
            cause: "training".to_string(),
            effect: "performance".to_string(),
            strength: 0.8,
            mechanism: Some("skill acquisition".to_string()),
        }];

        let results = engine.analyze_causal(&relations);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_invalid_counterfactual() {
        let engine = CounterfactualEngine::new(CounterfactualConfig::default());

        // Circular causation should be invalid
        let counterfactual = Counterfactual {
            description: "Invalid circular".to_string(),
            interventions: vec![],
            predicted_effects: vec![CausalEffect {
                cause: "X".to_string(),
                effect: "X".to_string(), // Circular!
                strength: 1.0,
                direct: true,
                mediators: vec![],
                confidence: 1.0,
            }],
            assumptions: vec![],
        };

        let result = engine.evaluate_counterfactual(counterfactual);
        assert!(!result.is_valid);
    }
}
