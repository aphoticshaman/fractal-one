//! ═══════════════════════════════════════════════════════════════════════════════
//! CONDUCTOR — Meta-Level Orchestration
//! ═══════════════════════════════════════════════════════════════════════════════
//! The conductor doesn't vote. It synthesizes.
//! When agents disagree, it doesn't pick a winner - it generates tests.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::agent::{AgentConcern, AgentResponse, AgentType, ConcernCategory};
use super::consensus::ConsensusResult;

/// Configuration for the conductor
#[derive(Debug, Clone)]
pub struct ConductorConfig {
    /// Minimum confidence to include in synthesis
    pub min_confidence: f64,
    /// Weight multiplier for safety concerns
    pub safety_weight: f64,
    /// Weight multiplier for adversarial findings
    pub adversarial_weight: f64,
}

impl Default for ConductorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            safety_weight: 1.5,
            adversarial_weight: 1.2,
        }
    }
}

/// Result from ensemble synthesis
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    pub response: String,
    pub confidence: f64,
    pub contributing_agents: Vec<AgentType>,
    pub rejected_agents: Vec<AgentType>,
}

/// Resolution of a conflict between agents
#[derive(Debug, Clone)]
pub struct ConflictResolution {
    pub conflict_description: String,
    pub agents_involved: Vec<AgentType>,
    pub resolution_strategy: ResolutionStrategy,
    pub outcome: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResolutionStrategy {
    /// Accept the more cautious position
    DeferToCautious,
    /// Require additional evidence
    RequireEvidence,
    /// Escalate to human
    EscalateToHuman,
    /// Accept weighted average
    WeightedSynthesis,
    /// Run test to resolve
    TestToResolve,
}

/// Orchestrated response from the conductor
#[derive(Debug, Clone)]
pub struct OrchestratedResponse {
    pub response: String,
    pub confidence: f64,
    pub method: SynthesisMethod,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SynthesisMethod {
    /// All agents agreed
    Unanimous,
    /// Weighted by confidence and role
    WeightedSynthesis,
    /// Safety concerns dominated
    SafetyOverride,
    /// Adversarial concerns dominated
    AdversarialVeto,
    /// Conflict required resolution
    ConflictResolution,
}

/// State of the orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorState {
    pub total_orchestrations: usize,
    pub unanimous_count: usize,
    pub conflict_count: usize,
    pub safety_override_count: usize,
    pub adversarial_veto_count: usize,
    pub average_confidence: f64,
}

impl Default for OrchestratorState {
    fn default() -> Self {
        Self {
            total_orchestrations: 0,
            unanimous_count: 0,
            conflict_count: 0,
            safety_override_count: 0,
            adversarial_veto_count: 0,
            average_confidence: 0.0,
        }
    }
}

/// The Conductor - orchestrates the ensemble
pub struct Conductor {
    config: ConductorConfig,
    state: OrchestratorState,
    history: Vec<OrchestratedResponse>,
}

impl Conductor {
    pub fn new(config: ConductorConfig) -> Self {
        Self {
            config,
            state: OrchestratorState::default(),
            history: Vec::new(),
        }
    }

    /// Synthesize responses from all agents
    pub fn synthesize(
        &mut self,
        responses: &[&AgentResponse],
        consensus: &ConsensusResult,
        conflicts: &[ConflictResolution],
    ) -> (String, f64) {
        self.state.total_orchestrations += 1;

        // Check for critical safety concerns
        let safety_concerns: Vec<&AgentConcern> = responses
            .iter()
            .flat_map(|r| r.concerns.iter())
            .filter(|c| c.category == ConcernCategory::Safety && c.severity > 0.7)
            .collect();

        if !safety_concerns.is_empty() {
            self.state.safety_override_count += 1;
            let warning = safety_concerns
                .iter()
                .map(|c| c.description.as_str())
                .collect::<Vec<_>>()
                .join("; ");
            return (format!("SAFETY OVERRIDE: {}", warning), 0.3);
        }

        // Check for critical security concerns (adversarial)
        let security_concerns: Vec<&AgentConcern> = responses
            .iter()
            .filter(|r| r.agent_type == AgentType::Gamma)
            .flat_map(|r| r.concerns.iter())
            .filter(|c| c.category == ConcernCategory::Security && c.severity > 0.8)
            .collect();

        if !security_concerns.is_empty() {
            self.state.adversarial_veto_count += 1;
            let warning = security_concerns
                .iter()
                .map(|c| c.description.as_str())
                .collect::<Vec<_>>()
                .join("; ");
            return (format!("ADVERSARIAL VETO: {}", warning), 0.2);
        }

        // If conflicts were resolved, incorporate that
        if !conflicts.is_empty() {
            self.state.conflict_count += 1;
            return self.synthesize_with_conflicts(responses, conflicts);
        }

        // Check for unanimous agreement
        if consensus.agreement > 0.9 {
            self.state.unanimous_count += 1;
            return self.synthesize_unanimous(responses);
        }

        // Weighted synthesis
        self.synthesize_weighted(responses)
    }

    fn synthesize_unanimous(&self, responses: &[&AgentResponse]) -> (String, f64) {
        // All agents agree - use the most confident response
        if let Some(best) = responses.iter().max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            (best.recommendation.clone(), best.confidence)
        } else {
            ("No response available".to_string(), 0.0)
        }
    }

    fn synthesize_weighted(&self, responses: &[&AgentResponse]) -> (String, f64) {
        // Calculate weighted confidence
        let mut total_weight = 0.0;
        let mut weighted_confidence = 0.0;

        for response in responses {
            let weight = self.weight_for_agent(response.agent_type);
            if response.confidence >= self.config.min_confidence {
                total_weight += weight;
                weighted_confidence += weight * response.confidence;
            }
        }

        let final_confidence = if total_weight > 0.0 {
            weighted_confidence / total_weight
        } else {
            0.5
        };

        // Combine recommendations (simplified - would be more sophisticated)
        let recommendations: Vec<&str> = responses
            .iter()
            .filter(|r| r.confidence >= self.config.min_confidence)
            .map(|r| r.recommendation.as_str())
            .collect();

        let combined = if recommendations.is_empty() {
            "Insufficient confidence for recommendation".to_string()
        } else {
            format!(
                "Synthesized from {} perspectives: {}",
                recommendations.len(),
                recommendations[0] // Take the first as primary
            )
        };

        (combined, final_confidence)
    }

    fn synthesize_with_conflicts(
        &self,
        responses: &[&AgentResponse],
        conflicts: &[ConflictResolution],
    ) -> (String, f64) {
        // Factor in conflict resolutions
        let avg_resolution_confidence: f64 =
            conflicts.iter().map(|c| c.confidence).sum::<f64>() / conflicts.len() as f64;

        let resolutions: Vec<&str> = conflicts.iter().map(|c| c.outcome.as_str()).collect();

        let base_confidence =
            responses.iter().map(|r| r.confidence).sum::<f64>() / responses.len() as f64;

        let final_confidence = (base_confidence + avg_resolution_confidence) / 2.0;

        (
            format!("After conflict resolution: {}", resolutions.join("; ")),
            final_confidence,
        )
    }

    fn weight_for_agent(&self, agent_type: AgentType) -> f64 {
        match agent_type {
            AgentType::Alpha => 1.0,
            AgentType::Beta => self.config.safety_weight,
            AgentType::Gamma => self.config.adversarial_weight,
            AgentType::Delta => 1.0,
        }
    }

    /// Resolve conflicts between agents
    pub fn resolve_conflicts(&mut self, responses: &[&AgentResponse]) -> Vec<ConflictResolution> {
        let mut conflicts = Vec::new();

        // Find pairs of agents with significantly different confidences
        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                let diff = (responses[i].confidence - responses[j].confidence).abs();

                if diff > 0.3 {
                    // Significant disagreement - resolve it
                    let resolution = self.resolve_pair(responses[i], responses[j]);
                    conflicts.push(resolution);
                }
            }
        }

        conflicts
    }

    fn resolve_pair(&self, a: &AgentResponse, b: &AgentResponse) -> ConflictResolution {
        // Determine resolution strategy based on agent types
        let strategy = match (a.agent_type, b.agent_type) {
            (AgentType::Beta, _) | (_, AgentType::Beta) => ResolutionStrategy::DeferToCautious,
            (AgentType::Gamma, _) | (_, AgentType::Gamma) => ResolutionStrategy::RequireEvidence,
            (AgentType::Alpha, AgentType::Delta) | (AgentType::Delta, AgentType::Alpha) => {
                ResolutionStrategy::WeightedSynthesis
            }
            _ => ResolutionStrategy::WeightedSynthesis,
        };

        let outcome = match strategy {
            ResolutionStrategy::DeferToCautious => {
                if a.agent_type == AgentType::Beta {
                    a.recommendation.clone()
                } else {
                    b.recommendation.clone()
                }
            }
            ResolutionStrategy::RequireEvidence => {
                "Requires additional evidence before proceeding".to_string()
            }
            ResolutionStrategy::WeightedSynthesis => {
                if a.confidence > b.confidence {
                    a.recommendation.clone()
                } else {
                    b.recommendation.clone()
                }
            }
            _ => "Resolution pending".to_string(),
        };

        ConflictResolution {
            conflict_description: format!("{} vs {} disagree", a.agent_type, b.agent_type),
            agents_involved: vec![a.agent_type, b.agent_type],
            resolution_strategy: strategy,
            outcome,
            confidence: (a.confidence + b.confidence) / 2.0 * 0.8, // Reduce confidence due to conflict
        }
    }

    /// Get current state
    pub fn state(&self) -> OrchestratorState {
        let avg = if self.state.total_orchestrations > 0 {
            self.history.iter().map(|r| r.confidence).sum::<f64>()
                / self.state.total_orchestrations as f64
        } else {
            0.0
        };

        OrchestratorState {
            average_confidence: avg,
            ..self.state.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::agent::AgentPerspective;
    use super::*;
    use crate::time::TimePoint;

    fn make_response(agent_type: AgentType, confidence: f64) -> AgentResponse {
        AgentResponse {
            agent_type,
            recommendation: format!("{} recommendation", agent_type),
            confidence,
            concerns: Vec::new(),
            perspective: AgentPerspective {
                agent_type,
                analysis: "Test analysis".to_string(),
                key_considerations: Vec::new(),
                risks_identified: Vec::new(),
                opportunities_identified: Vec::new(),
                confidence,
            },
            timestamp: TimePoint::now(),
        }
    }

    #[test]
    fn test_conductor_creation() {
        let conductor = Conductor::new(ConductorConfig::default());
        assert_eq!(conductor.state.total_orchestrations, 0);
    }

    #[test]
    fn test_weighted_synthesis() {
        let mut conductor = Conductor::new(ConductorConfig::default());
        let alpha = make_response(AgentType::Alpha, 0.9);
        let beta = make_response(AgentType::Beta, 0.8);
        let consensus = ConsensusResult {
            agreement: 0.7,
            convergence_type: super::super::consensus::ConvergenceType::Partial,
            disagreement: 0.3,
            disagreements: Vec::new(),
        };

        let (response, confidence) = conductor.synthesize(&[&alpha, &beta], &consensus, &[]);
        assert!(confidence > 0.0);
        assert!(!response.is_empty());
    }
}
