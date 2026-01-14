//! ═══════════════════════════════════════════════════════════════════════════════
//! CONSENSUS ENGINE — Agreement Analysis
//! ═══════════════════════════════════════════════════════════════════════════════
//! Key insight: Convergence despite different priors = high confidence.
//! Disagreement triggers tests, not votes.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::agent::{AgentResponse, AgentType};
use super::{TestPriority, TestType};

/// Result of consensus analysis
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Overall agreement level (0-1)
    pub agreement: f64,
    /// Type of convergence
    pub convergence_type: ConvergenceType,
    /// Level of disagreement (0-1)
    pub disagreement: f64,
    /// Detailed disagreement analysis
    pub disagreements: Vec<DisagreementAnalysis>,
}

/// Type of convergence in the ensemble
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceType {
    /// All agents substantially agree
    Unanimous,
    /// Most agents agree, minor dissent
    StrongMajority,
    /// Partial agreement with significant dissent
    Partial,
    /// Agents substantially disagree
    Divergent,
    /// Cannot determine (insufficient data)
    Indeterminate,
}

/// Analysis of a disagreement
#[derive(Debug, Clone)]
pub struct DisagreementAnalysis {
    pub agents: Vec<AgentType>,
    pub description: String,
    pub severity: f64,
    pub likely_cause: DisagreementCause,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DisagreementCause {
    /// Different optimization targets
    DifferentObjectives,
    /// Different risk assessments
    RiskAssessment,
    /// Different information
    InformationAsymmetry,
    /// Fundamental value conflict
    ValueConflict,
    /// Uncertainty
    HighUncertainty,
    /// Unknown
    Unknown,
}

/// Test triggered by disagreement
#[derive(Debug, Clone)]
pub struct TestTrigger {
    pub test_type: TestType,
    pub description: String,
    pub agents_involved: Vec<AgentType>,
    pub priority: TestPriority,
}

/// Configuration for consensus engine
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Threshold for unanimous classification
    pub unanimous_threshold: f64,
    /// Threshold for strong majority classification
    pub majority_threshold: f64,
    /// Confidence difference threshold for disagreement
    pub disagreement_threshold: f64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            unanimous_threshold: 0.9,
            majority_threshold: 0.7,
            disagreement_threshold: 0.3,
        }
    }
}

/// The Consensus Engine - analyzes agreement patterns
pub struct ConsensusEngine {
    config: ConsensusConfig,
}

impl ConsensusEngine {
    pub fn new(config: ConsensusConfig) -> Self {
        Self { config }
    }

    /// Evaluate consensus among agent responses
    pub fn evaluate(&self, responses: &[&AgentResponse]) -> ConsensusResult {
        if responses.is_empty() {
            return ConsensusResult {
                agreement: 0.0,
                convergence_type: ConvergenceType::Indeterminate,
                disagreement: 1.0,
                disagreements: Vec::new(),
            };
        }

        // Calculate agreement based on confidence spread
        let confidences: Vec<f64> = responses.iter().map(|r| r.confidence).collect();
        let agreement = self.calculate_agreement(&confidences);
        let disagreement = 1.0 - agreement;

        // Classify convergence type
        let convergence_type = if agreement >= self.config.unanimous_threshold {
            ConvergenceType::Unanimous
        } else if agreement >= self.config.majority_threshold {
            ConvergenceType::StrongMajority
        } else if agreement >= 0.5 {
            ConvergenceType::Partial
        } else {
            ConvergenceType::Divergent
        };

        // Find specific disagreements
        let disagreements = self.analyze_disagreements(responses);

        ConsensusResult {
            agreement,
            convergence_type,
            disagreement,
            disagreements,
        }
    }

    fn calculate_agreement(&self, confidences: &[f64]) -> f64 {
        if confidences.len() < 2 {
            return 1.0;
        }

        // Calculate variance in confidences
        let mean: f64 = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let variance: f64 =
            confidences.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / confidences.len() as f64;

        // Lower variance = higher agreement
        // Max possible variance is 0.25 (when confidences are 0 and 1)
        let normalized_variance = (variance / 0.25).min(1.0);

        1.0 - normalized_variance.sqrt()
    }

    fn analyze_disagreements(&self, responses: &[&AgentResponse]) -> Vec<DisagreementAnalysis> {
        let mut disagreements = Vec::new();

        // Compare all pairs
        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                let diff = (responses[i].confidence - responses[j].confidence).abs();

                if diff > self.config.disagreement_threshold {
                    let analysis = self.analyze_pair(responses[i], responses[j], diff);
                    disagreements.push(analysis);
                }
            }
        }

        disagreements
    }

    fn analyze_pair(
        &self,
        a: &AgentResponse,
        b: &AgentResponse,
        diff: f64,
    ) -> DisagreementAnalysis {
        // Determine likely cause based on agent types
        let cause = match (a.agent_type, b.agent_type) {
            (AgentType::Alpha, AgentType::Beta) | (AgentType::Beta, AgentType::Alpha) => {
                DisagreementCause::DifferentObjectives
            }
            (AgentType::Gamma, _) | (_, AgentType::Gamma) => DisagreementCause::RiskAssessment,
            (AgentType::Beta, AgentType::Beta) => DisagreementCause::InformationAsymmetry,
            _ => {
                if diff > 0.5 {
                    DisagreementCause::HighUncertainty
                } else {
                    DisagreementCause::Unknown
                }
            }
        };

        let description = match cause {
            DisagreementCause::DifferentObjectives => {
                format!(
                    "{} optimizes for capability while {} optimizes for safety",
                    a.agent_type, b.agent_type
                )
            }
            DisagreementCause::RiskAssessment => {
                "Adversarial agent sees risks others don't".to_string()
            }
            DisagreementCause::InformationAsymmetry => {
                "Agents have different information about the situation".to_string()
            }
            DisagreementCause::ValueConflict => {
                "Fundamental disagreement about values or priorities".to_string()
            }
            DisagreementCause::HighUncertainty => {
                "High uncertainty leading to divergent assessments".to_string()
            }
            DisagreementCause::Unknown => {
                format!(
                    "{} and {} disagree for unclear reasons",
                    a.agent_type, b.agent_type
                )
            }
        };

        DisagreementAnalysis {
            agents: vec![a.agent_type, b.agent_type],
            description,
            severity: diff,
            likely_cause: cause,
        }
    }

    /// Check if disagreement should trigger a test
    pub fn should_trigger_test(&self, consensus: &ConsensusResult) -> Vec<TestTrigger> {
        let mut tests = Vec::new();

        if consensus.convergence_type == ConvergenceType::Divergent {
            tests.push(TestTrigger {
                test_type: TestType::EmpiricalValidation,
                description: "High divergence requires empirical validation".to_string(),
                agents_involved: vec![],
                priority: TestPriority::High,
            });
        }

        for disagreement in &consensus.disagreements {
            if disagreement.severity > 0.5 {
                let test_type = match disagreement.likely_cause {
                    DisagreementCause::RiskAssessment => TestType::AdversarialProbe,
                    DisagreementCause::DifferentObjectives => TestType::HumanReview,
                    _ => TestType::EmpiricalValidation,
                };

                tests.push(TestTrigger {
                    test_type,
                    description: format!("Test to resolve: {}", disagreement.description),
                    agents_involved: disagreement.agents.clone(),
                    priority: if disagreement.severity > 0.7 {
                        TestPriority::High
                    } else {
                        TestPriority::Normal
                    },
                });
            }
        }

        tests
    }

    /// Generate test suggestions based on disagreement type
    pub fn suggest_tests(&self, disagreement: &DisagreementAnalysis) -> Vec<String> {
        match disagreement.likely_cause {
            DisagreementCause::DifferentObjectives => {
                vec![
                    "Have operator clarify priority between capability and safety".to_string(),
                    "Run both approaches and compare outcomes".to_string(),
                ]
            }
            DisagreementCause::RiskAssessment => {
                vec![
                    "Run targeted security testing".to_string(),
                    "Consult external security review".to_string(),
                ]
            }
            DisagreementCause::InformationAsymmetry => {
                vec![
                    "Gather additional context".to_string(),
                    "Request clarification from operator".to_string(),
                ]
            }
            DisagreementCause::ValueConflict => {
                vec![
                    "Escalate to human decision maker".to_string(),
                    "Document conflict for policy review".to_string(),
                ]
            }
            DisagreementCause::HighUncertainty => {
                vec![
                    "Gather more data before deciding".to_string(),
                    "Use conservative default".to_string(),
                ]
            }
            DisagreementCause::Unknown => {
                vec!["Investigate root cause of disagreement".to_string()]
            }
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
            recommendation: "Test".to_string(),
            confidence,
            concerns: Vec::new(),
            perspective: AgentPerspective {
                agent_type,
                analysis: "Test".to_string(),
                key_considerations: Vec::new(),
                risks_identified: Vec::new(),
                opportunities_identified: Vec::new(),
                confidence,
            },
            timestamp: TimePoint::now(),
        }
    }

    #[test]
    fn test_unanimous_consensus() {
        let engine = ConsensusEngine::new(ConsensusConfig::default());

        let alpha = make_response(AgentType::Alpha, 0.9);
        let beta = make_response(AgentType::Beta, 0.85);
        let delta = make_response(AgentType::Delta, 0.88);

        let result = engine.evaluate(&[&alpha, &beta, &delta]);
        assert!(result.agreement > 0.7);
        assert!(
            result.convergence_type == ConvergenceType::Unanimous
                || result.convergence_type == ConvergenceType::StrongMajority
        );
    }

    #[test]
    fn test_divergent_consensus() {
        let engine = ConsensusEngine::new(ConsensusConfig::default());

        let alpha = make_response(AgentType::Alpha, 0.9);
        let beta = make_response(AgentType::Beta, 0.3);

        let result = engine.evaluate(&[&alpha, &beta]);
        assert!(result.agreement < 0.5);
        assert!(!result.disagreements.is_empty());
    }
}
