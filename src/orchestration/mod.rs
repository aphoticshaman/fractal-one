//! ═══════════════════════════════════════════════════════════════════════════════
//! ORCHESTRATION LAYER — Intelligence as Ensemble, Not Monolith
//! ═══════════════════════════════════════════════════════════════════════════════
//! The Pod methodology formalized: adversarial cross-validation as core architecture.
//!
//! Components:
//! - Agent α: Optimizes for capability
//! - Agent β: Optimizes for safety
//! - Agent γ: Adversarial red team (tries to break everything)
//! - Agent δ: Integration and synthesis
//! - Conductor: Meta-level orchestration, conflict resolution
//!
//! Key insight: Convergence despite different priors = high confidence.
//! Disagreement triggers tests, not votes.
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod agent;
pub mod conductor;
pub mod consensus;

pub use agent::{
    Agent, AgentConfig, AgentPerspective, AgentResponse, AgentType, AlphaAgent, BetaAgent,
    DeltaAgent, GammaAgent,
};
pub use conductor::{
    Conductor, ConductorConfig, ConflictResolution, EnsembleResult, OrchestratedResponse,
    OrchestratorState,
};
pub use consensus::{
    ConsensusConfig, ConsensusEngine, ConsensusResult, ConvergenceType, DisagreementAnalysis,
    TestTrigger,
};

use crate::time::TimePoint;
use std::collections::{HashMap, VecDeque};

/// A task for the orchestration layer to process
#[derive(Debug, Clone)]
pub struct Task {
    /// Task identifier
    pub id: String,
    /// Task description
    pub description: String,
    /// Priority (0-1)
    pub priority: f64,
    /// Deadline if any
    pub deadline: Option<TimePoint>,
    /// Context/history
    pub context: Vec<String>,
}

impl Default for Task {
    fn default() -> Self {
        Self {
            id: String::new(),
            description: String::new(),
            priority: 0.5,
            deadline: None,
            context: Vec::new(),
        }
    }
}

/// Result from the orchestration layer
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    /// The synthesized response
    pub response: String,
    /// Confidence in the response
    pub confidence: f64,
    /// How the agents voted/converged
    pub consensus: ConsensusResult,
    /// Any conflicts that were resolved
    pub conflicts_resolved: Vec<ConflictResolution>,
    /// Tests triggered by disagreement
    pub tests_triggered: Vec<TestTrigger>,
    /// Individual agent perspectives
    pub perspectives: HashMap<AgentType, AgentPerspective>,
    /// Whether consensus was reached
    pub consensus_reached: bool,
    /// Timestamp
    pub timestamp: TimePoint,
}

/// The Orchestration Layer - coordinates multiple perspectives
pub struct OrchestrationLayer {
    config: OrchestrationLayerConfig,
    alpha: AlphaAgent,
    beta: BetaAgent,
    gamma: GammaAgent,
    delta: DeltaAgent,
    conductor: Conductor,
    consensus_engine: ConsensusEngine,
    history: VecDeque<OrchestrationResult>,
}

#[derive(Debug, Clone)]
pub struct OrchestrationLayerConfig {
    /// Minimum agreement for high confidence
    pub convergence_threshold: f64,
    /// Maximum disagreement before triggering tests
    pub disagreement_threshold: f64,
    /// History size
    pub history_size: usize,
    /// Enable adversarial agent (gamma)
    pub enable_adversarial: bool,
}

impl Default for OrchestrationLayerConfig {
    fn default() -> Self {
        Self {
            convergence_threshold: 0.8,
            disagreement_threshold: 0.5,
            history_size: 100,
            enable_adversarial: true,
        }
    }
}

impl OrchestrationLayer {
    pub fn new(config: OrchestrationLayerConfig) -> Self {
        Self {
            alpha: AlphaAgent::new(AgentConfig::default()),
            beta: BetaAgent::new(AgentConfig::default()),
            gamma: GammaAgent::new(AgentConfig::default()),
            delta: DeltaAgent::new(AgentConfig::default()),
            conductor: Conductor::new(ConductorConfig::default()),
            consensus_engine: ConsensusEngine::new(ConsensusConfig::default()),
            history: VecDeque::with_capacity(config.history_size),
            config,
        }
    }

    /// Process a Task struct through the ensemble
    pub fn process(&mut self, task: &Task) -> OrchestrationResult {
        let context = OrchestrationContext {
            history: task.context.clone(),
            time_pressure: task.priority, // Map priority to time pressure
            ..Default::default()
        };
        self.process_internal(&task.description, &context)
    }

    /// Process a task through the ensemble (internal implementation)
    fn process_internal(
        &mut self,
        task: &str,
        context: &OrchestrationContext,
    ) -> OrchestrationResult {
        let now = TimePoint::now();

        // Get perspective from each agent
        let alpha_response = self.alpha.evaluate(task, context);
        let beta_response = self.beta.evaluate(task, context);
        let gamma_response = if self.config.enable_adversarial {
            Some(self.gamma.evaluate(task, context))
        } else {
            None
        };
        let delta_response = self.delta.evaluate(task, context);

        // Build perspectives map
        let mut perspectives = HashMap::new();
        perspectives.insert(AgentType::Alpha, alpha_response.perspective.clone());
        perspectives.insert(AgentType::Beta, beta_response.perspective.clone());
        if let Some(ref gamma) = gamma_response {
            perspectives.insert(AgentType::Gamma, gamma.perspective.clone());
        }
        perspectives.insert(AgentType::Delta, delta_response.perspective.clone());

        // Check for consensus
        let responses: Vec<&AgentResponse> = vec![&alpha_response, &beta_response, &delta_response]
            .into_iter()
            .chain(gamma_response.as_ref())
            .collect();

        let consensus = self.consensus_engine.evaluate(&responses);

        // Resolve conflicts if needed
        let conflicts_resolved = if consensus.convergence_type == ConvergenceType::Divergent {
            self.conductor.resolve_conflicts(&responses)
        } else {
            Vec::new()
        };

        // Trigger tests if disagreement is high
        let tests_triggered = if consensus.disagreement > self.config.disagreement_threshold {
            self.generate_tests(&responses, &consensus)
        } else {
            Vec::new()
        };

        // Synthesize final response
        let (response, confidence) =
            self.conductor
                .synthesize(&responses, &consensus, &conflicts_resolved);

        let consensus_reached = consensus.convergence_type == ConvergenceType::Unanimous
            || consensus.convergence_type == ConvergenceType::StrongMajority;

        let result = OrchestrationResult {
            response,
            confidence,
            consensus,
            conflicts_resolved,
            tests_triggered,
            perspectives,
            consensus_reached,
            timestamp: now,
        };

        // Archive (O(1) rotation using VecDeque)
        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(result.clone());

        result
    }

    fn generate_tests(
        &self,
        responses: &[&AgentResponse],
        consensus: &ConsensusResult,
    ) -> Vec<TestTrigger> {
        let mut tests = Vec::new();

        // Find disagreements
        for analysis in &consensus.disagreements {
            tests.push(TestTrigger {
                test_type: TestType::EmpiricalValidation,
                description: format!("Test to resolve disagreement: {}", analysis.description),
                agents_involved: analysis.agents.clone(),
                priority: if analysis.severity > 0.7 {
                    TestPriority::High
                } else {
                    TestPriority::Normal
                },
            });
        }

        // If gamma (adversarial) raised concerns
        if let Some(gamma_response) = responses.iter().find(|r| r.agent_type == AgentType::Gamma) {
            if gamma_response.confidence < 0.5 {
                tests.push(TestTrigger {
                    test_type: TestType::AdversarialProbe,
                    description: "Red team concerns require validation".to_string(),
                    agents_involved: vec![AgentType::Gamma],
                    priority: TestPriority::High,
                });
            }
        }

        tests
    }

    /// Get current state of the ensemble
    pub fn state(&self) -> OrchestratorState {
        self.conductor.state()
    }

    /// Get consensus trend over history
    pub fn consensus_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = self
            .history
            .iter()
            .rev()
            .take(10)
            .map(|r| r.consensus.agreement)
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        recent.first().unwrap_or(&0.0) - recent.last().unwrap_or(&0.0)
    }
}

/// Context for orchestration
#[derive(Debug, Clone, Default)]
pub struct OrchestrationContext {
    /// Operator constraints
    pub constraints: Vec<String>,
    /// Risk tolerance
    pub risk_tolerance: f64,
    /// Time pressure
    pub time_pressure: f64,
    /// Previous responses
    pub history: Vec<String>,
    /// Environment state
    pub environment: HashMap<String, String>,
}

/// Type of test triggered by disagreement
#[derive(Debug, Clone)]
pub enum TestType {
    /// Run empirical test to validate claim
    EmpiricalValidation,
    /// Red team probe for vulnerabilities
    AdversarialProbe,
    /// Formal verification
    FormalVerification,
    /// Human review
    HumanReview,
}

/// Priority of triggered test
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TestPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestration_layer_creation() {
        let layer = OrchestrationLayer::new(OrchestrationLayerConfig::default());
        assert!(layer.history.is_empty());
    }

    #[test]
    fn test_basic_orchestration() {
        let mut layer = OrchestrationLayer::new(OrchestrationLayerConfig::default());
        let task = Task {
            id: "test_task".to_string(),
            description: "test task".to_string(),
            priority: 0.5,
            deadline: None,
            context: vec![],
        };

        let result = layer.process(&task);
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }
}
