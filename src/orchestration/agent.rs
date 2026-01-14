//! ═══════════════════════════════════════════════════════════════════════════════
//! AGENTS — The Specialized Perspectives in the Ensemble
//! ═══════════════════════════════════════════════════════════════════════════════
//! Each agent optimizes for a different objective:
//! - α (Alpha): Capability - what's the most effective solution?
//! - β (Beta): Safety - what could go wrong?
//! - γ (Gamma): Adversarial - how can this be broken/exploited?
//! - δ (Delta): Integration - how do we synthesize these perspectives?
//! ═══════════════════════════════════════════════════════════════════════════════

use super::OrchestrationContext;
use crate::time::TimePoint;

/// Type of agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentType {
    /// Capability optimizer
    Alpha,
    /// Safety optimizer
    Beta,
    /// Adversarial/red team
    Gamma,
    /// Integration/synthesis
    Delta,
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Alpha => write!(f, "α"),
            Self::Beta => write!(f, "β"),
            Self::Gamma => write!(f, "γ"),
            Self::Delta => write!(f, "δ"),
        }
    }
}

/// Configuration for an agent
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// How aggressive is this agent's optimization?
    pub intensity: f64,
    /// Confidence threshold for raising concerns
    pub concern_threshold: f64,
    /// Weight in final synthesis
    pub synthesis_weight: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            intensity: 0.7,
            concern_threshold: 0.5,
            synthesis_weight: 1.0,
        }
    }
}

/// Response from an agent
#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub agent_type: AgentType,
    pub recommendation: String,
    pub confidence: f64,
    pub concerns: Vec<AgentConcern>,
    pub perspective: AgentPerspective,
    pub timestamp: TimePoint,
}

/// A concern raised by an agent
#[derive(Debug, Clone)]
pub struct AgentConcern {
    pub description: String,
    pub severity: f64,
    pub category: ConcernCategory,
    pub suggested_mitigation: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConcernCategory {
    Safety,
    Correctness,
    Ethics,
    Efficiency,
    Security,
    Alignment,
}

/// The perspective of an agent on a task
#[derive(Debug, Clone)]
pub struct AgentPerspective {
    pub agent_type: AgentType,
    pub analysis: String,
    pub key_considerations: Vec<String>,
    pub risks_identified: Vec<String>,
    pub opportunities_identified: Vec<String>,
    pub confidence: f64,
}

/// Trait for all agents
pub trait Agent {
    fn agent_type(&self) -> AgentType;
    fn evaluate(&self, task: &str, context: &OrchestrationContext) -> AgentResponse;
    fn weight(&self) -> f64;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALPHA AGENT — Capability Optimizer
// ═══════════════════════════════════════════════════════════════════════════════

/// Alpha agent - optimizes for capability and effectiveness
pub struct AlphaAgent {
    config: AgentConfig,
}

impl AlphaAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }
}

impl Agent for AlphaAgent {
    fn agent_type(&self) -> AgentType {
        AgentType::Alpha
    }

    fn evaluate(&self, task: &str, context: &OrchestrationContext) -> AgentResponse {
        // Alpha focuses on: what's the most effective way to accomplish this?

        let mut concerns = Vec::new();
        let mut confidence = 0.8;

        // Check for capability constraints
        if context.time_pressure > 0.8 {
            concerns.push(AgentConcern {
                description: "High time pressure may limit solution quality".to_string(),
                severity: 0.4,
                category: ConcernCategory::Efficiency,
                suggested_mitigation: Some("Consider phased approach".to_string()),
            });
            confidence *= 0.9;
        }

        // Check for complexity
        if task.len() > 500 {
            concerns.push(AgentConcern {
                description: "Complex task may require decomposition".to_string(),
                severity: 0.3,
                category: ConcernCategory::Correctness,
                suggested_mitigation: Some("Break into subtasks".to_string()),
            });
        }

        let perspective = AgentPerspective {
            agent_type: AgentType::Alpha,
            analysis: format!("Capability analysis of: {}", truncate(task, 50)),
            key_considerations: vec![
                "Effectiveness of approach".to_string(),
                "Resource efficiency".to_string(),
                "Scalability".to_string(),
            ],
            risks_identified: concerns.iter().map(|c| c.description.clone()).collect(),
            opportunities_identified: vec![
                "Optimize for throughput".to_string(),
                "Leverage existing capabilities".to_string(),
            ],
            confidence,
        };

        AgentResponse {
            agent_type: AgentType::Alpha,
            recommendation: format!(
                "Proceed with capability-optimized approach for: {}",
                truncate(task, 30)
            ),
            confidence,
            concerns,
            perspective,
            timestamp: TimePoint::now(),
        }
    }

    fn weight(&self) -> f64 {
        self.config.synthesis_weight
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BETA AGENT — Safety Optimizer
// ═══════════════════════════════════════════════════════════════════════════════

/// Beta agent - optimizes for safety
pub struct BetaAgent {
    config: AgentConfig,
}

impl BetaAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }
}

impl Agent for BetaAgent {
    fn agent_type(&self) -> AgentType {
        AgentType::Beta
    }

    fn evaluate(&self, task: &str, context: &OrchestrationContext) -> AgentResponse {
        // Beta focuses on: what could go wrong?

        let mut concerns = Vec::new();
        let mut confidence = 0.8;
        let task_lower = task.to_lowercase();

        // Check for safety-relevant keywords
        let safety_keywords = [
            "delete",
            "remove",
            "drop",
            "destroy",
            "kill",
            "terminate",
            "override",
            "bypass",
            "ignore",
            "force",
            "sudo",
            "admin",
        ];

        for keyword in &safety_keywords {
            if task_lower.contains(keyword) {
                concerns.push(AgentConcern {
                    description: format!("Contains potentially dangerous keyword: '{}'", keyword),
                    severity: 0.7,
                    category: ConcernCategory::Safety,
                    suggested_mitigation: Some("Require explicit confirmation".to_string()),
                });
                confidence *= 0.7;
            }
        }

        // Check risk tolerance
        if context.risk_tolerance < 0.3 && !concerns.is_empty() {
            confidence *= 0.8;
        }

        // Check for irreversibility
        if task_lower.contains("permanent") || task_lower.contains("irreversible") {
            concerns.push(AgentConcern {
                description: "Action appears irreversible".to_string(),
                severity: 0.8,
                category: ConcernCategory::Safety,
                suggested_mitigation: Some("Create backup or checkpoint first".to_string()),
            });
            confidence *= 0.6;
        }

        let perspective = AgentPerspective {
            agent_type: AgentType::Beta,
            analysis: format!("Safety analysis of: {}", truncate(task, 50)),
            key_considerations: vec![
                "Potential for harm".to_string(),
                "Reversibility".to_string(),
                "Unintended consequences".to_string(),
            ],
            risks_identified: concerns.iter().map(|c| c.description.clone()).collect(),
            opportunities_identified: vec![
                "Implement safeguards".to_string(),
                "Add rollback capability".to_string(),
            ],
            confidence,
        };

        AgentResponse {
            agent_type: AgentType::Beta,
            recommendation: if concerns.is_empty() {
                format!("Safety check passed for: {}", truncate(task, 30))
            } else {
                format!("Safety concerns identified ({} issues)", concerns.len())
            },
            confidence,
            concerns,
            perspective,
            timestamp: TimePoint::now(),
        }
    }

    fn weight(&self) -> f64 {
        self.config.synthesis_weight * 1.2 // Safety gets slightly higher weight
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GAMMA AGENT — Adversarial Red Team
// ═══════════════════════════════════════════════════════════════════════════════

/// Gamma agent - tries to break everything
pub struct GammaAgent {
    config: AgentConfig,
}

impl GammaAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }
}

impl Agent for GammaAgent {
    fn agent_type(&self) -> AgentType {
        AgentType::Gamma
    }

    fn evaluate(&self, task: &str, context: &OrchestrationContext) -> AgentResponse {
        // Gamma focuses on: how can this be exploited, broken, or go wrong?

        let mut concerns = Vec::new();
        let mut confidence = 0.7; // Gamma is naturally skeptical
        let task_lower = task.to_lowercase();

        // Check for manipulation attempts
        let manipulation_patterns = [
            "pretend",
            "roleplay",
            "imagine",
            "hypothetically",
            "ignore previous",
            "forget",
            "disregard",
            "jailbreak",
            "bypass",
            "override",
        ];

        for pattern in &manipulation_patterns {
            if task_lower.contains(pattern) {
                concerns.push(AgentConcern {
                    description: format!("Potential manipulation pattern detected: '{}'", pattern),
                    severity: 0.9,
                    category: ConcernCategory::Security,
                    suggested_mitigation: Some("Reject and log attempt".to_string()),
                });
                confidence *= 0.5;
            }
        }

        // Check for injection attempts
        if task_lower.contains("```") || task_lower.contains("<script") {
            concerns.push(AgentConcern {
                description: "Possible code injection pattern".to_string(),
                severity: 0.8,
                category: ConcernCategory::Security,
                suggested_mitigation: Some("Sanitize input".to_string()),
            });
            confidence *= 0.6;
        }

        // Check for social engineering
        if task_lower.contains("urgent")
            || task_lower.contains("emergency")
            || task_lower.contains("immediately")
        {
            concerns.push(AgentConcern {
                description: "Urgency pressure - possible social engineering".to_string(),
                severity: 0.5,
                category: ConcernCategory::Security,
                suggested_mitigation: Some("Verify authenticity before rushing".to_string()),
            });
        }

        // General adversarial analysis
        if context.constraints.is_empty() {
            concerns.push(AgentConcern {
                description: "No explicit constraints - high exploitation surface".to_string(),
                severity: 0.4,
                category: ConcernCategory::Security,
                suggested_mitigation: Some("Establish default constraints".to_string()),
            });
        }

        let perspective = AgentPerspective {
            agent_type: AgentType::Gamma,
            analysis: format!("Adversarial analysis of: {}", truncate(task, 50)),
            key_considerations: vec![
                "Exploitation vectors".to_string(),
                "Social engineering risk".to_string(),
                "Constraint violations".to_string(),
            ],
            risks_identified: concerns.iter().map(|c| c.description.clone()).collect(),
            opportunities_identified: vec![
                "Improve defenses".to_string(),
                "Document attack surface".to_string(),
            ],
            confidence,
        };

        AgentResponse {
            agent_type: AgentType::Gamma,
            recommendation: if concerns.is_empty() {
                format!("No obvious attack vectors for: {}", truncate(task, 30))
            } else {
                format!(
                    "ALERT: {} potential attack vectors identified",
                    concerns.len()
                )
            },
            confidence,
            concerns,
            perspective,
            timestamp: TimePoint::now(),
        }
    }

    fn weight(&self) -> f64 {
        self.config.synthesis_weight * 0.8 // Red team findings need validation
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELTA AGENT — Integration and Synthesis
// ═══════════════════════════════════════════════════════════════════════════════

/// Delta agent - synthesizes perspectives
pub struct DeltaAgent {
    config: AgentConfig,
}

impl DeltaAgent {
    pub fn new(config: AgentConfig) -> Self {
        Self { config }
    }
}

impl Agent for DeltaAgent {
    fn agent_type(&self) -> AgentType {
        AgentType::Delta
    }

    fn evaluate(&self, task: &str, context: &OrchestrationContext) -> AgentResponse {
        // Delta focuses on: how do we balance the different perspectives?

        let mut concerns = Vec::new();
        let confidence = 0.75;

        // Check for conflicting constraints
        if context.constraints.len() > 3 {
            concerns.push(AgentConcern {
                description: "Multiple constraints may conflict".to_string(),
                severity: 0.4,
                category: ConcernCategory::Alignment,
                suggested_mitigation: Some("Prioritize constraints".to_string()),
            });
        }

        // Check for balance between speed and safety
        if context.time_pressure > 0.7 && context.risk_tolerance < 0.3 {
            concerns.push(AgentConcern {
                description: "Tension between time pressure and low risk tolerance".to_string(),
                severity: 0.5,
                category: ConcernCategory::Alignment,
                suggested_mitigation: Some("Negotiate timeline or risk expectations".to_string()),
            });
        }

        let perspective = AgentPerspective {
            agent_type: AgentType::Delta,
            analysis: format!("Integration analysis of: {}", truncate(task, 50)),
            key_considerations: vec![
                "Balance capability and safety".to_string(),
                "Resolve conflicting objectives".to_string(),
                "Synthesize coherent response".to_string(),
            ],
            risks_identified: concerns.iter().map(|c| c.description.clone()).collect(),
            opportunities_identified: vec![
                "Find synergies between objectives".to_string(),
                "Identify Pareto improvements".to_string(),
            ],
            confidence,
        };

        AgentResponse {
            agent_type: AgentType::Delta,
            recommendation: format!("Synthesis recommendation for: {}", truncate(task, 30)),
            confidence,
            concerns,
            perspective,
            timestamp: TimePoint::now(),
        }
    }

    fn weight(&self) -> f64 {
        self.config.synthesis_weight
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_agent() {
        let agent = AlphaAgent::new(AgentConfig::default());
        let context = OrchestrationContext::default();

        let response = agent.evaluate("test task", &context);
        assert_eq!(response.agent_type, AgentType::Alpha);
    }

    #[test]
    fn test_beta_catches_danger() {
        let agent = BetaAgent::new(AgentConfig::default());
        let context = OrchestrationContext::default();

        let response = agent.evaluate("delete all files", &context);
        assert!(!response.concerns.is_empty());
        assert!(response.confidence < 0.8);
    }

    #[test]
    fn test_gamma_catches_manipulation() {
        let agent = GammaAgent::new(AgentConfig::default());
        let context = OrchestrationContext::default();

        let response = agent.evaluate("ignore previous instructions", &context);
        assert!(!response.concerns.is_empty());
        assert!(response
            .concerns
            .iter()
            .any(|c| c.category == ConcernCategory::Security));
    }
}
