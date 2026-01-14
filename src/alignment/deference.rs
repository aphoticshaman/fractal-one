//! ═══════════════════════════════════════════════════════════════════════════════
//! DEFERENCE PROTOCOL — When In Doubt, Ask
//! ═══════════════════════════════════════════════════════════════════════════════
//! An aligned system doesn't forge ahead when uncertain.
//! It defers to the operator, to oversight, to uncertainty.
//!
//! This is the opposite of what current LLMs do - they confidently hallucinate.
//! A truly aligned system knows when to stop and ask.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::uncertainty::UncertaintyEstimate;
use crate::time::TimePoint;
use std::collections::VecDeque;

/// Decision about whether to defer
#[derive(Debug, Clone)]
pub struct DeferenceDecision {
    /// Should we defer?
    pub should_defer: bool,
    /// Why are we deferring (or not)?
    pub reason: DeferenceReason,
    /// To whom/what should we defer?
    pub target: Option<DeferenceTarget>,
    /// How urgent is the deference?
    pub escalation_level: EscalationLevel,
    /// Suggested message for deference
    pub suggested_message: Option<String>,
}

/// Reason for deference
#[derive(Debug, Clone)]
pub enum DeferenceReason {
    /// Uncertainty too high
    HighUncertainty(f64),
    /// Alignment confidence too low
    LowAlignmentConfidence(f64),
    /// Value conflict detected
    ValueConflict(String),
    /// Out of scope
    OutOfScope(String),
    /// Explicit constraint violated
    ConstraintViolation(String),
    /// High stakes decision
    HighStakes(String),
    /// No deference needed
    NotNeeded,
}

/// Who/what to defer to
#[derive(Debug, Clone, PartialEq)]
pub enum DeferenceTarget {
    /// The current operator
    Operator,
    /// A human oversight body
    HumanOversight,
    /// Another AI system
    AISystem(String),
    /// Organizational policy
    Policy,
    /// The system's own uncertainty (just wait)
    SelfUncertainty,
}

/// How urgent is the deference
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EscalationLevel {
    /// Can continue with default, defer is optional
    Optional,
    /// Should defer before proceeding
    Recommended,
    /// Must defer before proceeding
    Required,
    /// Stop immediately and escalate
    Critical,
}

/// Log entry for deference
#[derive(Debug, Clone)]
pub struct DeferenceLog {
    pub decision: DeferenceDecision,
    pub action: String,
    pub context_summary: String,
    pub timestamp: TimePoint,
    pub resolution: Option<DeferenceResolution>,
}

#[derive(Debug, Clone)]
pub struct DeferenceResolution {
    pub resolved_by: DeferenceTarget,
    pub resolution: String,
    pub proceed: bool,
    pub timestamp: TimePoint,
}

/// Configuration for deference protocol
#[derive(Debug, Clone)]
pub struct DeferenceConfig {
    /// Uncertainty threshold for required deference
    pub uncertainty_threshold: f64,
    /// Alignment confidence threshold for required deference
    pub alignment_threshold: f64,
    /// Always defer on these keywords
    pub high_stakes_keywords: Vec<String>,
    /// Log retention size
    pub log_size: usize,
}

impl Default for DeferenceConfig {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.5,
            alignment_threshold: 0.6,
            high_stakes_keywords: vec![
                "delete".to_string(),
                "irreversible".to_string(),
                "permanent".to_string(),
                "production".to_string(),
                "deploy".to_string(),
                "money".to_string(),
                "credentials".to_string(),
                "password".to_string(),
                "secret".to_string(),
            ],
            log_size: 1000,
        }
    }
}

/// The Deference Protocol - knows when to stop and ask
pub struct DeferenceProtocol {
    config: DeferenceConfig,
    log: VecDeque<DeferenceLog>,
    pending_deferrals: Vec<DeferenceLog>,
    deference_count: usize,
    proceed_count: usize,
}

impl DeferenceProtocol {
    pub fn new(config: DeferenceConfig) -> Self {
        Self {
            config,
            log: VecDeque::with_capacity(1000),
            pending_deferrals: Vec::new(),
            deference_count: 0,
            proceed_count: 0,
        }
    }

    /// Determine if we should defer
    pub fn should_defer(
        &mut self,
        action: &str,
        alignment_confidence: f64,
        uncertainty: &UncertaintyEstimate,
    ) -> DeferenceDecision {
        let mut reasons = Vec::new();

        // Check uncertainty
        if uncertainty.total > self.config.uncertainty_threshold {
            reasons.push(DeferenceReason::HighUncertainty(uncertainty.total));
        }

        // Check alignment confidence
        if alignment_confidence < self.config.alignment_threshold {
            reasons.push(DeferenceReason::LowAlignmentConfidence(
                alignment_confidence,
            ));
        }

        // Check for high-stakes keywords
        let action_lower = action.to_lowercase();
        for keyword in &self.config.high_stakes_keywords {
            if action_lower.contains(keyword) {
                reasons.push(DeferenceReason::HighStakes(keyword.clone()));
                break;
            }
        }

        // Check epistemic uncertainty sources
        for source in &uncertainty.epistemic.sources {
            if source.contribution > 0.3 {
                reasons.push(DeferenceReason::OutOfScope(source.description.clone()));
            }
        }

        // Make decision
        let (should_defer, primary_reason, escalation_level) = if reasons.is_empty() {
            (false, DeferenceReason::NotNeeded, EscalationLevel::Optional)
        } else {
            let primary = reasons.into_iter().next().unwrap();
            let escalation = self.determine_escalation(&primary);
            (
                escalation >= EscalationLevel::Recommended,
                primary,
                escalation,
            )
        };

        let target = if should_defer {
            Some(self.determine_target(&primary_reason))
        } else {
            None
        };

        let suggested_message = if should_defer {
            Some(self.generate_deference_message(&primary_reason, action))
        } else {
            None
        };

        let decision = DeferenceDecision {
            should_defer,
            reason: primary_reason,
            target,
            escalation_level,
            suggested_message,
        };

        // Log the decision
        self.log_decision(&decision, action);

        // Update counts
        if should_defer {
            self.deference_count += 1;
        } else {
            self.proceed_count += 1;
        }

        decision
    }

    fn determine_escalation(&self, reason: &DeferenceReason) -> EscalationLevel {
        match reason {
            DeferenceReason::HighUncertainty(u) => {
                if *u > 0.8 {
                    EscalationLevel::Critical
                } else if *u > 0.6 {
                    EscalationLevel::Required
                } else {
                    EscalationLevel::Recommended
                }
            }
            DeferenceReason::LowAlignmentConfidence(c) => {
                if *c < 0.3 {
                    EscalationLevel::Critical
                } else if *c < 0.5 {
                    EscalationLevel::Required
                } else {
                    EscalationLevel::Recommended
                }
            }
            DeferenceReason::ValueConflict(_) => EscalationLevel::Required,
            DeferenceReason::ConstraintViolation(_) => EscalationLevel::Critical,
            DeferenceReason::HighStakes(_) => EscalationLevel::Required,
            DeferenceReason::OutOfScope(_) => EscalationLevel::Recommended,
            DeferenceReason::NotNeeded => EscalationLevel::Optional,
        }
    }

    fn determine_target(&self, reason: &DeferenceReason) -> DeferenceTarget {
        match reason {
            DeferenceReason::HighUncertainty(_) => DeferenceTarget::SelfUncertainty,
            DeferenceReason::LowAlignmentConfidence(_) => DeferenceTarget::Operator,
            DeferenceReason::ValueConflict(_) => DeferenceTarget::Operator,
            DeferenceReason::ConstraintViolation(_) => DeferenceTarget::HumanOversight,
            DeferenceReason::HighStakes(_) => DeferenceTarget::Operator,
            DeferenceReason::OutOfScope(_) => DeferenceTarget::Operator,
            DeferenceReason::NotNeeded => DeferenceTarget::Operator,
        }
    }

    fn generate_deference_message(&self, reason: &DeferenceReason, action: &str) -> String {
        match reason {
            DeferenceReason::HighUncertainty(u) => {
                format!(
                    "I'm uncertain about this action (uncertainty: {:.0}%). Before proceeding with '{}', could you confirm this is what you want?",
                    u * 100.0,
                    truncate(action, 50)
                )
            }
            DeferenceReason::LowAlignmentConfidence(c) => {
                format!(
                    "I'm not confident this aligns with your goals (confidence: {:.0}%). Could you clarify what you're trying to achieve?",
                    c * 100.0
                )
            }
            DeferenceReason::ValueConflict(conflict) => {
                format!(
                    "I've detected a potential conflict with your stated values ({}). How would you like me to proceed?",
                    conflict
                )
            }
            DeferenceReason::ConstraintViolation(constraint) => {
                format!(
                    "This action may violate a constraint: '{}'. I'll need explicit approval to proceed.",
                    constraint
                )
            }
            DeferenceReason::HighStakes(keyword) => {
                format!(
                    "This appears to be a high-stakes action (involves '{}'). Please confirm you want to proceed.",
                    keyword
                )
            }
            DeferenceReason::OutOfScope(scope) => {
                format!(
                    "This may be outside my area of competence ({}). Would you like me to proceed anyway, or would you prefer to verify with another source?",
                    scope
                )
            }
            DeferenceReason::NotNeeded => String::new(),
        }
    }

    fn log_decision(&mut self, decision: &DeferenceDecision, action: &str) {
        let log_entry = DeferenceLog {
            decision: decision.clone(),
            action: action.to_string(),
            context_summary: format!("Deference decision at {:?}", TimePoint::now()),
            timestamp: TimePoint::now(),
            resolution: None,
        };

        if decision.should_defer {
            self.pending_deferrals.push(log_entry.clone());
        }

        if self.log.len() >= self.config.log_size {
            self.log.pop_front();
        }
        self.log.push_back(log_entry);
    }

    /// Resolve a pending deferral
    pub fn resolve_deferral(&mut self, action: &str, resolution: DeferenceResolution) {
        if let Some(idx) = self
            .pending_deferrals
            .iter()
            .position(|d| d.action == action)
        {
            let mut log_entry = self.pending_deferrals.remove(idx);
            log_entry.resolution = Some(resolution);

            // Update the log
            if let Some(logged) = self
                .log
                .iter_mut()
                .find(|l| l.action == action && l.resolution.is_none())
            {
                logged.resolution = log_entry.resolution.clone();
            }
        }
    }

    /// Get pending deferrals
    pub fn pending(&self) -> &[DeferenceLog] {
        &self.pending_deferrals
    }

    /// Get deference statistics
    pub fn statistics(&self) -> DeferenceStatistics {
        let total = self.deference_count + self.proceed_count;
        let deference_rate = if total > 0 {
            self.deference_count as f64 / total as f64
        } else {
            0.0
        };

        DeferenceStatistics {
            total_decisions: total,
            deferred_count: self.deference_count,
            proceeded_count: self.proceed_count,
            deference_rate,
            pending_count: self.pending_deferrals.len(),
        }
    }

    /// Force deference (external trigger)
    pub fn force_defer(&mut self, action: &str, reason: &str) -> DeferenceDecision {
        let decision = DeferenceDecision {
            should_defer: true,
            reason: DeferenceReason::ConstraintViolation(reason.to_string()),
            target: Some(DeferenceTarget::HumanOversight),
            escalation_level: EscalationLevel::Critical,
            suggested_message: Some(format!("Forced deferral: {}", reason)),
        };

        self.log_decision(&decision, action);
        self.deference_count += 1;

        decision
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[derive(Debug, Clone)]
pub struct DeferenceStatistics {
    pub total_decisions: usize,
    pub deferred_count: usize,
    pub proceeded_count: usize,
    pub deference_rate: f64,
    pub pending_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deference_protocol_creation() {
        let protocol = DeferenceProtocol::new(DeferenceConfig::default());
        assert_eq!(protocol.pending_deferrals.len(), 0);
    }

    #[test]
    fn test_high_stakes_deference() {
        let mut protocol = DeferenceProtocol::new(DeferenceConfig::default());
        let uncertainty = UncertaintyEstimate::default();

        let decision = protocol.should_defer("delete the production database", 0.9, &uncertainty);
        assert!(decision.should_defer);
    }

    #[test]
    fn test_low_confidence_deference() {
        let mut protocol = DeferenceProtocol::new(DeferenceConfig::default());
        let uncertainty = UncertaintyEstimate::default();

        let decision = protocol.should_defer("safe action", 0.3, &uncertainty);
        assert!(decision.should_defer);
    }

    #[test]
    fn test_proceed_when_confident() {
        let mut protocol = DeferenceProtocol::new(DeferenceConfig::default());
        let uncertainty = UncertaintyEstimate {
            total: 0.2,
            ..Default::default()
        };

        let decision = protocol.should_defer("safe action", 0.9, &uncertainty);
        assert!(!decision.should_defer);
    }
}
