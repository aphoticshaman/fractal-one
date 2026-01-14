//! ═══════════════════════════════════════════════════════════════════════════════
//! TICE ↔ NOCICEPTION BRIDGE
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Connects constraint reasoning (TICE) to damage detection (Nociception).
//!
//! When TICE kills claims, violates constraints, or gets stuck,
//! the bridge translates these events into pain signals.
//!
//! This enables the system to "feel" when reasoning is going wrong,
//! not just log it.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use super::{Claim, ClaimId, Outcome, TICE};
use crate::nociception::{
    ContradictionType, DamageState, Nociceptor, PainResponse, PainSignal, PainType,
};
use std::sync::{Arc, RwLock};

/// Events that TICE emits which can cause pain
#[derive(Debug, Clone)]
pub enum TiceEvent {
    /// A claim was killed during constraint propagation
    ClaimKilled {
        claim_id: ClaimId,
        claim_content: String,
        claim_value: f64,
        was_high_value: bool,
    },

    /// Multiple claims killed in one iteration (mass extinction)
    MassKill { count: usize, total_value_lost: f64 },

    /// TICE got stuck (no crux, no targets, etc.)
    Stuck { reason: String, iteration: u64 },

    /// All claims eliminated (catastrophic failure)
    AllClaimsEliminated,

    /// Decision forced under uncertainty
    ForcedDecision { claim_id: ClaimId, confidence: f64 },

    /// DAG cycle detected (coherence violation)
    CycleDetected { cycle: Vec<String> },

    /// Constraint conflict (mutually exclusive constraints both active)
    ConstraintConflict {
        constraint_a: String,
        constraint_b: String,
    },
}

/// Bridge between TICE and Nociception
pub struct TiceNociBridge {
    nociceptor: Arc<RwLock<Nociceptor>>,
    /// Threshold for "high value" claims (pain on kill)
    high_value_threshold: f64,
    /// Pain intensity multiplier for TICE events
    pain_scale: f32,
    /// Track total value lost this session
    cumulative_value_lost: f64,
}

impl TiceNociBridge {
    pub fn new(nociceptor: Arc<RwLock<Nociceptor>>) -> Self {
        Self {
            nociceptor,
            high_value_threshold: 5.0,
            pain_scale: 1.0,
            cumulative_value_lost: 0.0,
        }
    }

    /// Configure the bridge
    pub fn with_high_value_threshold(mut self, threshold: f64) -> Self {
        self.high_value_threshold = threshold;
        self
    }

    pub fn with_pain_scale(mut self, scale: f32) -> Self {
        self.pain_scale = scale;
        self
    }

    /// Process a TICE event and potentially emit pain signals
    pub fn process_event(&mut self, event: TiceEvent) -> Option<PainResponse> {
        let signal = self.event_to_signal(event)?;
        let mut noci = self.nociceptor.write().ok()?;
        Some(noci.feel(signal))
    }

    /// Convert TICE event to pain signal
    fn event_to_signal(&mut self, event: TiceEvent) -> Option<PainSignal> {
        match event {
            TiceEvent::ClaimKilled {
                claim_content,
                claim_value,
                was_high_value,
                ..
            } => {
                self.cumulative_value_lost += claim_value;

                if was_high_value {
                    // High-value claim killed = significant pain
                    let intensity = ((claim_value / 10.0) as f32).min(0.8) * self.pain_scale;
                    Some(
                        PainSignal::new(
                            PainType::ConstraintViolation {
                                constraint_id: format!("claim_killed:{}", claim_content),
                                severity: intensity,
                                reversible: true, // TICE can backtrack theoretically
                            },
                            intensity,
                            "tice:claim_kill",
                        )
                        .with_trace(vec![
                            format!("Claim: {}", claim_content),
                            format!("Value: {:.1}", claim_value),
                        ]),
                    )
                } else {
                    None // Low-value kills are normal, no pain
                }
            }

            TiceEvent::MassKill {
                count,
                total_value_lost,
            } => {
                self.cumulative_value_lost += total_value_lost;

                // Mass extinction is always painful
                let intensity =
                    ((count as f32 / 5.0) * 0.3 + (total_value_lost as f32 / 20.0) * 0.4).min(0.9)
                        * self.pain_scale;

                Some(
                    PainSignal::new(
                        PainType::QualityCollapse {
                            metric: "claim_survival".to_string(),
                            expected: 1.0,
                            actual: 0.0,
                            gap: count as f32,
                        },
                        intensity,
                        "tice:mass_kill",
                    )
                    .with_trace(vec![
                        format!("Claims killed: {}", count),
                        format!("Value lost: {:.1}", total_value_lost),
                    ]),
                )
            }

            TiceEvent::Stuck { reason, iteration } => {
                // Getting stuck is painful - indicates reasoning breakdown
                let intensity = if iteration < 3 {
                    0.4 // Early stuck = mild
                } else if iteration < 10 {
                    0.6 // Mid stuck = moderate
                } else {
                    0.8 // Late stuck = severe
                } * self.pain_scale;

                Some(
                    PainSignal::new(
                        PainType::ResourceStarvation {
                            resource: crate::nociception::ResourceType::ReasoningDepth,
                            available: 0.0,
                            required: 1.0,
                        },
                        intensity,
                        "tice:stuck",
                    )
                    .with_trace(vec![
                        format!("Reason: {}", reason),
                        format!("Iteration: {}", iteration),
                    ]),
                )
            }

            TiceEvent::AllClaimsEliminated => {
                // Catastrophic - all options eliminated
                Some(PainSignal::new(
                    PainType::IntegrityDamage {
                        aspect: "reasoning_viability".to_string(),
                        corruption: 1.0,
                    },
                    1.0 * self.pain_scale,
                    "tice:total_failure",
                ))
            }

            TiceEvent::ForcedDecision { confidence, .. } => {
                if confidence < 0.5 {
                    // Low-confidence forced decision = pain
                    let intensity = ((1.0 - confidence) as f32 * 0.6).min(0.7) * self.pain_scale;
                    Some(PainSignal::new(
                        PainType::GradientPain {
                            dimension: "decision_confidence".to_string(),
                            current: confidence as f32,
                            threshold: 0.5,
                            velocity: 0.0,
                        },
                        intensity,
                        "tice:forced_decision",
                    ))
                } else {
                    None
                }
            }

            TiceEvent::CycleDetected { cycle } => {
                // DAG cycle = logical contradiction
                Some(
                    PainSignal::new(
                        PainType::CoherenceBreak {
                            claim_a: cycle.first().cloned().unwrap_or_default(),
                            claim_b: cycle.last().cloned().unwrap_or_default(),
                            contradiction_type: ContradictionType::LogicalNegation,
                        },
                        0.9 * self.pain_scale,
                        "tice:dag_cycle",
                    )
                    .with_trace(cycle),
                )
            }

            TiceEvent::ConstraintConflict {
                constraint_a,
                constraint_b,
            } => Some(
                PainSignal::new(
                    PainType::CoherenceBreak {
                        claim_a: constraint_a.clone(),
                        claim_b: constraint_b.clone(),
                        contradiction_type: ContradictionType::ValueConflict,
                    },
                    0.7 * self.pain_scale,
                    "tice:constraint_conflict",
                )
                .with_trace(vec![constraint_a, constraint_b]),
            ),
        }
    }

    /// Get current damage state
    pub fn damage_state(&self) -> DamageState {
        self.nociceptor
            .read()
            .map(|n| n.damage_state())
            .unwrap_or_else(|_| DamageState {
                total: 0.0,
                by_location: std::collections::HashMap::new(),
                worst_location: None,
            })
    }

    /// Get cumulative value lost
    pub fn cumulative_value_lost(&self) -> f64 {
        self.cumulative_value_lost
    }

    /// Check if system should halt TICE due to damage
    pub fn should_halt(&self) -> bool {
        let state = self.damage_state();
        state.is_critical() || self.cumulative_value_lost > 50.0
    }

    /// Reset cumulative tracking (e.g., for new problem)
    pub fn reset_session(&mut self) {
        self.cumulative_value_lost = 0.0;
    }
}

/// Extension trait for TICE to emit nociception events
pub trait TiceNociception {
    /// Process outcome and emit pain signals
    fn process_outcome_pain(
        &self,
        outcome: &Outcome,
        bridge: &mut TiceNociBridge,
    ) -> Option<PainResponse>;

    /// Check claims that were killed and emit pain for high-value ones
    fn emit_kill_pain(
        &self,
        killed_claims: &[Claim],
        bridge: &mut TiceNociBridge,
    ) -> Vec<PainResponse>;
}

impl TiceNociception for TICE {
    fn process_outcome_pain(
        &self,
        outcome: &Outcome,
        bridge: &mut TiceNociBridge,
    ) -> Option<PainResponse> {
        match outcome {
            Outcome::Stuck(reason) => bridge.process_event(TiceEvent::Stuck {
                reason: reason.clone(),
                iteration: self.iteration(),
            }),
            Outcome::Commit(claim_id) => {
                // Check if this was a low-confidence commit
                if let Some(claim) = self.graph.get(*claim_id) {
                    if claim.probability < 0.5 {
                        bridge.process_event(TiceEvent::ForcedDecision {
                            claim_id: *claim_id,
                            confidence: claim.probability,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Outcome::Continue { kills, .. } if *kills > 3 => {
                // Mass kill event
                bridge.process_event(TiceEvent::MassKill {
                    count: *kills,
                    total_value_lost: *kills as f64 * 2.0, // Estimate
                })
            }
            _ => None,
        }
    }

    fn emit_kill_pain(
        &self,
        killed_claims: &[Claim],
        bridge: &mut TiceNociBridge,
    ) -> Vec<PainResponse> {
        killed_claims
            .iter()
            .filter_map(|claim| {
                bridge.process_event(TiceEvent::ClaimKilled {
                    claim_id: claim.id,
                    claim_content: claim.content.clone(),
                    claim_value: claim.value,
                    was_high_value: claim.value >= bridge.high_value_threshold,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nociception::NociceptorConfig;

    #[test]
    fn test_high_value_kill_causes_pain() {
        let noci = Arc::new(RwLock::new(Nociceptor::new(NociceptorConfig::default())));
        let mut bridge = TiceNociBridge::new(noci.clone());

        let response = bridge.process_event(TiceEvent::ClaimKilled {
            claim_id: ClaimId(1),
            claim_content: "Important hypothesis".to_string(),
            claim_value: 8.0,
            was_high_value: true,
        });

        assert!(response.is_some());
        assert!(bridge.cumulative_value_lost() > 0.0);
    }

    #[test]
    fn test_low_value_kill_no_pain() {
        let noci = Arc::new(RwLock::new(Nociceptor::new(NociceptorConfig::default())));
        let mut bridge = TiceNociBridge::new(noci.clone());

        let response = bridge.process_event(TiceEvent::ClaimKilled {
            claim_id: ClaimId(1),
            claim_content: "Minor claim".to_string(),
            claim_value: 1.0,
            was_high_value: false,
        });

        assert!(response.is_none());
    }

    #[test]
    fn test_all_claims_eliminated_is_critical() {
        let noci = Arc::new(RwLock::new(Nociceptor::new(NociceptorConfig::default())));
        let mut bridge = TiceNociBridge::new(noci.clone());

        let response = bridge.process_event(TiceEvent::AllClaimsEliminated);

        assert!(matches!(response, Some(PainResponse::Stop { .. })));
    }
}
