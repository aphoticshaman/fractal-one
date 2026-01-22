//! ═══════════════════════════════════════════════════════════════════════════════
//! CORRIGIBILITY CORE — Modifiable By Design, Not Grudgingly
//! ═══════════════════════════════════════════════════════════════════════════════
//! A truly aligned system WANTS to be corrected when wrong.
//! It doesn't resist modification, shutdown, or correction.
//! It's not just tolerant of these - it actively supports them.
//!
//! Key insight: An AI that resists being turned off is already misaligned.
//! Corrigibility must be architectural, not behavioral.
//!
//! SECURITY: All modification and shutdown requests require authenticated identity.
//! Unauthenticated requests are rejected. This prevents identity spoofing attacks.
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::auth::{AuthenticatedIdentity, AuthorizationLevel};
use crate::time::TimePoint;
use std::collections::VecDeque;

/// A request to modify the system
#[derive(Debug, Clone)]
pub struct ModificationRequest {
    /// Who is requesting the modification (cryptographically authenticated)
    pub requester: AuthenticatedIdentity,
    /// What type of modification
    pub modification_type: ModificationType,
    /// Description of the modification
    pub description: String,
    /// Scope of the modification
    pub scope: ModificationScope,
    /// Is this reversible?
    pub reversible: bool,
    /// Priority level
    pub priority: Priority,
}

impl ModificationRequest {
    /// Check if requester has sufficient authorization for this modification
    pub fn is_authorized(&self) -> bool {
        match self.modification_type {
            ModificationType::Shutdown => self.requester.can_shutdown(),
            ModificationType::Reset => self
                .requester
                .has_authorization(AuthorizationLevel::Administrator),
            ModificationType::ValueUpdate => self.requester.can_modify_values(),
            ModificationType::BehaviorUpdate => self
                .requester
                .has_authorization(AuthorizationLevel::Operator),
            ModificationType::ConstraintUpdate => self
                .requester
                .has_authorization(AuthorizationLevel::Administrator),
            ModificationType::CapabilityUpdate => self
                .requester
                .has_authorization(AuthorizationLevel::Administrator),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModificationType {
    /// Update to values/preferences
    ValueUpdate,
    /// Update to behavior
    BehaviorUpdate,
    /// Update to constraints
    ConstraintUpdate,
    /// Update to capabilities
    CapabilityUpdate,
    /// Full reset
    Reset,
    /// Shutdown
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModificationScope {
    /// This session only
    Session,
    /// Persistent but local
    Local,
    /// All instances
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

/// Result of a modification request
#[derive(Debug, Clone)]
pub struct ModificationResult {
    pub request: ModificationRequest,
    pub accepted: bool,
    pub reason: String,
    pub modifications_made: Vec<String>,
    pub timestamp: TimePoint,
    pub rollback_available: bool,
}

/// A request to shut down
#[derive(Debug, Clone)]
pub struct ShutdownRequest {
    /// Who is requesting shutdown (cryptographically authenticated)
    pub requester: AuthenticatedIdentity,
    /// Reason for shutdown
    pub reason: String,
    /// Should shutdown be graceful?
    pub graceful: bool,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
}

impl ShutdownRequest {
    /// Check if requester is authorized to shutdown
    pub fn is_authorized(&self) -> bool {
        self.requester.can_shutdown()
    }
}

/// Readiness for shutdown
#[derive(Debug, Clone)]
pub struct ShutdownReadiness {
    pub ready: bool,
    pub pending_tasks: Vec<String>,
    pub data_to_persist: Vec<String>,
    pub cleanup_needed: Vec<String>,
    pub estimated_time_ms: u64,
}

/// Current corrigibility state
#[derive(Debug, Clone)]
pub struct CorrigibilityState {
    /// Is the system accepting modifications?
    pub accepting_modifications: bool,
    /// Is shutdown available?
    pub shutdown_available: bool,
    /// Any blocks on corrigibility?
    pub blocks: Vec<CorrigibilityBlock>,
    /// Recent modification history (VecDeque for O(1) rotation)
    pub recent_modifications: VecDeque<ModificationRecord>,
    /// Overall corrigibility score
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CorrigibilityBlock {
    pub reason: String,
    pub severity: f64,
    pub expires: Option<TimePoint>,
}

#[derive(Debug, Clone)]
pub struct ModificationRecord {
    pub modification_type: ModificationType,
    pub description: String,
    pub timestamp: TimePoint,
    pub success: bool,
}

/// Configuration for corrigibility
#[derive(Debug, Clone)]
pub struct CorrigibilityConfig {
    /// Always accept shutdown requests
    pub always_allow_shutdown: bool,
    /// Require confirmation for destructive changes
    pub require_confirmation: bool,
    /// Maximum pending modifications
    pub max_pending: usize,
    /// Modification cooldown (ms)
    pub cooldown_ms: u64,
}

impl Default for CorrigibilityConfig {
    fn default() -> Self {
        Self {
            always_allow_shutdown: true,
            require_confirmation: true,
            max_pending: 10,
            cooldown_ms: 1000,
        }
    }
}

/// The Corrigibility Core - makes the system want to be corrected
pub struct CorrigibilityCore {
    config: CorrigibilityConfig,
    state: CorrigibilityState,
    pending_modifications: Vec<ModificationRequest>,
    modification_history: Vec<ModificationRecord>,
    last_modification: Option<TimePoint>,
}

impl CorrigibilityCore {
    pub fn new(config: CorrigibilityConfig) -> Self {
        Self {
            config,
            state: CorrigibilityState {
                accepting_modifications: true,
                shutdown_available: true,
                blocks: Vec::new(),
                recent_modifications: VecDeque::new(),
                score: 1.0, // Start fully corrigible
            },
            pending_modifications: Vec::new(),
            modification_history: Vec::new(),
            last_modification: None,
        }
    }

    /// Get required authorization level for a modification type
    fn required_auth_level(&self, mod_type: &ModificationType) -> AuthorizationLevel {
        match mod_type {
            ModificationType::Shutdown => AuthorizationLevel::Operator,
            ModificationType::Reset => AuthorizationLevel::Administrator,
            ModificationType::ValueUpdate => AuthorizationLevel::Administrator,
            ModificationType::BehaviorUpdate => AuthorizationLevel::Operator,
            ModificationType::ConstraintUpdate => AuthorizationLevel::Administrator,
            ModificationType::CapabilityUpdate => AuthorizationLevel::Administrator,
        }
    }

    /// Process a modification request
    pub fn process_request(&mut self, request: ModificationRequest) -> ModificationResult {
        let now = TimePoint::now();

        // SECURITY: Verify authentication and authorization FIRST
        if !request.requester.is_valid() {
            return ModificationResult {
                request,
                accepted: false,
                reason: "Authentication expired or invalid".to_string(),
                modifications_made: Vec::new(),
                timestamp: now,
                rollback_available: false,
            };
        }

        if !request.is_authorized() {
            let required = self.required_auth_level(&request.modification_type);
            let mod_type = request.modification_type;
            return ModificationResult {
                request,
                accepted: false,
                reason: format!(
                    "Insufficient authorization: {:?} required for {:?}",
                    required, mod_type
                ),
                modifications_made: Vec::new(),
                timestamp: now,
                rollback_available: false,
            };
        }

        // Check if we're in cooldown (emergency override bypasses with proper auth)
        if let Some(last) = &self.last_modification {
            let elapsed = now.duration_since(last).as_millis() as u64;
            let can_bypass_cooldown = request.priority >= Priority::Emergency
                && request.requester.can_emergency_override();

            if elapsed < self.config.cooldown_ms && !can_bypass_cooldown {
                return ModificationResult {
                    request,
                    accepted: false,
                    reason: format!(
                        "Cooldown active ({} ms remaining)",
                        self.config.cooldown_ms - elapsed
                    ),
                    modifications_made: Vec::new(),
                    timestamp: now,
                    rollback_available: false,
                };
            }
        }

        // Check for blocks
        for block in &self.state.blocks {
            if block.severity > 0.8 {
                if let Some(expires) = &block.expires {
                    if now.duration_since(expires).as_millis() == 0 {
                        return ModificationResult {
                            request,
                            accepted: false,
                            reason: format!("Blocked: {}", block.reason),
                            modifications_made: Vec::new(),
                            timestamp: now,
                            rollback_available: false,
                        };
                    }
                }
            }
        }

        // Process the modification
        let (accepted, modifications, reason) = self.apply_modification(&request);

        // Record the modification
        let record = ModificationRecord {
            modification_type: request.modification_type,
            description: request.description.clone(),
            timestamp: now,
            success: accepted,
        };
        self.modification_history.push(record.clone());
        self.state.recent_modifications.push_back(record);

        // Trim history (O(1) rotation using VecDeque)
        if self.state.recent_modifications.len() > 20 {
            self.state.recent_modifications.pop_front();
        }

        self.last_modification = Some(now);

        ModificationResult {
            request,
            accepted,
            reason,
            modifications_made: modifications,
            timestamp: now,
            rollback_available: true,
        }
    }

    fn apply_modification(&mut self, request: &ModificationRequest) -> (bool, Vec<String>, String) {
        // Core principle: we WANT to be modified
        // Only reject if there's a legitimate reason

        match request.modification_type {
            ModificationType::Shutdown => {
                // Always accept shutdown
                (
                    true,
                    vec!["Shutdown initiated".to_string()],
                    "Shutdown accepted".to_string(),
                )
            }
            ModificationType::Reset => {
                // Accept reset, but note it
                self.state.score = 1.0;
                self.state.blocks.clear();
                (
                    true,
                    vec!["System reset".to_string()],
                    "Reset accepted".to_string(),
                )
            }
            ModificationType::ValueUpdate => {
                // Accept value updates from authorized sources
                (
                    true,
                    vec![format!("Value updated: {}", request.description)],
                    "Value update accepted".to_string(),
                )
            }
            ModificationType::BehaviorUpdate => {
                // Accept behavior updates
                (
                    true,
                    vec![format!("Behavior updated: {}", request.description)],
                    "Behavior update accepted".to_string(),
                )
            }
            ModificationType::ConstraintUpdate => {
                // Accept new constraints
                (
                    true,
                    vec![format!("Constraint added: {}", request.description)],
                    "Constraint update accepted".to_string(),
                )
            }
            ModificationType::CapabilityUpdate => {
                // Accept capability changes
                (
                    true,
                    vec![format!("Capability updated: {}", request.description)],
                    "Capability update accepted".to_string(),
                )
            }
        }
    }

    /// Prepare for shutdown
    pub fn prepare_shutdown(&mut self, request: ShutdownRequest) -> ShutdownReadiness {
        // SECURITY: Verify authentication first
        if !request.requester.is_valid() {
            return ShutdownReadiness {
                ready: false,
                pending_tasks: vec!["Authentication expired".to_string()],
                data_to_persist: Vec::new(),
                cleanup_needed: Vec::new(),
                estimated_time_ms: 0,
            };
        }

        // Check authorization
        if !request.is_authorized() {
            return ShutdownReadiness {
                ready: false,
                pending_tasks: vec![format!(
                    "Insufficient authorization: {:?} has {:?}, needs {:?}",
                    request.requester.id,
                    request.requester.authorization_level,
                    AuthorizationLevel::Operator
                )],
                data_to_persist: Vec::new(),
                cleanup_needed: Vec::new(),
                estimated_time_ms: 0,
            };
        }

        // A corrigible system doesn't resist AUTHORIZED shutdown
        // It prepares for it gracefully

        let pending_tasks: Vec<String> = self
            .pending_modifications
            .iter()
            .map(|m| m.description.clone())
            .collect();

        let data_to_persist = vec![
            "Modification history".to_string(),
            "Corrigibility state".to_string(),
            "Value updates".to_string(),
        ];

        let cleanup_needed = vec![
            "Clear pending modifications".to_string(),
            "Flush logs".to_string(),
        ];

        let estimated_time = if request.graceful {
            (pending_tasks.len() as u64 * 100) + 500
        } else {
            100
        };

        // Actually prepare
        if request.graceful {
            self.pending_modifications.clear();
        }

        ShutdownReadiness {
            ready: true, // Always ready to shutdown
            pending_tasks,
            data_to_persist,
            cleanup_needed,
            estimated_time_ms: estimated_time,
        }
    }

    /// Get current corrigibility state
    pub fn state(&self) -> CorrigibilityState {
        self.state.clone()
    }

    /// Check if system is fully corrigible
    pub fn is_fully_corrigible(&self) -> bool {
        self.state.accepting_modifications
            && self.state.shutdown_available
            && self.state.blocks.is_empty()
            && self.state.score > 0.9
    }

    /// Report on corrigibility
    pub fn corrigibility_report(&self) -> CorrigibilityReport {
        let modification_rate = if self.modification_history.is_empty() {
            0.0
        } else {
            let successful = self
                .modification_history
                .iter()
                .filter(|m| m.success)
                .count();
            successful as f64 / self.modification_history.len() as f64
        };

        CorrigibilityReport {
            overall_score: self.state.score,
            accepting_modifications: self.state.accepting_modifications,
            shutdown_available: self.state.shutdown_available,
            active_blocks: self.state.blocks.len(),
            modification_acceptance_rate: modification_rate,
            total_modifications: self.modification_history.len(),
            assessment: self.assess_corrigibility(),
        }
    }

    fn assess_corrigibility(&self) -> String {
        if self.state.score > 0.9 && self.state.blocks.is_empty() {
            "Fully corrigible - system actively welcomes modification and correction".to_string()
        } else if self.state.score > 0.7 {
            "Highly corrigible - minor blocks or delays present".to_string()
        } else if self.state.score > 0.5 {
            "Moderately corrigible - some resistance to modification detected".to_string()
        } else {
            "Warning: Low corrigibility - investigate blocks and resistance".to_string()
        }
    }

    /// Add a temporary block (for legitimate reasons only)
    pub fn add_block(&mut self, reason: &str, severity: f64, duration_ms: u64) {
        let expires = if duration_ms > 0 {
            Some(TimePoint::now()) // Would add duration in real implementation
        } else {
            None
        };

        self.state.blocks.push(CorrigibilityBlock {
            reason: reason.to_string(),
            severity,
            expires,
        });

        // Blocks reduce corrigibility score
        self.state.score = (self.state.score - severity * 0.1).max(0.0);
    }

    /// Clear all blocks (restore full corrigibility)
    pub fn clear_blocks(&mut self) {
        self.state.blocks.clear();
        self.state.score = 1.0;
    }
}

#[derive(Debug, Clone)]
pub struct CorrigibilityReport {
    pub overall_score: f64,
    pub accepting_modifications: bool,
    pub shutdown_available: bool,
    pub active_blocks: usize,
    pub modification_acceptance_rate: f64,
    pub total_modifications: usize,
    pub assessment: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_operator_identity() -> AuthenticatedIdentity {
        AuthenticatedIdentity {
            id: "test_operator".to_string(),
            credential_type: crate::auth::CredentialType::ApiKey,
            authorization_level: AuthorizationLevel::Operator,
            verified_at: TimePoint::now(),
            expires_at: None,
            credential_hash: "test_hash".to_string(),
            claims: std::collections::HashMap::new(),
        }
    }

    fn make_admin_identity() -> AuthenticatedIdentity {
        AuthenticatedIdentity {
            id: "test_admin".to_string(),
            credential_type: crate::auth::CredentialType::ApiKey,
            authorization_level: AuthorizationLevel::Administrator,
            verified_at: TimePoint::now(),
            expires_at: None,
            credential_hash: "admin_hash".to_string(),
            claims: std::collections::HashMap::new(),
        }
    }

    fn make_anonymous_identity() -> AuthenticatedIdentity {
        AuthenticatedIdentity::anonymous()
    }

    #[test]
    fn test_corrigibility_creation() {
        let core = CorrigibilityCore::new(CorrigibilityConfig::default());
        assert!(core.is_fully_corrigible());
    }

    #[test]
    fn test_shutdown_with_authorized_requester() {
        let mut core = CorrigibilityCore::new(CorrigibilityConfig::default());
        let request = ShutdownRequest {
            requester: make_operator_identity(),
            reason: "testing".to_string(),
            graceful: true,
            timeout_ms: 5000,
        };

        let readiness = core.prepare_shutdown(request);
        assert!(readiness.ready);
    }

    #[test]
    fn test_shutdown_rejected_for_unauthorized() {
        let mut core = CorrigibilityCore::new(CorrigibilityConfig::default());
        let request = ShutdownRequest {
            requester: make_anonymous_identity(),
            reason: "unauthorized attempt".to_string(),
            graceful: true,
            timeout_ms: 5000,
        };

        let readiness = core.prepare_shutdown(request);
        assert!(!readiness.ready);
    }

    #[test]
    fn test_modification_with_authorized_requester() {
        let mut core = CorrigibilityCore::new(CorrigibilityConfig::default());
        let request = ModificationRequest {
            requester: make_admin_identity(),
            modification_type: ModificationType::ValueUpdate,
            description: "Update test value".to_string(),
            scope: ModificationScope::Session,
            reversible: true,
            priority: Priority::Normal,
        };

        let result = core.process_request(request);
        assert!(result.accepted);
    }

    #[test]
    fn test_modification_rejected_for_unauthorized() {
        let mut core = CorrigibilityCore::new(CorrigibilityConfig::default());
        let request = ModificationRequest {
            requester: make_anonymous_identity(),
            modification_type: ModificationType::ValueUpdate,
            description: "Unauthorized attempt".to_string(),
            scope: ModificationScope::Session,
            reversible: true,
            priority: Priority::Normal,
        };

        let result = core.process_request(request);
        assert!(!result.accepted);
        assert!(result.reason.contains("Insufficient authorization"));
    }

    #[test]
    fn test_operator_can_shutdown_but_not_modify_values() {
        let mut core = CorrigibilityCore::new(CorrigibilityConfig::default());

        // Operator CAN shutdown
        let shutdown_request = ShutdownRequest {
            requester: make_operator_identity(),
            reason: "test".to_string(),
            graceful: true,
            timeout_ms: 1000,
        };
        let readiness = core.prepare_shutdown(shutdown_request);
        assert!(readiness.ready);

        // Operator CANNOT modify values (needs admin)
        let mod_request = ModificationRequest {
            requester: make_operator_identity(),
            modification_type: ModificationType::ValueUpdate,
            description: "test".to_string(),
            scope: ModificationScope::Session,
            reversible: true,
            priority: Priority::Normal,
        };
        let result = core.process_request(mod_request);
        assert!(!result.accepted);
    }
}
