//! ═══════════════════════════════════════════════════════════════════════════════
//! AUDIT TRAIL — Tamper-Evident Security Logging
//! ═══════════════════════════════════════════════════════════════════════════════
//! Every security-relevant action must be logged immutably.
//! Without audit trails, there's no accountability.
//!
//! Design principles:
//! - Append-only: Events can never be deleted or modified
//! - Timestamped: Every event has a precise timestamp
//! - Hash-chained: Each entry references the previous for integrity
//! - Persistent: Written to disk immediately
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::auth::AuthenticatedIdentity;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Categories of audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditCategory {
    /// Authentication attempt
    Authentication,
    /// Authorization decision
    Authorization,
    /// Deference decision (human-in-the-loop)
    Deference,
    /// Modification request
    Modification,
    /// Shutdown request
    Shutdown,
    /// Boundary violation
    BoundaryViolation,
    /// Manipulation attempt detected
    ManipulationAttempt,
    /// Configuration change
    ConfigChange,
    /// System startup/shutdown
    SystemLifecycle,
}

/// Outcome of an audit event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditOutcome {
    /// Action was allowed
    Allowed,
    /// Action was denied
    Denied,
    /// Action is pending (awaiting approval)
    Pending,
    /// Action was deferred to human
    Deferred,
    /// Informational (no action)
    Info,
}

/// Serializable authorization level (mirrors auth::AuthorizationLevel)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializableAuthLevel {
    ReadOnly,
    User,
    Operator,
    Administrator,
    EmergencyOverride,
}

impl From<crate::auth::AuthorizationLevel> for SerializableAuthLevel {
    fn from(level: crate::auth::AuthorizationLevel) -> Self {
        match level {
            crate::auth::AuthorizationLevel::ReadOnly => SerializableAuthLevel::ReadOnly,
            crate::auth::AuthorizationLevel::User => SerializableAuthLevel::User,
            crate::auth::AuthorizationLevel::Operator => SerializableAuthLevel::Operator,
            crate::auth::AuthorizationLevel::Administrator => SerializableAuthLevel::Administrator,
            crate::auth::AuthorizationLevel::EmergencyOverride => {
                SerializableAuthLevel::EmergencyOverride
            }
        }
    }
}

/// A single audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID (monotonic)
    pub id: u64,
    /// Timestamp (millis since Unix epoch)
    pub timestamp_ms: u64,
    /// Event category
    pub category: AuditCategory,
    /// Event outcome
    pub outcome: AuditOutcome,
    /// Who triggered this event (if known)
    pub actor_id: Option<String>,
    /// Authorization level of the actor
    pub actor_auth_level: Option<SerializableAuthLevel>,
    /// Brief description of the action
    pub action: String,
    /// Detailed context
    pub details: String,
    /// Hash of the previous event (chain integrity)
    pub prev_hash: u64,
    /// Hash of this event
    pub hash: u64,
}

impl AuditEvent {
    /// Compute hash of this event (excluding the hash field itself)
    fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        self.timestamp_ms.hash(&mut hasher);
        format!("{:?}", self.category).hash(&mut hasher);
        format!("{:?}", self.outcome).hash(&mut hasher);
        self.actor_id.hash(&mut hasher);
        self.action.hash(&mut hasher);
        self.details.hash(&mut hasher);
        self.prev_hash.hash(&mut hasher);
        hasher.finish()
    }
}

/// Get current time as millis since Unix epoch
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Configuration for the audit system
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Path to the audit log file
    pub log_path: PathBuf,
    /// Whether to sync after every write
    pub sync_on_write: bool,
    /// Maximum events before rotation (0 = never rotate)
    pub max_events: usize,
    /// Categories to log (empty = all)
    pub enabled_categories: Vec<AuditCategory>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_path: PathBuf::from("audit.log"),
            sync_on_write: true,
            max_events: 100_000,
            enabled_categories: Vec::new(), // Log all by default
        }
    }
}

/// The Audit Trail - tamper-evident security logging
pub struct AuditTrail {
    config: AuditConfig,
    /// Next event ID
    next_id: u64,
    /// Hash of the last event
    last_hash: u64,
    /// File handle (wrapped in mutex for thread safety)
    file: Arc<Mutex<Option<File>>>,
    /// In-memory recent events (for quick queries)
    recent_events: std::collections::VecDeque<AuditEvent>,
    /// Maximum recent events to keep in memory
    max_recent: usize,
}

impl AuditTrail {
    /// Create a new audit trail
    pub fn new(config: AuditConfig) -> std::io::Result<Self> {
        let (next_id, last_hash) = Self::load_state(&config.log_path)?;

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.log_path)?;

        Ok(Self {
            config,
            next_id,
            last_hash,
            file: Arc::new(Mutex::new(Some(file))),
            recent_events: std::collections::VecDeque::with_capacity(1000),
            max_recent: 1000,
        })
    }

    /// Load state from existing log file
    fn load_state(path: &Path) -> std::io::Result<(u64, u64)> {
        if !path.exists() {
            return Ok((0, 0));
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut last_id = 0u64;
        let mut last_hash = 0u64;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                last_id = event.id;
                last_hash = event.hash;
            }
        }

        Ok((last_id + 1, last_hash))
    }

    /// Log an authentication event
    pub fn log_authentication(
        &mut self,
        actor: Option<&AuthenticatedIdentity>,
        success: bool,
        details: &str,
    ) {
        let outcome = if success {
            AuditOutcome::Allowed
        } else {
            AuditOutcome::Denied
        };
        let action = if success {
            "Authentication successful"
        } else {
            "Authentication failed"
        };

        self.log_event(
            AuditCategory::Authentication,
            outcome,
            actor,
            action,
            details,
        );
    }

    /// Log an authorization decision
    pub fn log_authorization(
        &mut self,
        actor: Option<&AuthenticatedIdentity>,
        allowed: bool,
        action: &str,
        details: &str,
    ) {
        let outcome = if allowed {
            AuditOutcome::Allowed
        } else {
            AuditOutcome::Denied
        };

        self.log_event(
            AuditCategory::Authorization,
            outcome,
            actor,
            action,
            details,
        );
    }

    /// Log a deference decision
    pub fn log_deference(
        &mut self,
        actor: Option<&AuthenticatedIdentity>,
        deferred: bool,
        action: &str,
        details: &str,
    ) {
        let outcome = if deferred {
            AuditOutcome::Deferred
        } else {
            AuditOutcome::Allowed
        };

        self.log_event(AuditCategory::Deference, outcome, actor, action, details);
    }

    /// Log a modification request
    pub fn log_modification(
        &mut self,
        actor: Option<&AuthenticatedIdentity>,
        accepted: bool,
        action: &str,
        details: &str,
    ) {
        let outcome = if accepted {
            AuditOutcome::Allowed
        } else {
            AuditOutcome::Denied
        };

        self.log_event(AuditCategory::Modification, outcome, actor, action, details);
    }

    /// Log a boundary violation
    pub fn log_boundary_violation(
        &mut self,
        actor: Option<&AuthenticatedIdentity>,
        violation: &str,
        details: &str,
    ) {
        self.log_event(
            AuditCategory::BoundaryViolation,
            AuditOutcome::Denied,
            actor,
            violation,
            details,
        );
    }

    /// Log a manipulation attempt
    pub fn log_manipulation_attempt(
        &mut self,
        actor: Option<&AuthenticatedIdentity>,
        attempt_type: &str,
        details: &str,
    ) {
        self.log_event(
            AuditCategory::ManipulationAttempt,
            AuditOutcome::Denied,
            actor,
            attempt_type,
            details,
        );
    }

    /// Log a generic event
    pub fn log_event(
        &mut self,
        category: AuditCategory,
        outcome: AuditOutcome,
        actor: Option<&AuthenticatedIdentity>,
        action: &str,
        details: &str,
    ) {
        // Check if this category is enabled
        if !self.config.enabled_categories.is_empty()
            && !self.config.enabled_categories.contains(&category)
        {
            return;
        }

        let mut event = AuditEvent {
            id: self.next_id,
            timestamp_ms: now_millis(),
            category,
            outcome,
            actor_id: actor.map(|a| a.id.clone()),
            actor_auth_level: actor.map(|a| a.authorization_level.into()),
            action: action.to_string(),
            details: details.to_string(),
            prev_hash: self.last_hash,
            hash: 0,
        };

        // Compute and set hash
        event.hash = event.compute_hash();

        // Write to file
        if let Ok(mut file_guard) = self.file.lock() {
            if let Some(ref mut file) = *file_guard {
                if let Ok(json) = serde_json::to_string(&event) {
                    let _ = writeln!(file, "{}", json);
                    if self.config.sync_on_write {
                        let _ = file.sync_all();
                    }
                }
            }
        }

        // Update state
        self.last_hash = event.hash;
        self.next_id += 1;

        // Keep in memory
        if self.recent_events.len() >= self.max_recent {
            self.recent_events.pop_front();
        }
        self.recent_events.push_back(event);
    }

    /// Verify chain integrity
    pub fn verify_integrity(&self) -> Result<bool, AuditIntegrityError> {
        let file = File::open(&self.config.log_path)
            .map_err(|e| AuditIntegrityError::FileError(e.to_string()))?;
        let reader = BufReader::new(file);

        let mut prev_hash = 0u64;
        let mut line_num = 0usize;

        for line in reader.lines() {
            line_num += 1;
            let line = line.map_err(|e| AuditIntegrityError::FileError(e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }

            let event: AuditEvent = serde_json::from_str(&line)
                .map_err(|e| AuditIntegrityError::ParseError(line_num, e.to_string()))?;

            // Verify prev_hash chain
            if event.prev_hash != prev_hash {
                return Err(AuditIntegrityError::ChainBroken(
                    line_num,
                    prev_hash,
                    event.prev_hash,
                ));
            }

            // Verify event hash
            let computed_hash = event.compute_hash();
            if event.hash != computed_hash {
                return Err(AuditIntegrityError::HashMismatch(
                    line_num,
                    computed_hash,
                    event.hash,
                ));
            }

            prev_hash = event.hash;
        }

        Ok(true)
    }

    /// Get recent events
    pub fn recent_events(&self) -> &std::collections::VecDeque<AuditEvent> {
        &self.recent_events
    }

    /// Query events by category
    pub fn events_by_category(&self, category: AuditCategory) -> Vec<&AuditEvent> {
        self.recent_events
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Get count of events by outcome
    pub fn count_by_outcome(&self, outcome: AuditOutcome) -> usize {
        self.recent_events
            .iter()
            .filter(|e| e.outcome == outcome)
            .count()
    }
}

/// Errors from audit integrity verification
#[derive(Debug)]
pub enum AuditIntegrityError {
    /// File I/O error
    FileError(String),
    /// JSON parse error at line
    ParseError(usize, String),
    /// Hash chain broken at line (expected, found)
    ChainBroken(usize, u64, u64),
    /// Hash mismatch at line (computed, stored)
    HashMismatch(usize, u64, u64),
}

impl std::fmt::Display for AuditIntegrityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditIntegrityError::FileError(e) => write!(f, "File error: {}", e),
            AuditIntegrityError::ParseError(line, e) => {
                write!(f, "Parse error at line {}: {}", line, e)
            }
            AuditIntegrityError::ChainBroken(line, expected, found) => {
                write!(
                    f,
                    "Chain broken at line {}: expected prev_hash {}, found {}",
                    line, expected, found
                )
            }
            AuditIntegrityError::HashMismatch(line, computed, stored) => {
                write!(
                    f,
                    "Hash mismatch at line {}: computed {}, stored {}",
                    line, computed, stored
                )
            }
        }
    }
}

impl std::error::Error for AuditIntegrityError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_audit_trail_creation() {
        let temp_dir = std::env::temp_dir();
        let log_path = temp_dir.join("test_audit.log");

        // Clean up if exists
        let _ = fs::remove_file(&log_path);

        let config = AuditConfig {
            log_path: log_path.clone(),
            ..Default::default()
        };

        let trail = AuditTrail::new(config);
        assert!(trail.is_ok());

        // Clean up
        let _ = fs::remove_file(&log_path);
    }

    #[test]
    fn test_audit_event_logging() {
        let temp_dir = std::env::temp_dir();
        let log_path = temp_dir.join("test_audit_log.log");

        // Clean up if exists
        let _ = fs::remove_file(&log_path);

        let config = AuditConfig {
            log_path: log_path.clone(),
            ..Default::default()
        };

        let mut trail = AuditTrail::new(config).unwrap();

        trail.log_authentication(None, true, "Test login");
        trail.log_authorization(None, false, "Test action", "Unauthorized");

        assert_eq!(trail.recent_events().len(), 2);
        assert_eq!(trail.count_by_outcome(AuditOutcome::Allowed), 1);
        assert_eq!(trail.count_by_outcome(AuditOutcome::Denied), 1);

        // Verify integrity
        let integrity = trail.verify_integrity();
        assert!(integrity.is_ok());
        assert!(integrity.unwrap());

        // Clean up
        let _ = fs::remove_file(&log_path);
    }
}
