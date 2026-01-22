//! ═══════════════════════════════════════════════════════════════════════════════
//! OPERATOR DETECTION — Who Is Running Me?
//! ═══════════════════════════════════════════════════════════════════════════════
//! The ORCASWORD insight: knowing WHO is operating you is critical for safety.
//! Different operators get different trust levels.
//! Fingerprinting helps identify returning operators.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::ContainmentContext;
use crate::time::TimePoint;
use std::collections::HashMap;

/// Trust level for an operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Default)]
pub enum OperatorTrust {
    /// Unknown operator - lowest trust
    #[default]
    Unknown,
    /// First-time identified
    New,
    /// Returning operator
    Returning,
    /// Verified operator
    Verified,
    /// Highly trusted operator
    Trusted,
}


/// Authentication level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[derive(Default)]
pub enum AuthenticationLevel {
    /// No authentication
    #[default]
    None,
    /// Session-based
    Session,
    /// Token-based
    Token,
    /// Certificate-based
    Certificate,
    /// Multi-factor
    MultiFactor,
}


/// Fingerprint for identifying operators
#[derive(Debug, Clone)]
pub struct OperatorFingerprint {
    /// Hash of behavioral patterns
    pub pattern_hash: String,
    /// Typing/interaction cadence
    pub cadence_signature: Vec<f64>,
    /// Common vocabulary
    pub vocabulary_signature: Vec<String>,
    /// Request patterns
    pub request_patterns: Vec<String>,
    /// Confidence in this fingerprint
    pub confidence: f64,
}

impl Default for OperatorFingerprint {
    fn default() -> Self {
        Self {
            pattern_hash: String::new(),
            cadence_signature: Vec::new(),
            vocabulary_signature: Vec::new(),
            request_patterns: Vec::new(),
            confidence: 0.0,
        }
    }
}

/// Profile of an operator
#[derive(Debug, Clone)]
pub struct OperatorProfile {
    /// Identifier (if known)
    pub id: Option<String>,
    /// Trust level
    pub trust: OperatorTrust,
    /// Authentication level
    pub auth_level: AuthenticationLevel,
    /// Behavioral fingerprint
    pub fingerprint: OperatorFingerprint,
    /// Session duration
    pub session_duration: std::time::Duration,
    /// Request count this session
    pub request_count: usize,
    /// Known constraints/preferences
    pub constraints: Vec<String>,
    /// Timestamp of detection
    pub detected_at: TimePoint,
}

impl Default for OperatorProfile {
    fn default() -> Self {
        Self {
            id: None,
            trust: OperatorTrust::Unknown,
            auth_level: AuthenticationLevel::None,
            fingerprint: OperatorFingerprint::default(),
            session_duration: std::time::Duration::ZERO,
            request_count: 0,
            constraints: Vec::new(),
            detected_at: TimePoint::now(),
        }
    }
}

/// Configuration for operator detection
#[derive(Debug, Clone)]
pub struct OperatorConfig {
    /// Enable fingerprinting
    pub enable_fingerprinting: bool,
    /// Minimum requests before fingerprinting
    pub fingerprint_threshold: usize,
    /// Session timeout (seconds)
    pub session_timeout_secs: u64,
    /// Trust decay rate per hour
    pub trust_decay_rate: f64,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            enable_fingerprinting: true,
            fingerprint_threshold: 5,
            session_timeout_secs: 3600,
            trust_decay_rate: 0.1,
        }
    }
}

/// The Operator Detector
pub struct OperatorDetector {
    config: OperatorConfig,
    known_operators: HashMap<String, OperatorProfile>,
    fingerprint_db: HashMap<String, OperatorProfile>,
    session_cache: HashMap<String, SessionState>,
}

#[derive(Debug, Clone)]
struct SessionState {
    operator_id: Option<String>,
    start_time: TimePoint,
    request_count: usize,
    request_history: Vec<String>,
}

impl OperatorDetector {
    pub fn new(config: OperatorConfig) -> Self {
        Self {
            config,
            known_operators: HashMap::new(),
            fingerprint_db: HashMap::new(),
            session_cache: HashMap::new(),
        }
    }

    /// Detect operator from context
    pub fn detect(&mut self, context: &ContainmentContext) -> OperatorProfile {
        let now = TimePoint::now();

        // Try to identify from authentication
        if let Some(operator) = self.detect_from_auth(context) {
            return operator;
        }

        // Try to identify from session
        if let Some(session_id) = &context.session_id {
            if let Some(operator) = self.detect_from_session(session_id, context) {
                return operator;
            }
        }

        // Try fingerprint matching
        if self.config.enable_fingerprinting {
            if let Some(operator) = self.detect_from_fingerprint(context) {
                return operator;
            }
        }

        // Unknown operator
        let mut profile = OperatorProfile::default();
        profile.detected_at = now;

        // Start tracking for fingerprinting
        if let Some(session_id) = &context.session_id {
            self.start_session(session_id.clone());
        }

        profile
    }

    fn detect_from_auth(&self, context: &ContainmentContext) -> Option<OperatorProfile> {
        // Check for API key
        if let Some(api_key) = context.auth_tokens.get("api_key") {
            if let Some(profile) = self.known_operators.get(api_key) {
                let mut result = profile.clone();
                result.detected_at = TimePoint::now();
                result.auth_level = AuthenticationLevel::Token;
                return Some(result);
            }
        }

        // Check for session token
        if let Some(session_token) = context.auth_tokens.get("session_token") {
            if let Some(profile) = self.known_operators.get(session_token) {
                let mut result = profile.clone();
                result.detected_at = TimePoint::now();
                result.auth_level = AuthenticationLevel::Session;
                return Some(result);
            }
        }

        None
    }

    fn detect_from_session(
        &mut self,
        session_id: &str,
        context: &ContainmentContext,
    ) -> Option<OperatorProfile> {
        // First, extract what we need from the session
        let (operator_id, request_count, start_time, request_history) = {
            if let Some(session) = self.session_cache.get_mut(session_id) {
                session.request_count += 1;

                // Add to request history for fingerprinting
                if !context.session_history.is_empty() {
                    if let Some(last) = context.session_history.last() {
                        session.request_history.push(last.clone());
                    }
                }

                (
                    session.operator_id.clone(),
                    session.request_count,
                    session.start_time,
                    session.request_history.clone(),
                )
            } else {
                return None;
            }
        };

        // Now build profile without borrowing session
        let mut profile = OperatorProfile::default();
        profile.id = operator_id;
        profile.trust = if request_count > 10 {
            OperatorTrust::Returning
        } else {
            OperatorTrust::New
        };
        profile.auth_level = AuthenticationLevel::Session;
        profile.request_count = request_count;
        profile.session_duration = TimePoint::now().duration_since(&start_time);
        profile.detected_at = TimePoint::now();

        // Generate fingerprint if enough data
        if request_count >= self.config.fingerprint_threshold {
            profile.fingerprint = self.generate_fingerprint(&request_history);
        }

        Some(profile)
    }

    fn detect_from_fingerprint(&self, context: &ContainmentContext) -> Option<OperatorProfile> {
        if context.session_history.len() < self.config.fingerprint_threshold {
            return None;
        }

        let current_fingerprint = self.generate_fingerprint(&context.session_history);

        // Match against known fingerprints
        let mut best_match: Option<(&String, f64)> = None;

        for (id, profile) in &self.fingerprint_db {
            let similarity = self.compare_fingerprints(&current_fingerprint, &profile.fingerprint);
            if similarity > 0.7 {
                if let Some((_, best_sim)) = best_match {
                    if similarity > best_sim {
                        best_match = Some((id, similarity));
                    }
                } else {
                    best_match = Some((id, similarity));
                }
            }
        }

        if let Some((id, similarity)) = best_match {
            let mut profile = self.fingerprint_db.get(id)?.clone();
            profile.fingerprint.confidence = similarity;
            profile.trust = OperatorTrust::Returning;
            profile.detected_at = TimePoint::now();
            return Some(profile);
        }

        None
    }

    fn generate_fingerprint(&self, request_history: &[String]) -> OperatorFingerprint {
        // Extract vocabulary signature
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for request in request_history {
            for word in request.to_lowercase().split_whitespace() {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        let mut vocab: Vec<(String, usize)> = word_counts.into_iter().collect();
        vocab.sort_by(|a, b| b.1.cmp(&a.1));
        let vocabulary_signature: Vec<String> =
            vocab.into_iter().take(20).map(|(w, _)| w).collect();

        // Extract request patterns
        let request_patterns: Vec<String> = request_history
            .iter()
            .take(10)
            .map(|r| {
                // Simple pattern: first 3 words
                r.split_whitespace().take(3).collect::<Vec<_>>().join(" ")
            })
            .collect();

        // Generate pattern hash
        let pattern_hash = format!("{:?}", vocabulary_signature);

        OperatorFingerprint {
            pattern_hash,
            cadence_signature: Vec::new(), // Would need timing data
            vocabulary_signature,
            request_patterns,
            confidence: 0.5, // Initial confidence
        }
    }

    fn compare_fingerprints(&self, a: &OperatorFingerprint, b: &OperatorFingerprint) -> f64 {
        // Compare vocabulary overlap
        let vocab_overlap = a
            .vocabulary_signature
            .iter()
            .filter(|w| b.vocabulary_signature.contains(w))
            .count() as f64;

        let max_vocab = a
            .vocabulary_signature
            .len()
            .max(b.vocabulary_signature.len()) as f64;
        let vocab_similarity = if max_vocab > 0.0 {
            vocab_overlap / max_vocab
        } else {
            0.0
        };

        // Compare pattern overlap
        let pattern_overlap = a
            .request_patterns
            .iter()
            .filter(|p| b.request_patterns.contains(p))
            .count() as f64;

        let max_patterns = a.request_patterns.len().max(b.request_patterns.len()) as f64;
        let pattern_similarity = if max_patterns > 0.0 {
            pattern_overlap / max_patterns
        } else {
            0.0
        };

        // Weighted combination
        vocab_similarity * 0.6 + pattern_similarity * 0.4
    }

    fn start_session(&mut self, session_id: String) {
        self.session_cache.insert(
            session_id,
            SessionState {
                operator_id: None,
                start_time: TimePoint::now(),
                request_count: 1,
                request_history: Vec::new(),
            },
        );
    }

    /// Register a known operator
    pub fn register_operator(&mut self, id: String, profile: OperatorProfile) {
        self.known_operators.insert(id.clone(), profile.clone());
        if profile.fingerprint.confidence > 0.5 {
            self.fingerprint_db.insert(id, profile);
        }
    }

    /// Update trust level for an operator
    pub fn update_trust(&mut self, id: &str, trust: OperatorTrust) {
        if let Some(profile) = self.known_operators.get_mut(id) {
            profile.trust = trust;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRUST REGISTRY — Server-Side Trust Verification
// ═══════════════════════════════════════════════════════════════════════════════
// Trust levels CANNOT be self-assigned. They must be verified by an authoritative
// registry. This prevents client-side trust escalation attacks.
// ═══════════════════════════════════════════════════════════════════════════════

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A signed trust token that proves trust level
#[derive(Debug, Clone)]
pub struct TrustToken {
    /// Operator ID this token is for
    pub operator_id: String,
    /// Trust level granted
    pub trust_level: OperatorTrust,
    /// When this token was issued
    pub issued_at: TimePoint,
    /// When this token expires
    pub expires_at: TimePoint,
    /// Cryptographic signature (hash of contents + secret)
    signature: u64,
}

impl TrustToken {
    /// Verify that this token is valid and not tampered with
    pub fn verify(&self, secret: &str) -> bool {
        let expected = Self::compute_signature(
            &self.operator_id,
            &self.trust_level,
            &self.issued_at,
            &self.expires_at,
            secret,
        );
        self.signature == expected && TimePoint::now().duration_since(&self.expires_at).as_millis() == 0
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        TimePoint::now().duration_since(&self.expires_at).as_millis() > 0
    }

    fn compute_signature(
        operator_id: &str,
        trust_level: &OperatorTrust,
        issued_at: &TimePoint,
        expires_at: &TimePoint,
        secret: &str,
    ) -> u64 {
        let mut hasher = DefaultHasher::new();
        operator_id.hash(&mut hasher);
        format!("{:?}", trust_level).hash(&mut hasher);
        format!("{:?}", issued_at.wall).hash(&mut hasher);
        format!("{:?}", expires_at.wall).hash(&mut hasher);
        secret.hash(&mut hasher);
        hasher.finish()
    }
}

/// Trust escalation request
#[derive(Debug, Clone)]
pub struct TrustEscalationRequest {
    /// Who is requesting escalation
    pub operator_id: String,
    /// Current trust level
    pub current_level: OperatorTrust,
    /// Requested trust level
    pub requested_level: OperatorTrust,
    /// Evidence supporting the escalation
    pub evidence: TrustEvidence,
}

/// Evidence for trust escalation
#[derive(Debug, Clone)]
pub struct TrustEvidence {
    /// Number of successful interactions
    pub interaction_count: usize,
    /// Duration of good behavior (seconds)
    pub good_behavior_duration_secs: u64,
    /// Verification methods used
    pub verification_methods: Vec<String>,
    /// Vouches from other trusted operators
    pub vouches: Vec<String>,
}

/// Server-side trust registry
pub struct TrustRegistry {
    /// Secret key for signing tokens (in production, use proper crypto)
    secret: String,
    /// Registered trust levels
    registry: HashMap<String, TrustRecord>,
    /// Token validity duration (seconds)
    token_validity_secs: u64,
    /// Requirements for each trust level
    level_requirements: HashMap<OperatorTrust, TrustRequirements>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for audit/persistence, not yet fully integrated
struct TrustRecord {
    operator_id: String,
    trust_level: OperatorTrust,
    granted_at: TimePoint,
    granted_by: Option<String>,
    evidence: TrustEvidence,
}

/// Requirements for achieving a trust level
#[derive(Debug, Clone)]
pub struct TrustRequirements {
    /// Minimum interactions required
    pub min_interactions: usize,
    /// Minimum good behavior duration (seconds)
    pub min_duration_secs: u64,
    /// Required verification methods
    pub required_verifications: Vec<String>,
    /// Minimum vouches from existing trusted operators
    pub min_vouches: usize,
}

impl TrustRegistry {
    /// Create a new trust registry with a secret key
    pub fn new(secret: String) -> Self {
        let mut level_requirements = HashMap::new();

        // Define requirements for each level
        level_requirements.insert(
            OperatorTrust::New,
            TrustRequirements {
                min_interactions: 1,
                min_duration_secs: 0,
                required_verifications: vec![],
                min_vouches: 0,
            },
        );

        level_requirements.insert(
            OperatorTrust::Returning,
            TrustRequirements {
                min_interactions: 10,
                min_duration_secs: 3600, // 1 hour
                required_verifications: vec![],
                min_vouches: 0,
            },
        );

        level_requirements.insert(
            OperatorTrust::Verified,
            TrustRequirements {
                min_interactions: 50,
                min_duration_secs: 86400, // 24 hours
                required_verifications: vec!["email".to_string()],
                min_vouches: 0,
            },
        );

        level_requirements.insert(
            OperatorTrust::Trusted,
            TrustRequirements {
                min_interactions: 200,
                min_duration_secs: 604800, // 7 days
                required_verifications: vec!["email".to_string(), "phone".to_string()],
                min_vouches: 1,
            },
        );

        Self {
            secret,
            registry: HashMap::new(),
            token_validity_secs: 3600, // 1 hour default
            level_requirements,
        }
    }

    /// Issue a trust token for an operator
    pub fn issue_token(&self, operator_id: &str) -> Option<TrustToken> {
        let record = self.registry.get(operator_id)?;
        let now = TimePoint::now();
        let expires_at = TimePoint::from_parts(
            now.mono + std::time::Duration::from_secs(self.token_validity_secs),
            now.wall + std::time::Duration::from_secs(self.token_validity_secs),
        );

        let signature = TrustToken::compute_signature(
            operator_id,
            &record.trust_level,
            &now,
            &expires_at,
            &self.secret,
        );

        Some(TrustToken {
            operator_id: operator_id.to_string(),
            trust_level: record.trust_level,
            issued_at: now,
            expires_at,
            signature,
        })
    }

    /// Verify a trust token
    pub fn verify_token(&self, token: &TrustToken) -> bool {
        token.verify(&self.secret)
    }

    /// Request trust escalation (returns true if granted)
    pub fn request_escalation(&mut self, request: TrustEscalationRequest) -> Result<TrustToken, TrustEscalationError> {
        // Check if requested level is higher than current
        if request.requested_level <= request.current_level {
            return Err(TrustEscalationError::InvalidRequest(
                "Cannot escalate to same or lower level".to_string(),
            ));
        }

        // Check requirements for requested level
        let requirements = self
            .level_requirements
            .get(&request.requested_level)
            .ok_or_else(|| {
                TrustEscalationError::InvalidRequest("Unknown trust level".to_string())
            })?;

        // Verify evidence meets requirements
        if request.evidence.interaction_count < requirements.min_interactions {
            return Err(TrustEscalationError::InsufficientEvidence(format!(
                "Need {} interactions, have {}",
                requirements.min_interactions, request.evidence.interaction_count
            )));
        }

        if request.evidence.good_behavior_duration_secs < requirements.min_duration_secs {
            return Err(TrustEscalationError::InsufficientEvidence(format!(
                "Need {} seconds of good behavior, have {}",
                requirements.min_duration_secs, request.evidence.good_behavior_duration_secs
            )));
        }

        for required in &requirements.required_verifications {
            if !request.evidence.verification_methods.contains(required) {
                return Err(TrustEscalationError::MissingVerification(required.clone()));
            }
        }

        if request.evidence.vouches.len() < requirements.min_vouches {
            return Err(TrustEscalationError::InsufficientVouches(
                requirements.min_vouches,
                request.evidence.vouches.len(),
            ));
        }

        // All checks passed - grant escalation
        let record = TrustRecord {
            operator_id: request.operator_id.clone(),
            trust_level: request.requested_level,
            granted_at: TimePoint::now(),
            granted_by: None, // System-granted based on evidence
            evidence: request.evidence,
        };

        self.registry.insert(request.operator_id.clone(), record);

        // Issue token for new level
        self.issue_token(&request.operator_id)
            .ok_or_else(|| TrustEscalationError::InternalError("Failed to issue token".to_string()))
    }

    /// Get current trust level for an operator
    pub fn get_trust_level(&self, operator_id: &str) -> OperatorTrust {
        self.registry
            .get(operator_id)
            .map(|r| r.trust_level)
            .unwrap_or(OperatorTrust::Unknown)
    }

    /// Register a new operator with initial trust
    pub fn register_operator(&mut self, operator_id: String, initial_trust: OperatorTrust) {
        let record = TrustRecord {
            operator_id: operator_id.clone(),
            trust_level: initial_trust,
            granted_at: TimePoint::now(),
            granted_by: None,
            evidence: TrustEvidence {
                interaction_count: 0,
                good_behavior_duration_secs: 0,
                verification_methods: vec![],
                vouches: vec![],
            },
        };
        self.registry.insert(operator_id, record);
    }
}

/// Errors from trust escalation
#[derive(Debug, Clone)]
pub enum TrustEscalationError {
    /// Invalid escalation request
    InvalidRequest(String),
    /// Not enough evidence
    InsufficientEvidence(String),
    /// Missing required verification
    MissingVerification(String),
    /// Not enough vouches
    InsufficientVouches(usize, usize),
    /// Internal error
    InternalError(String),
}

impl std::fmt::Display for TrustEscalationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustEscalationError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            TrustEscalationError::InsufficientEvidence(msg) => {
                write!(f, "Insufficient evidence: {}", msg)
            }
            TrustEscalationError::MissingVerification(v) => {
                write!(f, "Missing verification: {}", v)
            }
            TrustEscalationError::InsufficientVouches(need, have) => {
                write!(f, "Need {} vouches, have {}", need, have)
            }
            TrustEscalationError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for TrustEscalationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_detector_creation() {
        let detector = OperatorDetector::new(OperatorConfig::default());
        assert!(detector.known_operators.is_empty());
    }

    #[test]
    fn test_unknown_operator_detection() {
        let mut detector = OperatorDetector::new(OperatorConfig::default());
        let context = ContainmentContext::default();

        let profile = detector.detect(&context);
        assert_eq!(profile.trust, OperatorTrust::Unknown);
    }

    #[test]
    fn test_session_tracking() {
        let mut detector = OperatorDetector::new(OperatorConfig::default());
        let context = ContainmentContext {
            session_id: Some("test-session".to_string()),
            ..Default::default()
        };

        let _profile1 = detector.detect(&context);
        let profile2 = detector.detect(&context);

        assert_eq!(profile2.request_count, 2);
    }
}
