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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
