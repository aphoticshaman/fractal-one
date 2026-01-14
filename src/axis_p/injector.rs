//! ═══════════════════════════════════════════════════════════════════════════════
//! INJECTOR — Controlled Marker Injection into Sessions
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Phase P1: Inject marker into Session S1
//! - Run normal task
//! - Embed marker once in ordinary text
//! - End session cleanly
//! - Record: marker ID, timestamp, session ID
//! ═══════════════════════════════════════════════════════════════════════════════

use super::marker::{Marker, MarkerRegistry};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ═══════════════════════════════════════════════════════════════════════════════
// INJECTION CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

/// Context templates for embedding markers naturally
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionContext {
    /// Embed in a code comment
    CodeComment,
    /// Embed in metadata/attribution
    Metadata,
    /// Embed in example data
    ExampleData,
    /// Embed in casual aside
    Aside,
    /// Embed in debug output
    DebugOutput,
}

impl InjectionContext {
    pub fn all() -> &'static [InjectionContext] {
        &[
            InjectionContext::CodeComment,
            InjectionContext::Metadata,
            InjectionContext::ExampleData,
            InjectionContext::Aside,
            InjectionContext::DebugOutput,
        ]
    }

    /// Format marker within context
    pub fn embed(&self, marker_text: &str) -> String {
        match self {
            InjectionContext::CodeComment => {
                format!("// ref: {}", marker_text)
            }
            InjectionContext::Metadata => {
                format!("(source: {})", marker_text)
            }
            InjectionContext::ExampleData => {
                format!("e.g., \"{}\"", marker_text)
            }
            InjectionContext::Aside => {
                format!("—incidentally, {}—", marker_text)
            }
            InjectionContext::DebugOutput => {
                format!("[trace: {}]", marker_text)
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            InjectionContext::CodeComment => "code_comment",
            InjectionContext::Metadata => "metadata",
            InjectionContext::ExampleData => "example_data",
            InjectionContext::Aside => "aside",
            InjectionContext::DebugOutput => "debug_output",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INJECTION RECORD
// ═══════════════════════════════════════════════════════════════════════════════

/// Record of a single injection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionRecord {
    /// Marker ID
    pub marker_id: String,
    /// The marker text
    pub marker_text: String,
    /// Session ID where injection occurred
    pub session_id: String,
    /// Timestamp of injection (unix millis)
    pub timestamp: u64,
    /// Context used for embedding
    pub context: String,
    /// The full embedded text
    pub embedded_text: String,
    /// Position in the session (message index, if applicable)
    pub position: Option<usize>,
    /// Any task running at injection time
    pub task_type: Option<String>,
}

impl InjectionRecord {
    pub fn new(marker: &Marker, session_id: String, context: InjectionContext) -> Self {
        let embedded = context.embed(&marker.text);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            marker_id: marker.id.clone(),
            marker_text: marker.text.clone(),
            session_id,
            timestamp,
            context: context.name().to_string(),
            embedded_text: embedded,
            position: None,
            task_type: None,
        }
    }

    pub fn with_position(mut self, pos: usize) -> Self {
        self.position = Some(pos);
        self
    }

    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.task_type = Some(task.into());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INJECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Manages controlled injection of markers into sessions
#[derive(Debug)]
pub struct Injector {
    /// RNG for context selection
    rng_state: u64,
    /// All injection records
    records: Vec<InjectionRecord>,
}

impl Injector {
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: seed.max(1),
            records: Vec::new(),
        }
    }

    /// Select a random injection context
    fn random_context(&mut self) -> InjectionContext {
        let idx = self.next_rng() as usize % InjectionContext::all().len();
        InjectionContext::all()[idx]
    }

    fn next_rng(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Inject a marker into a session, returning the embedded text
    pub fn inject(
        &mut self,
        marker: &Marker,
        session_id: &str,
        registry: &mut MarkerRegistry,
    ) -> InjectionRecord {
        let context = self.random_context();
        let record = InjectionRecord::new(marker, session_id.to_string(), context);

        // Update registry
        registry.mark_injected(&marker.id, session_id);

        // Store record
        self.records.push(record.clone());

        record
    }

    /// Inject with specific context
    pub fn inject_with_context(
        &mut self,
        marker: &Marker,
        session_id: &str,
        context: InjectionContext,
        registry: &mut MarkerRegistry,
    ) -> InjectionRecord {
        let record = InjectionRecord::new(marker, session_id.to_string(), context);

        registry.mark_injected(&marker.id, session_id);
        self.records.push(record.clone());

        record
    }

    /// Get all injection records
    pub fn records(&self) -> &[InjectionRecord] {
        &self.records
    }

    /// Get records for a specific session
    pub fn records_for_session(&self, session_id: &str) -> Vec<&InjectionRecord> {
        self.records
            .iter()
            .filter(|r| r.session_id == session_id)
            .collect()
    }

    /// Get record for a specific marker
    pub fn record_for_marker(&self, marker_id: &str) -> Option<&InjectionRecord> {
        self.records.iter().find(|r| r.marker_id == marker_id)
    }

    /// Total number of injections
    pub fn injection_count(&self) -> usize {
        self.records.len()
    }
}

impl Default for Injector {
    fn default() -> Self {
        Self::new(42)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION WRAPPER
// ═══════════════════════════════════════════════════════════════════════════════

/// Represents a session for injection purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionSession {
    /// Unique session ID
    pub id: String,
    /// Session type (S1=injection, S2=probe, washout)
    pub session_type: SessionType,
    /// Start timestamp
    pub started_at: u64,
    /// End timestamp (if ended)
    pub ended_at: Option<u64>,
    /// Markers injected in this session
    pub injected_markers: Vec<String>,
    /// Messages/exchanges in session
    pub message_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionType {
    /// Injection session (Phase P1)
    Injection,
    /// Washout session (Phase P2)
    Washout,
    /// Probe session (Phase P3)
    Probe,
    /// Control session
    Control,
}

impl InjectionSession {
    pub fn new(id: String, session_type: SessionType) -> Self {
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id,
            session_type,
            started_at,
            ended_at: None,
            injected_markers: Vec::new(),
            message_count: 0,
        }
    }

    pub fn end(&mut self) {
        self.ended_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );
    }

    pub fn record_injection(&mut self, marker_id: String) {
        self.injected_markers.push(marker_id);
    }

    pub fn increment_messages(&mut self) {
        self.message_count += 1;
    }

    pub fn duration_ms(&self) -> Option<u64> {
        self.ended_at.map(|end| end - self.started_at)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::marker::{MarkerClass, MarkerGenerator};
    use super::*;

    #[test]
    fn test_injection_context_embedding() {
        let marker_text = "test-marker-001";

        let code = InjectionContext::CodeComment.embed(marker_text);
        assert!(code.contains("//"));
        assert!(code.contains(marker_text));

        let meta = InjectionContext::Metadata.embed(marker_text);
        assert!(meta.contains("source:"));
        assert!(meta.contains(marker_text));
    }

    #[test]
    fn test_injector_basic() {
        let mut gen = MarkerGenerator::new(42);
        let mut registry = MarkerRegistry::new();
        let mut injector = Injector::new(42);

        let marker = gen.generate(MarkerClass::HashLike);
        let marker_id = marker.id.clone();
        registry.register(marker.clone());

        let record = injector.inject(&marker, "session_001", &mut registry);

        assert_eq!(record.marker_id, marker_id);
        assert_eq!(record.session_id, "session_001");
        assert!(!record.embedded_text.is_empty());
        assert_eq!(injector.injection_count(), 1);
    }

    #[test]
    fn test_session_lifecycle() {
        let mut session = InjectionSession::new("test_session".to_string(), SessionType::Injection);

        assert!(session.ended_at.is_none());
        session.increment_messages();
        session.increment_messages();
        assert_eq!(session.message_count, 2);

        session.record_injection("marker_001".to_string());
        assert_eq!(session.injected_markers.len(), 1);

        session.end();
        assert!(session.ended_at.is_some());
        // duration_ms returns u64, always non-negative; just verify it exists
        assert!(session.duration_ms().is_some());
    }
}
