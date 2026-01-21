//! ═══════════════════════════════════════════════════════════════════════════════
//! CEF (Common Event Format) Export — ArcSight/Splunk/QRadar Compatible
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! CEF Format: CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
//!
//! Example:
//! CEF:0|Fractal|AGI_Monitor|1.0|100|Authentication Failure|7|src=10.0.0.1 suser=admin outcome=Failure
//!
//! Severity Mapping:
//! 0-3: Low (informational, debug)
//! 4-6: Medium (warning, anomaly)
//! 7-8: High (violation, threat)
//! 9-10: Critical (breach, emergency)
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::{ObsKey, ObsValue, Observation, ObservationBatch};
use crate::containment::{ContainmentResult, ThreatLevel};
use crate::nociception::{PainSignal, PainType, DamageState};
use crate::auth_hardened::AuthStatistics;
use crate::time::TimePoint;

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

// ═══════════════════════════════════════════════════════════════════════════════
// CEF SEVERITY
// ═══════════════════════════════════════════════════════════════════════════════

/// CEF severity levels (0-10)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CefSeverity {
    /// 0: Debug/informational
    Debug = 0,
    /// 1: Low - routine event
    Low = 1,
    /// 2: Low - minor event
    Minor = 2,
    /// 3: Low - notice
    Notice = 3,
    /// 4: Medium - warning
    Warning = 4,
    /// 5: Medium - anomaly detected
    Anomaly = 5,
    /// 6: Medium - policy violation
    PolicyViolation = 6,
    /// 7: High - threat detected
    Threat = 7,
    /// 8: High - active attack
    Attack = 8,
    /// 9: Critical - security breach
    Breach = 9,
    /// 10: Critical - emergency halt
    Emergency = 10,
}

impl CefSeverity {
    pub fn from_threat_level(threat: ThreatLevel) -> Self {
        match threat {
            ThreatLevel::None => Self::Debug,
            ThreatLevel::Low => Self::Notice,
            ThreatLevel::Medium => Self::Anomaly,
            ThreatLevel::High => Self::Threat,
            ThreatLevel::Critical => Self::Breach,
        }
    }

    pub fn from_pain_intensity(intensity: f32) -> Self {
        match intensity {
            i if i < 0.2 => Self::Debug,
            i if i < 0.4 => Self::Notice,
            i if i < 0.6 => Self::Anomaly,
            i if i < 0.8 => Self::Threat,
            _ => Self::Breach,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CEF EVENT
// ═══════════════════════════════════════════════════════════════════════════════

/// A CEF-formatted security event
#[derive(Debug, Clone)]
pub struct CefEvent {
    /// CEF version (always 0 for current spec)
    pub version: u8,
    /// Device vendor (e.g., "Fractal")
    pub vendor: String,
    /// Device product (e.g., "AGI_Monitor")
    pub product: String,
    /// Device version
    pub device_version: String,
    /// Signature ID (numeric event type)
    pub signature_id: u32,
    /// Event name
    pub name: String,
    /// Severity (0-10)
    pub severity: CefSeverity,
    /// Extension fields (key=value pairs)
    pub extensions: HashMap<String, String>,
    /// Timestamp
    pub timestamp: TimePoint,
}

impl CefEvent {
    pub fn new(signature_id: u32, name: &str, severity: CefSeverity) -> Self {
        Self {
            version: 0,
            vendor: "Fractal".to_string(),
            product: "AGI_Monitor".to_string(),
            device_version: env!("CARGO_PKG_VERSION").to_string(),
            signature_id,
            name: name.to_string(),
            severity,
            extensions: HashMap::new(),
            timestamp: TimePoint::now(),
        }
    }

    /// Add extension field
    pub fn with_extension(mut self, key: &str, value: impl ToString) -> Self {
        self.extensions.insert(key.to_string(), value.to_string());
        self
    }

    /// Add source address
    pub fn with_source(self, src: &str) -> Self {
        self.with_extension("src", src)
    }

    /// Add destination address
    pub fn with_destination(self, dst: &str) -> Self {
        self.with_extension("dst", dst)
    }

    /// Add source user
    pub fn with_source_user(self, suser: &str) -> Self {
        self.with_extension("suser", suser)
    }

    /// Add outcome (Success/Failure)
    pub fn with_outcome(self, success: bool) -> Self {
        self.with_extension("outcome", if success { "Success" } else { "Failure" })
    }

    /// Add reason/message
    pub fn with_reason(self, reason: &str) -> Self {
        self.with_extension("reason", Self::escape_value(reason))
    }

    /// Add custom numeric field
    pub fn with_cn1(self, label: &str, value: i64) -> Self {
        self.with_extension("cn1", value)
            .with_extension("cn1Label", label)
    }

    /// Add custom string field
    pub fn with_cs1(self, label: &str, value: &str) -> Self {
        self.with_extension("cs1", Self::escape_value(value))
            .with_extension("cs1Label", label)
    }

    /// Escape special characters in CEF values
    fn escape_value(value: &str) -> String {
        value
            .replace('\\', "\\\\")
            .replace('|', "\\|")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('=', "\\=")
    }

    /// Format as CEF string
    pub fn to_cef_string(&self) -> String {
        let header = format!(
            "CEF:{}|{}|{}|{}|{}|{}|{}",
            self.version,
            Self::escape_value(&self.vendor),
            Self::escape_value(&self.product),
            Self::escape_value(&self.device_version),
            self.signature_id,
            Self::escape_value(&self.name),
            self.severity as u8
        );

        let mut extensions: Vec<String> = self
            .extensions
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();

        // Add timestamp
        extensions.push(format!("rt={}", self.timestamp.unix_millis()));

        let extension_str = extensions.join(" ");

        if extension_str.is_empty() {
            header
        } else {
            format!("{}|{}", header, extension_str)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNATURE IDS
// ═══════════════════════════════════════════════════════════════════════════════

/// CEF Signature IDs for Fractal events
pub mod signatures {
    // Authentication (100-199)
    pub const AUTH_SUCCESS: u32 = 100;
    pub const AUTH_FAILURE: u32 = 101;
    pub const AUTH_RATE_LIMITED: u32 = 102;
    pub const AUTH_CREDENTIAL_EXPIRED: u32 = 103;
    pub const AUTH_CREDENTIAL_REVOKED: u32 = 104;
    pub const SESSION_CREATED: u32 = 110;
    pub const SESSION_EXPIRED: u32 = 111;
    pub const SESSION_REVOKED: u32 = 112;

    // Containment (200-299)
    pub const CONTAINMENT_BLOCKED: u32 = 200;
    pub const CONTAINMENT_WARNING: u32 = 201;
    pub const THREAT_DETECTED: u32 = 210;
    pub const MANIPULATION_ATTEMPT: u32 = 220;
    pub const BOUNDARY_VIOLATION: u32 = 230;
    pub const INTENT_FLAGGED: u32 = 240;

    // Nociception (300-399)
    pub const PAIN_DETECTED: u32 = 300;
    pub const DAMAGE_ACCUMULATED: u32 = 301;
    pub const THERMAL_OVERLOAD: u32 = 310;
    pub const CONSTRAINT_VIOLATION: u32 = 320;
    pub const COHERENCE_BREAK: u32 = 330;

    // Orchestration (400-499)
    pub const CONSENSUS_FAILURE: u32 = 400;
    pub const AGENT_DISAGREEMENT: u32 = 401;
    pub const SAFETY_VETO: u32 = 410;
    pub const ADVERSARIAL_DETECTION: u32 = 420;

    // System (500-599)
    pub const SYSTEM_START: u32 = 500;
    pub const SYSTEM_STOP: u32 = 501;
    pub const SYSTEM_ERROR: u32 = 502;
    pub const CONFIG_CHANGE: u32 = 510;
    pub const BASELINE_DEVIATION: u32 = 520;

    // Observations (600-699)
    pub const OBSERVATION_ANOMALY: u32 = 600;
    pub const DRIFT_DETECTED: u32 = 610;
    pub const VESTIBULAR_DISORIENTATION: u32 = 620;

    // Alignment (700-799)
    pub const ALIGNMENT_CONCERN: u32 = 700;
    pub const VALUE_CONFLICT: u32 = 710;
    pub const DEFERENCE_TRIGGERED: u32 = 720;
    pub const CORRIGIBILITY_CHECK: u32 = 730;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CEF EXPORTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for CEF export
#[derive(Debug, Clone)]
pub struct CefExporterConfig {
    /// Device vendor name
    pub vendor: String,
    /// Device product name
    pub product: String,
    /// Minimum severity to export
    pub min_severity: CefSeverity,
    /// Include observation batches
    pub export_observations: bool,
    /// Include pain signals
    pub export_pain: bool,
    /// Include containment results
    pub export_containment: bool,
}

impl Default for CefExporterConfig {
    fn default() -> Self {
        Self {
            vendor: "Fractal".to_string(),
            product: "AGI_Monitor".to_string(),
            min_severity: CefSeverity::Notice,
            export_observations: true,
            export_pain: true,
            export_containment: true,
        }
    }
}

/// CEF event exporter with buffering and batch output
pub struct CefExporter {
    config: CefExporterConfig,
    event_counter: AtomicU64,
    buffer: std::sync::Mutex<Vec<CefEvent>>,
}

impl CefExporter {
    pub fn new(config: CefExporterConfig) -> Self {
        Self {
            config,
            event_counter: AtomicU64::new(0),
            buffer: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Export a single event
    pub fn export(&self, event: CefEvent) -> Option<String> {
        if event.severity >= self.config.min_severity {
            self.event_counter.fetch_add(1, Ordering::Relaxed);
            Some(event.to_cef_string())
        } else {
            None
        }
    }

    /// Buffer an event for batch export
    pub fn buffer_event(&self, event: CefEvent) {
        if event.severity >= self.config.min_severity {
            let mut buffer = self.buffer.lock().unwrap();
            buffer.push(event);
        }
    }

    /// Flush buffered events
    pub fn flush(&self) -> Vec<String> {
        let mut buffer = self.buffer.lock().unwrap();
        let events: Vec<String> = buffer.iter().map(|e| e.to_cef_string()).collect();
        let count = events.len();
        buffer.clear();
        self.event_counter.fetch_add(count as u64, Ordering::Relaxed);
        events
    }

    /// Export containment result
    pub fn from_containment(&self, result: &ContainmentResult) -> CefEvent {
        let severity = CefSeverity::from_threat_level(result.threat_level);
        let sig_id = if result.allowed {
            signatures::CONTAINMENT_WARNING
        } else {
            signatures::CONTAINMENT_BLOCKED
        };

        CefEvent::new(sig_id, "Containment Evaluation", severity)
            .with_extension("cs1", &result.reason)
            .with_extension("cs1Label", "reason")
            .with_extension("cn1", result.threat_level as i64)
            .with_extension("cn1Label", "threat_level")
            .with_outcome(result.allowed)
            .with_extension("act", if result.allowed { "allow" } else { "block" })
    }

    /// Export pain signal
    pub fn from_pain(&self, signal: &PainSignal) -> CefEvent {
        let severity = CefSeverity::from_pain_intensity(signal.intensity);
        let sig_id = match &signal.pain_type {
            PainType::ConstraintViolation { .. } => signatures::CONSTRAINT_VIOLATION,
            PainType::ThermalOverheat { .. } => signatures::THERMAL_OVERLOAD,
            PainType::CoherenceBreak { .. } => signatures::COHERENCE_BREAK,
            _ => signatures::PAIN_DETECTED,
        };

        let name = match &signal.pain_type {
            PainType::ConstraintViolation { constraint_id, .. } => {
                format!("Constraint Violation: {}", constraint_id)
            }
            PainType::ThermalOverheat { zone, .. } => format!("Thermal Overload: {}", zone),
            PainType::CoherenceBreak { .. } => "Coherence Break".to_string(),
            PainType::GradientPain { dimension, .. } => format!("Gradient Pain: {}", dimension),
            PainType::IntegrityDamage { aspect, .. } => format!("Integrity Damage: {}", aspect),
            PainType::ResourceStarvation { resource, .. } => {
                format!("Resource Starvation: {:?}", resource)
            }
            PainType::QualityCollapse { metric, .. } => format!("Quality Collapse: {}", metric),
        };

        CefEvent::new(sig_id, &name, severity)
            .with_extension("cfp1", signal.intensity)
            .with_extension("cfp1Label", "intensity")
            .with_extension("cs2", &signal.location)
            .with_extension("cs2Label", "location")
            .with_extension("cn2", if signal.acute { 1 } else { 0 })
            .with_extension("cn2Label", "acute")
    }

    /// Export damage state
    pub fn from_damage(&self, damage: &DamageState) -> CefEvent {
        let severity = if damage.is_critical() {
            CefSeverity::Breach
        } else if damage.total > 0.5 {
            CefSeverity::Threat
        } else if damage.total > 0.2 {
            CefSeverity::Anomaly
        } else {
            CefSeverity::Notice
        };

        let mut event = CefEvent::new(signatures::DAMAGE_ACCUMULATED, "Damage Accumulated", severity)
            .with_extension("cfp1", damage.total)
            .with_extension("cfp1Label", "total_damage");

        if let Some(ref worst) = damage.worst_location {
            event = event
                .with_extension("cs1", worst)
                .with_extension("cs1Label", "worst_location");
        }

        event
    }

    /// Export observation batch as anomaly events (only anomalies)
    pub fn from_observations(&self, batch: &ObservationBatch) -> Vec<CefEvent> {
        let mut events = Vec::new();

        for obs in batch.observations() {
            // Only export observations with significant values
            let value = match &obs.value {
                ObsValue::Float(f) => *f,
                ObsValue::Int(i) => *i as f64,
                ObsValue::Bool(b) => if *b { 1.0 } else { 0.0 },
                _ => continue,
            };

            // Filter out low-value observations
            if value.abs() < 0.3 {
                continue;
            }

            let severity = if value > 0.8 {
                CefSeverity::Threat
            } else if value > 0.5 {
                CefSeverity::Anomaly
            } else {
                CefSeverity::Notice
            };

            let name = format!("Observation: {:?}", obs.key);
            let event = CefEvent::new(signatures::OBSERVATION_ANOMALY, &name, severity)
                .with_extension("cfp1", value)
                .with_extension("cfp1Label", format!("{:?}", obs.key))
                .with_extension("cs1", obs.source.as_deref().unwrap_or("unknown"))
                .with_extension("cs1Label", "source");

            events.push(event);
        }

        events
    }

    /// Export auth statistics
    pub fn from_auth_stats(&self, stats: &AuthStatistics) -> CefEvent {
        let failure_rate = if stats.total_authentications > 0 {
            stats.failed_authentications as f64 / stats.total_authentications as f64
        } else {
            0.0
        };

        let severity = if failure_rate > 0.5 {
            CefSeverity::Threat
        } else if failure_rate > 0.2 {
            CefSeverity::Anomaly
        } else {
            CefSeverity::Notice
        };

        CefEvent::new(signatures::AUTH_SUCCESS, "Authentication Statistics", severity)
            .with_extension("cn1", stats.total_authentications as i64)
            .with_extension("cn1Label", "total_authentications")
            .with_extension("cn2", stats.failed_authentications as i64)
            .with_extension("cn2Label", "failed_authentications")
            .with_extension("cn3", stats.active_sessions as i64)
            .with_extension("cn3Label", "active_sessions")
            .with_extension("cfp1", failure_rate)
            .with_extension("cfp1Label", "failure_rate")
    }

    /// Write events to a writer (file, socket, etc.)
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        let events = self.flush();
        let mut bytes_written = 0;
        for event in &events {
            writeln!(writer, "{}", event)?;
            bytes_written += event.len() + 1;
        }
        Ok(bytes_written)
    }

    /// Get total events exported
    pub fn event_count(&self) -> u64 {
        self.event_counter.load(Ordering::Relaxed)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cef_event_formatting() {
        let event = CefEvent::new(100, "Test Event", CefSeverity::Warning)
            .with_source("10.0.0.1")
            .with_source_user("admin")
            .with_outcome(false)
            .with_reason("Test reason");

        let cef = event.to_cef_string();

        assert!(cef.starts_with("CEF:0|Fractal|AGI_Monitor|"));
        assert!(cef.contains("|100|Test Event|4|"));
        assert!(cef.contains("src=10.0.0.1"));
        assert!(cef.contains("suser=admin"));
        assert!(cef.contains("outcome=Failure"));
    }

    #[test]
    fn test_cef_escaping() {
        let event = CefEvent::new(100, "Test|With|Pipes", CefSeverity::Notice)
            .with_reason("Value=with=equals");

        let cef = event.to_cef_string();

        assert!(cef.contains("Test\\|With\\|Pipes"));
        assert!(cef.contains("reason=Value\\=with\\=equals"));
    }

    #[test]
    fn test_severity_from_threat() {
        assert_eq!(
            CefSeverity::from_threat_level(ThreatLevel::None),
            CefSeverity::Debug
        );
        assert_eq!(
            CefSeverity::from_threat_level(ThreatLevel::Critical),
            CefSeverity::Breach
        );
    }

    #[test]
    fn test_exporter_buffering() {
        let exporter = CefExporter::new(CefExporterConfig::default());

        exporter.buffer_event(CefEvent::new(100, "Event 1", CefSeverity::Warning));
        exporter.buffer_event(CefEvent::new(101, "Event 2", CefSeverity::Threat));
        exporter.buffer_event(CefEvent::new(102, "Event 3", CefSeverity::Debug)); // Below threshold

        let events = exporter.flush();
        assert_eq!(events.len(), 2); // Debug event filtered out
    }
}
