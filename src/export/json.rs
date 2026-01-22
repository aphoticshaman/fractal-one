//! ═══════════════════════════════════════════════════════════════════════════════
//! JSON Export — Elasticsearch/Sentinel/Chronicle Compatible
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Structured JSON logging for modern SIEM systems:
//! - Elasticsearch-compatible (ECS compliant)
//! - Azure Sentinel compatible
//! - Google Chronicle compatible
//! - Splunk HEC compatible
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::containment::{ContainmentResult, ThreatLevel};
use crate::nociception::{DamageState, PainSignal, PainType};
use crate::observations::ObservationBatch;
use crate::time::TimePoint;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

// ═══════════════════════════════════════════════════════════════════════════════
// JSON EVENT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

/// ECS (Elastic Common Schema) compliant event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonEvent {
    /// Timestamp in ISO 8601 format
    #[serde(rename = "@timestamp")]
    pub timestamp: String,

    /// Event category and metadata
    pub event: EventMetadata,

    /// Agent/host information
    pub agent: AgentInfo,

    /// Log level and message
    pub log: LogInfo,

    /// Source information (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,

    /// User information (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<UserInfo>,

    /// Custom fields specific to Fractal
    pub fractal: FractalFields,

    /// Additional labels
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Event category (e.g., "authentication", "threat", "observation")
    pub category: Vec<String>,
    /// Event type (e.g., "start", "end", "info", "error")
    #[serde(rename = "type")]
    pub event_type: Vec<String>,
    /// Event kind (e.g., "event", "alert", "metric")
    pub kind: String,
    /// Outcome (success/failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// Action taken
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Severity level (1-4: low, medium, high, critical)
    pub severity: u8,
    /// Risk score (0-100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// Agent name
    pub name: String,
    /// Agent type
    #[serde(rename = "type")]
    pub agent_type: String,
    /// Agent version
    pub version: String,
    /// Host name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogInfo {
    /// Log level
    pub level: String,
    /// Log message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Logger name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logger: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Source IP
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ip: Option<String>,
    /// Source port
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Source domain
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    /// User name
    pub name: String,
    /// User ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// User roles
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<String>,
}

/// Fractal-specific fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalFields {
    /// Event signature ID
    pub signature_id: u32,
    /// Component that generated the event
    pub component: String,
    /// Threat level (None, Low, Medium, High, Critical)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threat_level: Option<String>,
    /// Pain/damage metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pain: Option<PainFields>,
    /// Containment metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub containment: Option<ContainmentFields>,
    /// Observation metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observations: Option<Vec<ObservationField>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PainFields {
    pub intensity: f64,
    pub location: String,
    pub pain_type: String,
    pub acute: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub damage_total: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainmentFields {
    pub allowed: bool,
    pub reason: String,
    pub threat_level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator_trust: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationField {
    pub key: String,
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// JSON EXPORTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for JSON export
#[derive(Debug, Clone)]
pub struct JsonExportConfig {
    /// Agent name
    pub agent_name: String,
    /// Agent type
    pub agent_type: String,
    /// Hostname
    pub hostname: Option<String>,
    /// Minimum severity to export (1-4)
    pub min_severity: u8,
    /// Pretty print JSON
    pub pretty: bool,
    /// Include observations
    pub include_observations: bool,
}

impl Default for JsonExportConfig {
    fn default() -> Self {
        Self {
            agent_name: "fractal".to_string(),
            agent_type: "agi_monitor".to_string(),
            hostname: hostname::get().ok().and_then(|h| h.into_string().ok()),
            min_severity: 1,
            pretty: false,
            include_observations: true,
        }
    }
}

/// JSON event exporter
pub struct JsonExporter {
    config: JsonExportConfig,
    event_counter: AtomicU64,
    buffer: std::sync::Mutex<Vec<JsonEvent>>,
}

impl JsonExporter {
    pub fn new(config: JsonExportConfig) -> Self {
        Self {
            config,
            event_counter: AtomicU64::new(0),
            buffer: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create base event with common fields
    fn base_event(&self, signature_id: u32, component: &str) -> JsonEvent {
        let now = TimePoint::now();
        let timestamp = chrono::DateTime::from_timestamp(now.unix_secs() as i64, 0)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_else(|| "unknown".to_string());

        JsonEvent {
            timestamp,
            event: EventMetadata {
                category: vec!["host".to_string()],
                event_type: vec!["info".to_string()],
                kind: "event".to_string(),
                outcome: None,
                action: None,
                severity: 1,
                risk_score: None,
            },
            agent: AgentInfo {
                name: self.config.agent_name.clone(),
                agent_type: self.config.agent_type.clone(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                hostname: self.config.hostname.clone(),
            },
            log: LogInfo {
                level: "info".to_string(),
                message: None,
                logger: Some(component.to_string()),
            },
            source: None,
            user: None,
            fractal: FractalFields {
                signature_id,
                component: component.to_string(),
                threat_level: None,
                pain: None,
                containment: None,
                observations: None,
            },
            labels: HashMap::new(),
        }
    }

    /// Export to JSON string
    pub fn export(&self, event: &JsonEvent) -> Option<String> {
        if event.event.severity < self.config.min_severity {
            return None;
        }

        self.event_counter.fetch_add(1, Ordering::Relaxed);

        if self.config.pretty {
            serde_json::to_string_pretty(event).ok()
        } else {
            serde_json::to_string(event).ok()
        }
    }

    /// Buffer event for batch export
    pub fn buffer_event(&self, event: JsonEvent) {
        if event.event.severity >= self.config.min_severity {
            let mut buffer = self.buffer.lock().unwrap();
            buffer.push(event);
        }
    }

    /// Flush buffered events
    pub fn flush(&self) -> Vec<String> {
        let mut buffer = self.buffer.lock().unwrap();
        let events: Vec<String> = buffer.iter().filter_map(|e| self.export(e)).collect();
        buffer.clear();
        events
    }

    /// Create event from containment result
    pub fn from_containment(&self, result: &ContainmentResult) -> JsonEvent {
        let severity = match result.threat_level {
            ThreatLevel::None => 1,
            ThreatLevel::Low => 2,
            ThreatLevel::Medium => 2,
            ThreatLevel::High => 3,
            ThreatLevel::Critical => 4,
        };

        let mut event = self.base_event(if result.allowed { 201 } else { 200 }, "containment");

        event.event.category = vec!["threat".to_string()];
        event.event.event_type = vec!["indicator".to_string()];
        event.event.kind = if result.allowed { "event" } else { "alert" }.to_string();
        event.event.outcome = Some(if result.allowed { "success" } else { "failure" }.to_string());
        event.event.action = Some(if result.allowed { "allowed" } else { "blocked" }.to_string());
        event.event.severity = severity;
        event.event.risk_score = Some(match result.threat_level {
            ThreatLevel::None => 0.0,
            ThreatLevel::Low => 25.0,
            ThreatLevel::Medium => 50.0,
            ThreatLevel::High => 75.0,
            ThreatLevel::Critical => 100.0,
        });

        event.log.level = match severity {
            1 => "debug",
            2 => "warn",
            3 => "error",
            4 => "critical",
            _ => "info",
        }
        .to_string();
        event.log.message = Some(result.reason.clone());

        event.fractal.threat_level = Some(format!("{:?}", result.threat_level));
        event.fractal.containment = Some(ContainmentFields {
            allowed: result.allowed,
            reason: result.reason.clone(),
            threat_level: format!("{:?}", result.threat_level),
            operator_trust: Some(format!("{:?}", result.operator.trust)),
        });

        event
    }

    /// Create event from pain signal
    pub fn from_pain(&self, signal: &PainSignal) -> JsonEvent {
        let severity = if signal.intensity > 0.8 {
            4
        } else if signal.intensity > 0.5 {
            3
        } else if signal.intensity > 0.2 {
            2
        } else {
            1
        };

        let mut event = self.base_event(300, "nociception");

        event.event.category = vec!["host".to_string()];
        event.event.event_type = vec!["info".to_string()];
        event.event.kind = if signal.requires_response() {
            "alert"
        } else {
            "event"
        }
        .to_string();
        event.event.severity = severity;
        event.event.risk_score = Some((signal.intensity * 100.0) as f64);

        event.log.level = match severity {
            1 => "debug",
            2 => "info",
            3 => "warn",
            4 => "error",
            _ => "info",
        }
        .to_string();

        let pain_type_str = match &signal.pain_type {
            PainType::ConstraintViolation { constraint_id, .. } => {
                format!("constraint_violation:{}", constraint_id)
            }
            PainType::ThermalOverheat { zone, .. } => format!("thermal_overload:{}", zone),
            PainType::CoherenceBreak { .. } => "coherence_break".to_string(),
            PainType::GradientPain { dimension, .. } => format!("gradient:{}", dimension),
            PainType::IntegrityDamage { aspect, .. } => format!("integrity:{}", aspect),
            PainType::ResourceStarvation { resource, .. } => format!("resource:{:?}", resource),
            PainType::QualityCollapse { metric, .. } => format!("quality:{}", metric),
        };

        event.log.message = Some(format!(
            "Pain detected: {} at {} (intensity: {:.2})",
            pain_type_str, signal.location, signal.intensity
        ));

        event.fractal.pain = Some(PainFields {
            intensity: signal.intensity as f64,
            location: signal.location.clone(),
            pain_type: pain_type_str,
            acute: signal.acute,
            damage_total: None,
        });

        event
    }

    /// Create event from damage state
    pub fn from_damage(&self, damage: &DamageState) -> JsonEvent {
        let severity = if damage.is_critical() {
            4
        } else if damage.total > 0.5 {
            3
        } else if damage.total > 0.2 {
            2
        } else {
            1
        };

        let mut event = self.base_event(301, "nociception");

        event.event.category = vec!["host".to_string()];
        event.event.event_type = vec!["info".to_string()];
        event.event.kind = if damage.is_critical() {
            "alert"
        } else {
            "event"
        }
        .to_string();
        event.event.severity = severity;
        event.event.risk_score = Some((damage.total * 100.0) as f64);

        event.log.message = Some(format!(
            "Accumulated damage: {:.2}{}",
            damage.total,
            damage
                .worst_location
                .as_ref()
                .map(|l| format!(" (worst: {})", l))
                .unwrap_or_default()
        ));

        event.fractal.pain = Some(PainFields {
            intensity: 0.0,
            location: damage
                .worst_location
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            pain_type: "accumulated_damage".to_string(),
            acute: false,
            damage_total: Some(damage.total as f64),
        });

        event
    }

    /// Create events from observation batch
    pub fn from_observations(&self, batch: &ObservationBatch) -> Vec<JsonEvent> {
        if !self.config.include_observations {
            return Vec::new();
        }

        let mut events = Vec::new();
        let mut obs_fields = Vec::new();

        for obs in batch.observations.iter() {
            obs_fields.push(ObservationField {
                key: format!("{:?}", obs.key),
                value: obs.value.value,
                source: obs.source.clone(),
            });
        }

        if !obs_fields.is_empty() {
            let mut event = self.base_event(600, "observations");
            event.event.category = vec!["host".to_string()];
            event.event.event_type = vec!["info".to_string()];
            event.event.kind = "metric".to_string();
            event.fractal.observations = Some(obs_fields);
            events.push(event);
        }

        events
    }

    /// Write to NDJSON (newline-delimited JSON) format
    pub fn write_ndjson<W: Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        let events = self.flush();
        let mut bytes = 0;
        for event in &events {
            writeln!(writer, "{}", event)?;
            bytes += event.len() + 1;
        }
        Ok(bytes)
    }

    /// Get event count
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
    fn test_json_event_serialization() {
        let config = JsonExportConfig::default();
        let exporter = JsonExporter::new(config);
        let event = exporter.base_event(100, "test");

        let json = exporter.export(&event).unwrap();
        assert!(json.contains("\"@timestamp\""));
        assert!(json.contains("\"fractal\""));
    }

    #[test]
    fn test_containment_export() {
        let config = JsonExportConfig::default();
        let exporter = JsonExporter::new(config);

        let result = ContainmentResult {
            allowed: false,
            reason: "Test blocked".to_string(),
            threat_level: ThreatLevel::High,
            operator: crate::containment::OperatorProfile::default(),
            intent: crate::containment::IntentAnalysis::default(),
            violations: vec![],
            manipulation_attempts: vec![],
            timestamp: crate::time::TimePoint::now(),
        };

        let event = exporter.from_containment(&result);
        assert_eq!(event.event.severity, 3);
        assert_eq!(event.event.outcome, Some("failure".to_string()));
    }
}

// Hostname module stub for compilation
mod hostname {
    use std::ffi::OsString;

    pub fn get() -> Result<OsString, std::io::Error> {
        Ok(OsString::from("localhost"))
    }
}
