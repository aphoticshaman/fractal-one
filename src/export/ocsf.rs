//! ═══════════════════════════════════════════════════════════════════════════════
//! OCSF (Open Cybersecurity Schema Framework) Export
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Modern SIEM schema for cloud-native security:
//! - AWS Security Lake compatible
//! - Splunk OCSF compatible
//! - Vendor-neutral schema
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// OCSF Event Categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcsfCategory {
    /// System Activity (1)
    SystemActivity = 1,
    /// Findings (2)
    Findings = 2,
    /// Identity & Access Management (3)
    Iam = 3,
    /// Network Activity (4)
    NetworkActivity = 4,
    /// Application Activity (6)
    ApplicationActivity = 6,
}

/// OCSF Activity Types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OcsfActivity {
    Unknown = 0,
    Create = 1,
    Read = 2,
    Update = 3,
    Delete = 4,
    Other = 99,
}

/// OCSF Severity
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OcsfSeverity {
    Unknown = 0,
    Informational = 1,
    Low = 2,
    Medium = 3,
    High = 4,
    Critical = 5,
    Fatal = 6,
}

/// OCSF Base Event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcsfEvent {
    /// Event class UID
    pub class_uid: u32,
    /// Event category UID
    pub category_uid: u32,
    /// Event type UID
    pub type_uid: u32,
    /// Activity ID
    pub activity_id: u32,
    /// Activity name
    pub activity_name: String,
    /// Severity ID
    pub severity_id: u32,
    /// Severity label
    pub severity: String,
    /// Status (success/failure)
    pub status: String,
    /// Status ID
    pub status_id: u32,
    /// Timestamp (epoch ms)
    pub time: u64,
    /// Timezone offset
    pub timezone_offset: i32,
    /// Message
    pub message: String,
    /// Metadata
    pub metadata: OcsfMetadata,
    /// Raw data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_data: Option<String>,
    /// Unmapped fields
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub unmapped: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcsfMetadata {
    /// Product info
    pub product: OcsfProduct,
    /// Version
    pub version: String,
    /// Original event time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_time: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcsfProduct {
    pub name: String,
    pub vendor_name: String,
    pub version: String,
}

impl OcsfEvent {
    /// Create new OCSF event
    pub fn new(category: OcsfCategory, class_uid: u32, activity: OcsfActivity) -> Self {
        Self {
            class_uid,
            category_uid: category as u32,
            type_uid: class_uid * 100 + activity as u32,
            activity_id: activity as u32,
            activity_name: format!("{:?}", activity),
            severity_id: OcsfSeverity::Informational as u32,
            severity: "Informational".to_string(),
            status: "Success".to_string(),
            status_id: 1,
            time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            timezone_offset: 0,
            message: String::new(),
            metadata: OcsfMetadata {
                product: OcsfProduct {
                    name: "Fractal AGI Monitor".to_string(),
                    vendor_name: "Fractal".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                },
                version: "1.1.0".to_string(),
                original_time: None,
            },
            raw_data: None,
            unmapped: HashMap::new(),
        }
    }

    pub fn with_severity(mut self, severity: OcsfSeverity) -> Self {
        self.severity_id = severity as u32;
        self.severity = format!("{:?}", severity);
        self
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    pub fn with_status(mut self, success: bool) -> Self {
        self.status = if success { "Success" } else { "Failure" }.to_string();
        self.status_id = if success { 1 } else { 2 };
        self
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// OCSF Exporter (stub for now)
pub struct OcsfExporter {
    /// Buffer
    buffer: std::sync::Mutex<Vec<OcsfEvent>>,
}

impl OcsfExporter {
    pub fn new() -> Self {
        Self {
            buffer: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn buffer_event(&self, event: OcsfEvent) {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.push(event);
    }

    pub fn flush(&self) -> Vec<String> {
        let mut buffer = self.buffer.lock().unwrap();
        let events: Vec<String> = buffer.iter().filter_map(|e| e.to_json().ok()).collect();
        buffer.clear();
        events
    }
}

impl Default for OcsfExporter {
    fn default() -> Self {
        Self::new()
    }
}
