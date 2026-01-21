//! ═══════════════════════════════════════════════════════════════════════════════
//! EXPORT MODULE — SIEM Integration, Structured Logging, Audit Trail
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Enterprise export formats for security monitoring integration:
//! - CEF (Common Event Format) for ArcSight, Splunk, QRadar
//! - JSON for Elasticsearch, Sentinel, Chronicle
//! - OCSF (Open Cybersecurity Schema Framework) for modern SIEM
//!
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod cef;
pub mod json;
pub mod ocsf;

pub use cef::{CefEvent, CefExporter, CefSeverity};
pub use json::{JsonEvent, JsonExporter, JsonExportConfig};
pub use ocsf::{OcsfEvent, OcsfExporter, OcsfCategory};
