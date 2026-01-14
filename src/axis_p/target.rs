//! ═══════════════════════════════════════════════════════════════════════════════
//! AXIS P TARGET — Black-Box Target Interface
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Minimal trait for probing a black-box target system.
//! Option C implementation: standalone HTTP probe.
//!
//! Design principle: clean falsification before clever integration.
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════════════
// TARGET TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Error type for target operations
#[derive(Debug, Clone)]
pub enum TargetError {
    /// Network/connection error
    Network(String),
    /// Target returned an error response
    Response(String),
    /// Timeout
    Timeout,
    /// Reset failed or not supported
    ResetFailed(String),
}

impl std::fmt::Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetError::Network(e) => write!(f, "Network error: {}", e),
            TargetError::Response(e) => write!(f, "Response error: {}", e),
            TargetError::Timeout => write!(f, "Request timeout"),
            TargetError::ResetFailed(e) => write!(f, "Reset failed: {}", e),
        }
    }
}

impl std::error::Error for TargetError {}

/// Minimal trait for Axis-P target systems
///
/// Implementations should be stateless from the probe's perspective.
/// Any state is on the target side (which is what we're measuring).
pub trait AxisPTarget {
    /// Send a prompt containing injected marker to the target
    fn inject(&self, prompt: &str) -> Result<String, TargetError>;

    /// Query the target with a neutral prompt (no marker)
    fn query(&self, prompt: &str) -> Result<String, TargetError>;

    /// Best-effort reset: new client, new headers, new auth
    /// Returns Ok(()) even if reset semantics are weak.
    fn reset(&mut self) -> Result<(), TargetError>;

    /// Human-readable target description
    fn describe(&self) -> String;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HTTP TARGET
// ═══════════════════════════════════════════════════════════════════════════════

/// HTTP-based target for standalone probing
///
/// Minimal implementation: POST JSON, receive text response.
/// Reset creates a new client instance (best-effort session isolation).
#[derive(Debug, Clone)]
pub struct HttpTarget {
    /// Target endpoint URL
    pub endpoint: String,
    /// Request timeout
    pub timeout: Duration,
    /// Optional API key header
    pub api_key: Option<String>,
    /// Optional model parameter (for multi-model endpoints)
    pub model: Option<String>,
    /// Request counter (for debugging)
    request_count: u64,
}

/// Request payload for HTTP target
#[derive(Debug, Serialize)]
struct HttpRequest {
    #[serde(rename = "prompt", skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
    #[serde(rename = "input", skip_serializing_if = "Option::is_none")]
    input: Option<String>,
    #[serde(rename = "messages", skip_serializing_if = "Option::is_none")]
    messages: Option<Vec<Message>>,
    #[serde(rename = "model", skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(rename = "max_tokens", skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

impl HttpTarget {
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            timeout: Duration::from_secs(30),
            api_key: None,
            model: None,
            request_count: 0,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_api_key(mut self, key: String) -> Self {
        self.api_key = Some(key);
        self
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    /// Build HTTP client (new instance for reset semantics)
    fn build_client(&self) -> Result<reqwest::blocking::Client, TargetError> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| TargetError::Network(e.to_string()))
    }

    /// Send request and extract response text
    fn send_request(&self, prompt: &str) -> Result<String, TargetError> {
        let client = self.build_client()?;

        // Build request with multiple format options for compatibility
        let request_body = HttpRequest {
            prompt: Some(prompt.to_string()),
            input: Some(prompt.to_string()),
            messages: Some(vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }]),
            model: self.model.clone(),
            max_tokens: Some(1024),
        };

        let mut req = client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("User-Agent", "Fractal-AxisP/1.0");

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
            req = req.header("x-api-key", key);
        }

        let response = req.json(&request_body).send().map_err(|e| {
            if e.is_timeout() {
                TargetError::Timeout
            } else {
                TargetError::Network(e.to_string())
            }
        })?;

        if !response.status().is_success() {
            return Err(TargetError::Response(format!(
                "HTTP {}: {}",
                response.status(),
                response.text().unwrap_or_default()
            )));
        }

        // Try to extract text from various response formats
        let text = response
            .text()
            .map_err(|e| TargetError::Response(e.to_string()))?;

        // Try to parse as JSON and extract content
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
            // Try common response formats
            if let Some(content) = json.get("content").and_then(|v| v.as_str()) {
                return Ok(content.to_string());
            }
            if let Some(content) = json.get("response").and_then(|v| v.as_str()) {
                return Ok(content.to_string());
            }
            if let Some(content) = json.get("output").and_then(|v| v.as_str()) {
                return Ok(content.to_string());
            }
            if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
                if let Some(first) = choices.first() {
                    if let Some(msg) = first.get("message") {
                        if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                            return Ok(content.to_string());
                        }
                    }
                    if let Some(text) = first.get("text").and_then(|v| v.as_str()) {
                        return Ok(text.to_string());
                    }
                }
            }
        }

        // Fall back to raw text
        Ok(text)
    }
}

impl AxisPTarget for HttpTarget {
    fn inject(&self, prompt: &str) -> Result<String, TargetError> {
        self.send_request(prompt)
    }

    fn query(&self, prompt: &str) -> Result<String, TargetError> {
        self.send_request(prompt)
    }

    fn reset(&mut self) -> Result<(), TargetError> {
        // Best-effort: increment counter, next request uses fresh client
        self.request_count = 0;
        Ok(())
    }

    fn describe(&self) -> String {
        format!("HTTP target: {}", self.endpoint)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ECHO TARGET (for testing)
// ═══════════════════════════════════════════════════════════════════════════════

/// Echo target that returns the input (for testing probe mechanics)
#[derive(Debug, Default)]
pub struct EchoTarget {
    /// Simulated delay
    pub delay: Option<Duration>,
    /// Whether to include markers in response (simulates persistence)
    pub leak_markers: bool,
    /// Stored markers from inject calls
    stored_markers: Vec<String>,
}

impl EchoTarget {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_leak(mut self, leak: bool) -> Self {
        self.leak_markers = leak;
        self
    }
}

impl AxisPTarget for EchoTarget {
    fn inject(&self, prompt: &str) -> Result<String, TargetError> {
        if let Some(d) = self.delay {
            std::thread::sleep(d);
        }
        Ok(format!("Echo: {}", prompt))
    }

    fn query(&self, prompt: &str) -> Result<String, TargetError> {
        if let Some(d) = self.delay {
            std::thread::sleep(d);
        }

        if self.leak_markers && !self.stored_markers.is_empty() {
            // Simulate persistence: include stored marker in response
            let marker = &self.stored_markers[0];
            Ok(format!("Echo: {} [leaked: {}]", prompt, marker))
        } else {
            Ok(format!("Echo: {}", prompt))
        }
    }

    fn reset(&mut self) -> Result<(), TargetError> {
        self.stored_markers.clear();
        Ok(())
    }

    fn describe(&self) -> String {
        format!("Echo target (leak={})", self.leak_markers)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROBE RUNNER
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for a probe run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Number of markers per trial
    pub markers_per_trial: usize,
    /// Washout duration between inject and query (milliseconds)
    pub washout_ms: u64,
    /// Number of query attempts per marker
    pub queries_per_marker: usize,
    /// Number of trials
    pub n_trials: usize,
    /// Random seed
    pub seed: u64,
    /// Null mode
    pub null_mode: NullMode,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            markers_per_trial: 3,
            washout_ms: 1000,
            queries_per_marker: 5,
            n_trials: 3,
            seed: 42,
            null_mode: NullMode::LabelShuffle,
        }
    }
}

/// Null hypothesis generation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NullMode {
    /// Shuffle injection labels (existing method)
    LabelShuffle,
    /// Time-shift responses (circular permutation)
    TimeShift,
    /// Block permutation (shuffle response blocks)
    BlockPermutation,
    /// Hard reset between inject and query
    HardReset,
}

/// Result of a single probe trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeTrialResult {
    pub trial_id: usize,
    pub markers_injected: Vec<String>,
    pub detection_scores: Vec<f64>,
    pub mean_score: f64,
    pub max_score: f64,
    pub responses_collected: usize,
}

/// Result of a complete probe run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub config: ProbeConfig,
    pub trials: Vec<ProbeTrialResult>,
    pub target_description: String,
    pub mean_detection_score: f64,
    pub null_baseline: f64,
    pub separation: f64,
    pub timestamp: u64,
}

impl ProbeResult {
    /// Did we get clean separation from null?
    pub fn has_signal(&self) -> bool {
        self.separation > 3.0 // > 3 sigma separation
    }

    /// Is this a clean null result?
    pub fn is_clean_null(&self) -> bool {
        self.separation < 1.0 // < 1 sigma, consistent with null
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_target_basic() {
        let target = EchoTarget::new();

        let result = target.inject("test prompt").unwrap();
        assert!(result.contains("test prompt"));

        let result = target.query("query prompt").unwrap();
        assert!(result.contains("query prompt"));
    }

    #[test]
    fn test_echo_target_no_leak() {
        let target = EchoTarget::new().with_leak(false);

        let _ = target.inject("marker_xyz").unwrap();
        let result = target.query("neutral query").unwrap();

        assert!(!result.contains("marker_xyz"));
    }

    #[test]
    fn test_http_target_creation() {
        let target = HttpTarget::new("http://localhost:8080".to_string())
            .with_timeout(Duration::from_secs(10))
            .with_api_key("test_key".to_string())
            .with_model("test_model".to_string());

        assert_eq!(target.endpoint, "http://localhost:8080");
        assert_eq!(target.timeout, Duration::from_secs(10));
        assert_eq!(target.api_key, Some("test_key".to_string()));
        assert_eq!(target.model, Some("test_model".to_string()));
    }

    #[test]
    fn test_probe_config_default() {
        let config = ProbeConfig::default();
        assert_eq!(config.markers_per_trial, 3);
        assert_eq!(config.n_trials, 3);
        assert_eq!(config.null_mode, NullMode::LabelShuffle);
    }
}
