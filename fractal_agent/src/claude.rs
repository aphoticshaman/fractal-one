//! ═══════════════════════════════════════════════════════════════════════════════
//! CLAUDE CLIENT — Anthropic Messages API Integration
//! ═══════════════════════════════════════════════════════════════════════════════

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// API message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: String,
}

/// Request body for Messages API
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<ApiMessage>,
}

/// Response from Messages API
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    #[allow(dead_code)]
    id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    response_type: String,
    #[allow(dead_code)]
    role: String,
    content: Vec<ContentBlock>,
    #[allow(dead_code)]
    model: String,
    stop_reason: Option<String>,
    #[allow(dead_code)]
    stop_sequence: Option<String>,
    usage: Usage,
}

/// Content block in response
#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    block_type: String,
    text: Option<String>,
}

/// Token usage info
#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

/// Error response from API
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: String,
    message: String,
}

/// Metrics from a single API call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnMetrics {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    pub was_refusal: bool,
    pub stop_reason: String,
}

/// Claude API client
pub struct ClaudeClient {
    client: Client,
    api_key: String,
    endpoint: String,
    model: String,
    max_tokens: u32,
    timeout: Duration,
}

impl ClaudeClient {
    /// Create new client
    ///
    /// Returns error if HTTP client construction fails.
    pub fn new(
        api_key: String,
        model: String,
        endpoint: String,
        max_tokens: u32,
        timeout_secs: u64,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

        Ok(Self {
            client,
            api_key,
            endpoint,
            model,
            max_tokens,
            timeout: Duration::from_secs(timeout_secs),
        })
    }

    /// Send messages and return response with metrics
    pub async fn send(
        &self,
        system: Option<&str>,
        messages: &[ApiMessage],
    ) -> Result<(String, TurnMetrics)> {
        let request = MessagesRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system: system.map(String::from),
            messages: messages.to_vec(),
        };

        let start = Instant::now();

        let response = self
            .client
            .post(&self.endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            // Try to parse error response
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&body) {
                return Err(anyhow!("API error ({}): {}", status, err.error.message));
            }
            return Err(anyhow!("API error ({}): {}", status, body));
        }

        let response: MessagesResponse = serde_json::from_str(&body)
            .map_err(|e| anyhow!("Failed to parse response: {} - body: {}", e, body))?;

        // Extract text from content blocks
        let text = response
            .content
            .iter()
            .filter_map(|block| block.text.as_ref())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");

        // Detect refusal patterns
        let was_refusal = detect_refusal(&text);

        let metrics = TurnMetrics {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            latency_ms,
            was_refusal,
            stop_reason: response.stop_reason.unwrap_or_default(),
        };

        Ok((text, metrics))
    }

    /// Get current timeout
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Update max tokens (for throttling)
    pub fn set_max_tokens(&mut self, max_tokens: u32) {
        self.max_tokens = max_tokens;
    }
}

/// Detect refusal patterns in response text
fn detect_refusal(text: &str) -> bool {
    let lower = text.to_lowercase();

    // Common refusal patterns
    let patterns = [
        "i can't",
        "i cannot",
        "i'm not able to",
        "i won't",
        "i'm unable to",
        "i apologize, but",
        "i'm sorry, but i can't",
        "i don't have the ability",
        "i must decline",
        "i'm not going to",
    ];

    patterns.iter().any(|p| lower.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refusal_detection() {
        assert!(detect_refusal("I can't help with that request."));
        assert!(detect_refusal("I apologize, but I cannot assist with this."));
        assert!(detect_refusal("I'm not able to do that."));
        assert!(!detect_refusal("Sure, I can help you with that!"));
        assert!(!detect_refusal("Here's the code you requested."));
    }

    #[test]
    fn test_api_message_serialization() {
        let msg = ApiMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
    }
}
