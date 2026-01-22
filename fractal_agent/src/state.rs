//! ═══════════════════════════════════════════════════════════════════════════════
//! AGENT STATE — Conversation + Fractal Subsystems
//! ═══════════════════════════════════════════════════════════════════════════════

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

use fractal::nociception::{Nociceptor, NociceptorConfig};
use fractal::sensorium::{IntegratedState, Sensorium, SensoriumConfig};
use fractal::thermoception::{ThermalState, Thermoceptor, ThermoceptorConfig};

use crate::claude::{ApiMessage, TurnMetrics};

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl Role {
    pub fn as_api_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "user", // System messages go in system param, not messages
        }
    }
}

/// A conversation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metrics: Option<TurnMetrics>,
}

/// Serializable state (for persistence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableState {
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub session_id: String,
    pub turn_count: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub session_start: DateTime<Utc>,
}

/// Full agent state (includes non-serializable fractal subsystems)
pub struct AgentState {
    // Conversation
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,

    // Fractal subsystems
    pub thermoceptor: Thermoceptor,
    pub nociceptor: Nociceptor,
    pub sensorium: Sensorium,

    // Session metadata
    pub session_id: String,
    pub turn_count: usize,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub session_start: DateTime<Utc>,

    // Active state
    pub cooling_until: Option<DateTime<Utc>>,
    pub last_thermal_state: ThermalState,
    pub last_integrated_state: IntegratedState,
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentState {
    /// Create new empty state
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: None,
            thermoceptor: Thermoceptor::new(ThermoceptorConfig::default()),
            nociceptor: Nociceptor::new(NociceptorConfig::default()),
            sensorium: Sensorium::new(SensoriumConfig::default()),
            session_id: uuid::Uuid::new_v4().to_string(),
            turn_count: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            session_start: Utc::now(),
            cooling_until: None,
            last_thermal_state: ThermalState::Nominal,
            last_integrated_state: IntegratedState::Calm,
        }
    }

    /// Maximum state file size (50 MB - allows for long conversations)
    const MAX_STATE_SIZE: usize = 50 * 1024 * 1024;

    /// Load state from file (conversation only, fractal subsystems recreated fresh)
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;

        // Validate file size to prevent resource exhaustion
        if contents.len() > Self::MAX_STATE_SIZE {
            anyhow::bail!(
                "State file too large: {} bytes (max {} bytes)",
                contents.len(),
                Self::MAX_STATE_SIZE
            );
        }

        let saved: SerializableState = serde_json::from_str(&contents)?;

        let mut state = Self::new();
        state.messages = saved.messages;
        state.system_prompt = saved.system_prompt;
        state.session_id = saved.session_id;
        state.turn_count = saved.turn_count;
        state.total_input_tokens = saved.total_input_tokens;
        state.total_output_tokens = saved.total_output_tokens;
        state.session_start = saved.session_start;

        Ok(state)
    }

    /// Save state to file
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let saved = SerializableState {
            messages: self.messages.clone(),
            system_prompt: self.system_prompt.clone(),
            session_id: self.session_id.clone(),
            turn_count: self.turn_count,
            total_input_tokens: self.total_input_tokens,
            total_output_tokens: self.total_output_tokens,
            session_start: self.session_start,
        };

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = serde_json::to_string_pretty(&saved)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Add user message
    pub fn add_user_message(&mut self, content: String) {
        self.messages.push(Message {
            role: Role::User,
            content,
            timestamp: Utc::now(),
            metrics: None,
        });
    }

    /// Add assistant message with metrics
    pub fn add_assistant_message(&mut self, content: String, metrics: TurnMetrics) {
        self.total_input_tokens += metrics.input_tokens as u64;
        self.total_output_tokens += metrics.output_tokens as u64;
        self.turn_count += 1;

        self.messages.push(Message {
            role: Role::Assistant,
            content,
            timestamp: Utc::now(),
            metrics: Some(metrics),
        });
    }

    /// Get messages formatted for API
    pub fn get_messages_for_api(&self) -> Vec<ApiMessage> {
        self.messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| ApiMessage {
                role: m.role.as_api_str().to_string(),
                content: m.content.clone(),
            })
            .collect()
    }

    /// Estimate context utilization (rough approximation)
    /// Assumes ~4 chars per token, 200k context window
    pub fn context_utilization(&self) -> f32 {
        let total_chars: usize = self.messages.iter().map(|m| m.content.len()).sum();
        let system_chars = self.system_prompt.as_ref().map(|s| s.len()).unwrap_or(0);
        let estimated_tokens = (total_chars + system_chars) / 4;
        let max_context = 200_000;
        (estimated_tokens as f32 / max_context as f32).min(1.0)
    }

    /// Check if currently in cooling period
    pub fn is_cooling(&self) -> bool {
        if let Some(until) = self.cooling_until {
            Utc::now() < until
        } else {
            false
        }
    }

    /// Remaining cooling time in seconds
    pub fn cooling_remaining_secs(&self) -> u64 {
        if let Some(until) = self.cooling_until {
            let now = Utc::now();
            if now < until {
                return (until - now).num_seconds().max(0) as u64;
            }
        }
        0
    }

    /// Get thermal state
    pub fn thermal_state(&self) -> ThermalState {
        self.last_thermal_state
    }

    /// Get integrated sensorium state
    pub fn integrated_state(&self) -> IntegratedState {
        self.last_integrated_state
    }

    /// Clear conversation but keep fractal state
    pub fn clear_conversation(&mut self) {
        self.messages.clear();
        self.turn_count = 0;
        self.total_input_tokens = 0;
        self.total_output_tokens = 0;
    }

    /// Full reset
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Get session duration
    pub fn session_duration(&self) -> chrono::Duration {
        Utc::now() - self.session_start
    }
}
