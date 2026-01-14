//! ═══════════════════════════════════════════════════════════════════════════════
//! AGENT CONFIG — Settings and Thresholds
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Anthropic API key
    pub api_key: String,

    /// Model to use (e.g., "claude-sonnet-4-20250514")
    pub model: String,

    /// API endpoint
    pub api_endpoint: String,

    /// Max tokens per response
    pub max_tokens: u32,

    /// Request timeout in seconds
    pub timeout_secs: u64,

    /// Thermal thresholds
    pub thermal: ThermalThresholds,

    /// Pain thresholds
    pub pain: PainThresholds,

    /// Show fractal state after each turn
    pub verbose: bool,

    /// State file path for persistence
    pub state_file: Option<PathBuf>,

    /// Auto-save state after each turn
    pub auto_save: bool,
}

/// Thermal monitoring thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalThresholds {
    /// Warn user when utilization exceeds this (0.0-1.0)
    pub warn_utilization: f32,

    /// Throttle (add delay) when utilization exceeds this
    pub throttle_utilization: f32,

    /// Halt operations when utilization exceeds this
    pub halt_utilization: f32,

    /// Cooldown duration in seconds
    pub cooldown_secs: u64,
}

/// Pain/damage thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PainThresholds {
    /// Warn when pain intensity exceeds this
    pub warn_intensity: f32,

    /// Stop when pain intensity exceeds this
    pub stop_intensity: f32,

    /// Warn when damage accumulation exceeds this
    pub damage_warn: f32,

    /// Critical damage threshold
    pub damage_critical: f32,
}

impl Default for ThermalThresholds {
    fn default() -> Self {
        Self {
            warn_utilization: 0.6,
            throttle_utilization: 0.8,
            halt_utilization: 0.95,
            cooldown_secs: 30,
        }
    }
}

impl Default for PainThresholds {
    fn default() -> Self {
        Self {
            warn_intensity: 0.5,
            stop_intensity: 0.9,
            damage_warn: 0.3,
            damage_critical: 0.8,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "claude-sonnet-4-20250514".to_string(),
            api_endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            max_tokens: 4096,
            timeout_secs: 120,
            thermal: ThermalThresholds::default(),
            pain: PainThresholds::default(),
            verbose: false,
            state_file: None,
            auto_save: true,
        }
    }
}

impl AgentConfig {
    /// Load config from environment and optional file
    pub fn load() -> anyhow::Result<Self> {
        let mut config = Self::default();

        // Load API key from environment
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            config.api_key = key;
        }

        // Override model from environment if set
        if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
            config.model = model;
        }

        // Try to load config file
        let config_path = Self::config_path();
        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)?;
            let file_config: AgentConfig = serde_json::from_str(&contents)?;

            // Merge: file config overrides defaults, env overrides file
            if config.api_key.is_empty() {
                config.api_key = file_config.api_key;
            }
            config.model = file_config.model;
            config.max_tokens = file_config.max_tokens;
            config.timeout_secs = file_config.timeout_secs;
            config.thermal = file_config.thermal;
            config.pain = file_config.pain;
            config.verbose = file_config.verbose;
            config.state_file = file_config.state_file;
            config.auto_save = file_config.auto_save;
        }

        Ok(config)
    }

    /// Create config from environment only
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            config.api_key = key;
        }

        if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
            config.model = model;
        }

        config
    }

    /// Default config file path
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("fractal")
            .join("agent.json")
    }

    /// Default state file path
    pub fn default_state_path() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("fractal")
            .join("agent_state.json")
    }

    /// Save config to file
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Check if API key is configured
    pub fn has_api_key(&self) -> bool {
        !self.api_key.is_empty()
    }
}
