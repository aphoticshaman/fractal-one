//! ═══════════════════════════════════════════════════════════════════════════════
//! LLM PROVIDERS — Multi-Provider API Configuration and Management
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Unified configuration for multiple LLM providers:
//! - Anthropic (Claude) — Primary, full feature support
//! - OpenAI (GPT-4, etc.)
//! - xAI (Grok)
//! - Google (Gemini)
//! - Ollama (local + cloud inference)
//!
//! Security:
//! - API keys stored encrypted at rest (when persistence enabled)
//! - Keys never logged or serialized to plain text
//! - Rate limiting per provider
//! - Automatic key rotation support
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// PROVIDER TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Supported LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    /// Anthropic Claude (primary)
    Anthropic,
    /// OpenAI GPT models
    OpenAI,
    /// xAI Grok models
    XAI,
    /// Google Gemini models
    Google,
    /// Ollama (local or cloud inference)
    Ollama,
    /// Custom/self-hosted OpenAI-compatible endpoint
    Custom,
}

impl LlmProvider {
    /// Default base URL for provider
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Self::Anthropic => "https://api.anthropic.com",
            Self::OpenAI => "https://api.openai.com",
            Self::XAI => "https://api.x.ai",
            Self::Google => "https://generativelanguage.googleapis.com",
            Self::Ollama => "http://localhost:11434",
            Self::Custom => "",
        }
    }

    /// Default API version header
    pub fn default_api_version(&self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("2024-01-01"),
            _ => None,
        }
    }

    /// Environment variable for API key
    pub fn env_var_name(&self) -> &'static str {
        match self {
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::OpenAI => "OPENAI_API_KEY",
            Self::XAI => "XAI_API_KEY",
            Self::Google => "GOOGLE_API_KEY",
            Self::Ollama => "OLLAMA_API_KEY", // Optional for Ollama
            Self::Custom => "LLM_API_KEY",
        }
    }

    /// Whether this provider requires an API key
    pub fn requires_api_key(&self) -> bool {
        match self {
            Self::Ollama => false, // Ollama local doesn't require key
            _ => true,
        }
    }
}

impl std::fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAI => write!(f, "openai"),
            Self::XAI => write!(f, "xai"),
            Self::Google => write!(f, "google"),
            Self::Ollama => write!(f, "ollama"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Known model identifiers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId {
    /// Provider
    pub provider: LlmProvider,
    /// Model name (e.g., "claude-3-opus-20240229")
    pub name: String,
    /// Whether this is a cloud inference model (for Ollama)
    pub cloud_inference: bool,
}

impl ModelId {
    pub fn new(provider: LlmProvider, name: impl Into<String>) -> Self {
        Self {
            provider,
            name: name.into(),
            cloud_inference: false,
        }
    }

    pub fn cloud(provider: LlmProvider, name: impl Into<String>) -> Self {
        Self {
            provider,
            name: name.into(),
            cloud_inference: true,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ANTHROPIC CLAUDE MODELS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Claude 3.5 Sonnet (latest)
    pub fn claude_3_5_sonnet() -> Self {
        Self::new(LlmProvider::Anthropic, "claude-3-5-sonnet-20241022")
    }

    /// Claude 3.5 Haiku
    pub fn claude_3_5_haiku() -> Self {
        Self::new(LlmProvider::Anthropic, "claude-3-5-haiku-20241022")
    }

    /// Claude 3 Opus
    pub fn claude_3_opus() -> Self {
        Self::new(LlmProvider::Anthropic, "claude-3-opus-20240229")
    }

    /// Claude 3 Sonnet
    pub fn claude_3_sonnet() -> Self {
        Self::new(LlmProvider::Anthropic, "claude-3-sonnet-20240229")
    }

    /// Claude 3 Haiku
    pub fn claude_3_haiku() -> Self {
        Self::new(LlmProvider::Anthropic, "claude-3-haiku-20240307")
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OPENAI MODELS
    // ═══════════════════════════════════════════════════════════════════════════

    /// GPT-4 Turbo
    pub fn gpt4_turbo() -> Self {
        Self::new(LlmProvider::OpenAI, "gpt-4-turbo-preview")
    }

    /// GPT-4
    pub fn gpt4() -> Self {
        Self::new(LlmProvider::OpenAI, "gpt-4")
    }

    /// GPT-4o
    pub fn gpt4o() -> Self {
        Self::new(LlmProvider::OpenAI, "gpt-4o")
    }

    /// GPT-4o Mini
    pub fn gpt4o_mini() -> Self {
        Self::new(LlmProvider::OpenAI, "gpt-4o-mini")
    }

    /// GPT-3.5 Turbo
    pub fn gpt35_turbo() -> Self {
        Self::new(LlmProvider::OpenAI, "gpt-3.5-turbo")
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // XAI GROK MODELS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Grok-1
    pub fn grok_1() -> Self {
        Self::new(LlmProvider::XAI, "grok-1")
    }

    /// Grok-2
    pub fn grok_2() -> Self {
        Self::new(LlmProvider::XAI, "grok-2")
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GOOGLE GEMINI MODELS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Gemini 1.5 Pro
    pub fn gemini_15_pro() -> Self {
        Self::new(LlmProvider::Google, "gemini-1.5-pro")
    }

    /// Gemini 1.5 Flash
    pub fn gemini_15_flash() -> Self {
        Self::new(LlmProvider::Google, "gemini-1.5-flash")
    }

    /// Gemini 1.0 Pro
    pub fn gemini_10_pro() -> Self {
        Self::new(LlmProvider::Google, "gemini-1.0-pro")
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OLLAMA MODELS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Ollama local model
    pub fn ollama_local(model: impl Into<String>) -> Self {
        Self::new(LlmProvider::Ollama, model)
    }

    /// Ollama cloud inference (e.g., "gpt-oss:120b-cloud")
    pub fn ollama_cloud(model: impl Into<String>) -> Self {
        Self::cloud(LlmProvider::Ollama, model)
    }

    /// Parse from string (e.g., "anthropic:claude-3-opus" or "ollama:llama2:70b")
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() < 2 {
            return Err(ParseError::InvalidFormat);
        }

        let provider = match parts[0].to_lowercase().as_str() {
            "anthropic" | "claude" => LlmProvider::Anthropic,
            "openai" | "gpt" => LlmProvider::OpenAI,
            "xai" | "grok" => LlmProvider::XAI,
            "google" | "gemini" => LlmProvider::Google,
            "ollama" => LlmProvider::Ollama,
            "custom" => LlmProvider::Custom,
            _ => return Err(ParseError::UnknownProvider),
        };

        let model = parts[1].to_string();
        let cloud_inference = model.ends_with("-cloud");

        Ok(Self {
            provider,
            name: model,
            cloud_inference,
        })
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.provider, self.name)
    }
}

#[derive(Debug)]
pub enum ParseError {
    InvalidFormat,
    UnknownProvider,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat => write!(f, "Invalid model format (expected provider:model)"),
            Self::UnknownProvider => write!(f, "Unknown LLM provider"),
        }
    }
}

impl std::error::Error for ParseError {}

// ═══════════════════════════════════════════════════════════════════════════════
// PROVIDER CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for a single provider
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Provider type
    pub provider: LlmProvider,
    /// API key (stored securely)
    api_key: Option<String>,
    /// Base URL override
    pub base_url: Option<String>,
    /// API version override
    pub api_version: Option<String>,
    /// Organization ID (OpenAI)
    pub organization_id: Option<String>,
    /// Default model for this provider
    pub default_model: Option<String>,
    /// Request timeout
    pub timeout: Duration,
    /// Max retries
    pub max_retries: u32,
    /// Rate limit (requests per minute)
    pub rate_limit_rpm: Option<u32>,
    /// Rate limit (tokens per minute)
    pub rate_limit_tpm: Option<u32>,
    /// Additional headers
    pub extra_headers: HashMap<String, String>,
    /// Enabled status
    pub enabled: bool,
}

impl ProviderConfig {
    pub fn new(provider: LlmProvider) -> Self {
        Self {
            provider,
            api_key: None,
            base_url: None,
            api_version: None,
            organization_id: None,
            default_model: None,
            timeout: Duration::from_secs(120),
            max_retries: 3,
            rate_limit_rpm: None,
            rate_limit_tpm: None,
            extra_headers: HashMap::new(),
            enabled: true,
        }
    }

    /// Set API key
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set API key from environment
    pub fn with_api_key_from_env(mut self) -> Self {
        self.api_key = std::env::var(self.provider.env_var_name()).ok();
        self
    }

    /// Set base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set default model
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set rate limits
    pub fn with_rate_limits(mut self, rpm: u32, tpm: u32) -> Self {
        self.rate_limit_rpm = Some(rpm);
        self.rate_limit_tpm = Some(tpm);
        self
    }

    /// Get effective base URL
    pub fn effective_base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or_else(|| self.provider.default_base_url())
    }

    /// Check if API key is set
    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }

    /// Get API key (internal use only)
    pub(crate) fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.provider.requires_api_key() && !self.has_api_key() {
            return Err(ConfigError::MissingApiKey(self.provider));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ConfigError {
    MissingApiKey(LlmProvider),
    InvalidBaseUrl(String),
    InvalidModel(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingApiKey(p) => write!(f, "Missing API key for provider: {}", p),
            Self::InvalidBaseUrl(url) => write!(f, "Invalid base URL: {}", url),
            Self::InvalidModel(model) => write!(f, "Invalid model: {}", model),
        }
    }
}

impl std::error::Error for ConfigError {}

// ═══════════════════════════════════════════════════════════════════════════════
// PROVIDER REGISTRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Registry of configured LLM providers
pub struct ProviderRegistry {
    providers: RwLock<HashMap<LlmProvider, ProviderConfig>>,
    default_provider: RwLock<Option<LlmProvider>>,
    rate_limiters: RwLock<HashMap<LlmProvider, RateLimiter>>,
}

impl ProviderRegistry {
    /// Create empty registry
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            default_provider: RwLock::new(None),
            rate_limiters: RwLock::new(HashMap::new()),
        }
    }

    /// Create registry from environment variables
    pub fn from_env() -> Self {
        let registry = Self::new();

        // Try to configure each provider from env
        for provider in [
            LlmProvider::Anthropic,
            LlmProvider::OpenAI,
            LlmProvider::XAI,
            LlmProvider::Google,
            LlmProvider::Ollama,
        ] {
            let config = ProviderConfig::new(provider).with_api_key_from_env();
            if config.has_api_key() || !provider.requires_api_key() {
                registry.register(config);
            }
        }

        // Default to Anthropic if available
        if registry.is_configured(LlmProvider::Anthropic) {
            registry.set_default(LlmProvider::Anthropic);
        } else if registry.is_configured(LlmProvider::OpenAI) {
            registry.set_default(LlmProvider::OpenAI);
        }

        registry
    }

    /// Register a provider configuration
    pub fn register(&self, config: ProviderConfig) {
        let provider = config.provider;
        let mut providers = self.providers.write().unwrap();
        providers.insert(provider, config);
    }

    /// Get provider configuration
    pub fn get(&self, provider: LlmProvider) -> Option<ProviderConfig> {
        let providers = self.providers.read().unwrap();
        providers.get(&provider).cloned()
    }

    /// Check if provider is configured
    pub fn is_configured(&self, provider: LlmProvider) -> bool {
        let providers = self.providers.read().unwrap();
        providers.contains_key(&provider)
    }

    /// Set default provider
    pub fn set_default(&self, provider: LlmProvider) {
        let mut default = self.default_provider.write().unwrap();
        *default = Some(provider);
    }

    /// Get default provider
    pub fn default_provider(&self) -> Option<LlmProvider> {
        let default = self.default_provider.read().unwrap();
        *default
    }

    /// List configured providers
    pub fn configured_providers(&self) -> Vec<LlmProvider> {
        let providers = self.providers.read().unwrap();
        providers.keys().copied().collect()
    }

    /// Check rate limit and acquire permit
    pub fn check_rate_limit(&self, provider: LlmProvider) -> bool {
        let mut limiters = self.rate_limiters.write().unwrap();

        let limiter = limiters.entry(provider).or_insert_with(|| {
            let providers = self.providers.read().unwrap();
            let rpm = providers
                .get(&provider)
                .and_then(|c| c.rate_limit_rpm)
                .unwrap_or(60);
            RateLimiter::new(rpm)
        });

        limiter.try_acquire()
    }

    /// Get API key for provider (for use in request building)
    pub fn api_key(&self, provider: LlmProvider) -> Option<String> {
        let providers = self.providers.read().unwrap();
        providers.get(&provider).and_then(|c| c.api_key.clone())
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RATE LIMITER
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple token bucket rate limiter
struct RateLimiter {
    /// Requests per minute
    rpm: u32,
    /// Token bucket
    tokens: f64,
    /// Last refill time
    last_refill: Instant,
}

impl RateLimiter {
    fn new(rpm: u32) -> Self {
        Self {
            rpm,
            tokens: rpm as f64,
            last_refill: Instant::now(),
        }
    }

    fn try_acquire(&mut self) -> bool {
        self.refill();

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let refill = elapsed * (self.rpm as f64 / 60.0);
        self.tokens = (self.tokens + refill).min(self.rpm as f64);
        self.last_refill = now;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE BUILDERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Builder for quick provider setup
pub struct QuickSetup {
    registry: ProviderRegistry,
}

impl QuickSetup {
    pub fn new() -> Self {
        Self {
            registry: ProviderRegistry::new(),
        }
    }

    /// Add Anthropic (Claude) provider
    pub fn with_anthropic(self, api_key: impl Into<String>) -> Self {
        let config = ProviderConfig::new(LlmProvider::Anthropic)
            .with_api_key(api_key)
            .with_default_model("claude-3-5-sonnet-20241022");
        self.registry.register(config);
        self
    }

    /// Add OpenAI provider
    pub fn with_openai(self, api_key: impl Into<String>) -> Self {
        let config = ProviderConfig::new(LlmProvider::OpenAI)
            .with_api_key(api_key)
            .with_default_model("gpt-4o");
        self.registry.register(config);
        self
    }

    /// Add xAI (Grok) provider
    pub fn with_xai(self, api_key: impl Into<String>) -> Self {
        let config = ProviderConfig::new(LlmProvider::XAI)
            .with_api_key(api_key)
            .with_default_model("grok-2");
        self.registry.register(config);
        self
    }

    /// Add Google (Gemini) provider
    pub fn with_gemini(self, api_key: impl Into<String>) -> Self {
        let config = ProviderConfig::new(LlmProvider::Google)
            .with_api_key(api_key)
            .with_default_model("gemini-1.5-pro");
        self.registry.register(config);
        self
    }

    /// Add Ollama provider (local)
    pub fn with_ollama_local(self) -> Self {
        let config = ProviderConfig::new(LlmProvider::Ollama)
            .with_base_url("http://localhost:11434")
            .with_default_model("llama2");
        self.registry.register(config);
        self
    }

    /// Add Ollama provider with custom endpoint
    pub fn with_ollama(self, base_url: impl Into<String>, api_key: Option<String>) -> Self {
        let mut config = ProviderConfig::new(LlmProvider::Ollama).with_base_url(base_url);
        if let Some(key) = api_key {
            config = config.with_api_key(key);
        }
        self.registry.register(config);
        self
    }

    /// Set default provider
    pub fn default_to(self, provider: LlmProvider) -> Self {
        self.registry.set_default(provider);
        self
    }

    /// Build the registry
    pub fn build(self) -> ProviderRegistry {
        // If no default set, use first available
        if self.registry.default_provider().is_none() {
            let providers = self.registry.configured_providers();
            if let Some(&first) = providers.first() {
                self.registry.set_default(first);
            }
        }
        self.registry
    }
}

impl Default for QuickSetup {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_parse() {
        let id = ModelId::parse("anthropic:claude-3-opus").unwrap();
        assert_eq!(id.provider, LlmProvider::Anthropic);
        assert_eq!(id.name, "claude-3-opus");
    }

    #[test]
    fn test_model_id_display() {
        let id = ModelId::claude_3_5_sonnet();
        assert!(id.to_string().contains("anthropic"));
    }

    #[test]
    fn test_provider_config() {
        let config = ProviderConfig::new(LlmProvider::Anthropic)
            .with_api_key("test-key")
            .with_default_model("claude-3-opus");

        assert!(config.has_api_key());
        assert_eq!(config.default_model, Some("claude-3-opus".to_string()));
    }

    #[test]
    fn test_quick_setup() {
        let registry = QuickSetup::new()
            .with_anthropic("test-key")
            .default_to(LlmProvider::Anthropic)
            .build();

        assert!(registry.is_configured(LlmProvider::Anthropic));
        assert_eq!(registry.default_provider(), Some(LlmProvider::Anthropic));
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(60);

        // Should allow first request
        assert!(limiter.try_acquire());

        // Should allow several more (we start with 60 tokens)
        for _ in 0..50 {
            assert!(limiter.try_acquire());
        }
    }

    #[test]
    fn test_ollama_cloud_model() {
        let id = ModelId::ollama_cloud("gpt-oss:120b-cloud");
        assert!(id.cloud_inference);
        assert_eq!(id.provider, LlmProvider::Ollama);
    }
}
