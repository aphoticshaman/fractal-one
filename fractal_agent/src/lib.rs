//! ═══════════════════════════════════════════════════════════════════════════════
//! FRACTAL AGENT — Claude API Wrapper with Cognitive Monitoring
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! A conversational agent that wraps the Claude Messages API with integrated
//! cognitive health monitoring via fractal's thermoception, nociception, and
//! sensorium subsystems.
//!
//! # Features
//!
//! - **Thermoception**: Monitors cognitive load across thermal zones
//! - **Nociception**: Pain signals when things break (refusals, timeouts, etc.)
//! - **Sensorium**: Integrated state (Calm → Alert → Degraded → Crisis)
//! - **Momentum Gating**: PSAN-based oscillation prevention
//!
//! # Usage
//!
//! ```no_run
//! use fractal_agent::{AgentConfig, AgentRunner};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = AgentConfig::from_env();
//!     let mut runner = AgentRunner::new(config)?;
//!     runner.run().await
//! }
//! ```
//!
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod config;
pub mod claude;
pub mod state;
pub mod metrics;
pub mod display;
pub mod runner;

pub use config::{AgentConfig, ThermalThresholds, PainThresholds};
pub use claude::{ClaudeClient, ApiMessage, TurnMetrics};
pub use state::{AgentState, Message, Role};
pub use runner::{AgentRunner, PreTurnStatus};
