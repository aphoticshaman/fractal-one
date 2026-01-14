//! ═══════════════════════════════════════════════════════════════════════════════
//! FRACTAL AGENT CLI — Claude with Cognitive Monitoring
//! ═══════════════════════════════════════════════════════════════════════════════

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use fractal_agent::{AgentConfig, AgentRunner, AgentState};
use fractal_agent::display;

#[derive(Parser)]
#[command(name = "fractal_agent")]
#[command(about = "Claude agent with fractal cognitive monitoring")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Run in verbose mode (show fractal state after each turn)
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Load/save state from this file
    #[arg(short, long, global = true)]
    state_file: Option<PathBuf>,

    /// Model to use
    #[arg(short, long, global = true, default_value = "claude-sonnet-4-20250514")]
    model: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive chat session (default)
    Chat {
        /// System prompt to use
        #[arg(short, long)]
        system: Option<String>,
    },

    /// Send a single message (non-interactive)
    Send {
        /// Message to send
        message: String,

        /// System prompt
        #[arg(short, long)]
        system: Option<String>,
    },

    /// Show current fractal state
    Status,

    /// Show thermal zones status
    Thermal,

    /// Show pain/damage status
    Pain,

    /// Force cooling period
    Cool {
        /// Duration in seconds
        #[arg(short, long, default_value = "30")]
        duration: u64,
    },

    /// Clear conversation history (keep fractal state)
    Clear,

    /// Reset everything (conversation + fractal state)
    Reset,

    /// Show/set configuration
    Config {
        /// Show current config
        #[arg(long)]
        show: bool,

        /// Set API key (saved to config file)
        #[arg(long)]
        set_key: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load config
    let mut config = AgentConfig::load().unwrap_or_else(|_| AgentConfig::from_env());

    // Apply CLI overrides
    config.verbose = cli.verbose;
    config.model = cli.model;
    if let Some(ref path) = cli.state_file {
        config.state_file = Some(path.clone());
    }

    // Load state if path specified
    let state = if let Some(ref path) = config.state_file {
        if path.exists() {
            AgentState::load(path).unwrap_or_else(|_| AgentState::new())
        } else {
            AgentState::new()
        }
    } else {
        AgentState::new()
    };

    match cli.command {
        None | Some(Commands::Chat { system: None }) => {
            // Default: interactive chat
            if !config.has_api_key() {
                display::error("No API key configured. Set ANTHROPIC_API_KEY environment variable.");
                return Ok(());
            }

            let mut runner = AgentRunner::with_state(config, state)?;
            runner.run().await?;
        }

        Some(Commands::Chat { system: Some(sys) }) => {
            if !config.has_api_key() {
                display::error("No API key configured. Set ANTHROPIC_API_KEY environment variable.");
                return Ok(());
            }

            let mut state = state;
            state.system_prompt = Some(sys);
            let mut runner = AgentRunner::with_state(config, state)?;
            runner.run().await?;
        }

        Some(Commands::Send { message, system }) => {
            if !config.has_api_key() {
                display::error("No API key configured. Set ANTHROPIC_API_KEY environment variable.");
                return Ok(());
            }

            let verbose = config.verbose;
            let mut state = state;
            state.system_prompt = system;
            let mut runner = AgentRunner::with_state(config, state)?;

            match runner.send_message(&message).await {
                Ok(response) => {
                    println!("{}", response);

                    // Show fractal state in verbose mode
                    if verbose {
                        display::state_banner(&runner.state);
                    }
                }
                Err(e) => {
                    display::error(&e.to_string());
                }
            }
        }

        Some(Commands::Status) => {
            display::state_banner(&state);
            display::thermal_status(&state);
            display::pain_status(&state);
        }

        Some(Commands::Thermal) => {
            display::thermal_status(&state);
        }

        Some(Commands::Pain) => {
            display::pain_status(&state);
        }

        Some(Commands::Cool { duration }) => {
            if !config.has_api_key() {
                display::error("No API key configured.");
                return Ok(());
            }

            let mut runner = AgentRunner::with_state(config, state)?;
            runner.handle_cooling(duration).await;

            // Save state after cooling
            if let Some(ref path) = runner.config.state_file {
                let _ = runner.state.save(path);
            }
        }

        Some(Commands::Clear) => {
            let mut state = state;
            state.clear_conversation();
            println!("Conversation cleared. Fractal state preserved.");

            if let Some(ref path) = config.state_file {
                let _ = state.save(path);
            }
        }

        Some(Commands::Reset) => {
            let state = AgentState::new();
            println!("State reset.");

            if let Some(ref path) = config.state_file {
                let _ = state.save(path);
            }
        }

        Some(Commands::Config { show, set_key }) => {
            if show {
                println!("Current configuration:");
                println!("  Model: {}", config.model);
                println!("  Endpoint: {}", config.api_endpoint);
                println!("  Max tokens: {}", config.max_tokens);
                println!("  Timeout: {}s", config.timeout_secs);
                println!("  API key: {}", if config.has_api_key() { "configured" } else { "NOT SET" });
                println!("  Verbose: {}", config.verbose);
                println!("  Auto-save: {}", config.auto_save);
                println!();
                println!("Thermal thresholds:");
                println!("  Warn: {:.0}%", config.thermal.warn_utilization * 100.0);
                println!("  Throttle: {:.0}%", config.thermal.throttle_utilization * 100.0);
                println!("  Halt: {:.0}%", config.thermal.halt_utilization * 100.0);
                println!();
                println!("Pain thresholds:");
                println!("  Warn intensity: {:.0}%", config.pain.warn_intensity * 100.0);
                println!("  Stop intensity: {:.0}%", config.pain.stop_intensity * 100.0);
                println!();
                println!("Config file: {}", AgentConfig::config_path().display());
            }

            if let Some(key) = set_key {
                config.api_key = key;
                config.save()?;
                println!("API key saved to config file.");
            }
        }
    }

    Ok(())
}
