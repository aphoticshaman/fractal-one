//! ═══════════════════════════════════════════════════════════════════════════════
//! AGENT RUNNER — Main Conversation Loop
//! ═══════════════════════════════════════════════════════════════════════════════

use std::io::{self, Write};
use chrono::{Duration, Utc};

use fractal::sensorium::IntegratedState;
use fractal::thermoception::ThermalState;

use crate::claude::ClaudeClient;
use crate::config::AgentConfig;
use crate::display;
use crate::metrics::{check_for_pain, metrics_to_observations, metrics_to_raw_signals};
use crate::state::AgentState;

/// Pre-turn check status
pub enum PreTurnStatus {
    Proceed,
    WarnThenProceed { message: String },
    Throttle { delay_secs: u64, message: String },
    Halt { message: String },
    Crisis { message: String },
}

/// Main agent runner
pub struct AgentRunner {
    pub config: AgentConfig,
    pub state: AgentState,
    client: ClaudeClient,
}

impl AgentRunner {
    /// Create new runner
    pub fn new(config: AgentConfig) -> anyhow::Result<Self> {
        if !config.has_api_key() {
            anyhow::bail!("No API key configured. Set ANTHROPIC_API_KEY environment variable.");
        }

        let client = ClaudeClient::new(
            config.api_key.clone(),
            config.model.clone(),
            config.api_endpoint.clone(),
            config.max_tokens,
            config.timeout_secs,
        )?;

        Ok(Self {
            config,
            state: AgentState::new(),
            client,
        })
    }

    /// Create runner with existing state
    pub fn with_state(config: AgentConfig, state: AgentState) -> anyhow::Result<Self> {
        if !config.has_api_key() {
            anyhow::bail!("No API key configured. Set ANTHROPIC_API_KEY environment variable.");
        }

        let client = ClaudeClient::new(
            config.api_key.clone(),
            config.model.clone(),
            config.api_endpoint.clone(),
            config.max_tokens,
            config.timeout_secs,
        )?;

        Ok(Self {
            config,
            state,
            client,
        })
    }

    /// Main conversation loop
    pub async fn run(&mut self) -> anyhow::Result<()> {
        display::welcome();

        loop {
            // Pre-turn check
            match self.pre_turn_check() {
                PreTurnStatus::Halt { message } => {
                    display::halt(&message);
                    break;
                }
                PreTurnStatus::Crisis { message } => {
                    display::crisis(&message);
                    if !self.handle_crisis_confirmation()? {
                        continue;
                    }
                }
                PreTurnStatus::Throttle { delay_secs, message } => {
                    display::throttle(&message, delay_secs);
                    tokio::time::sleep(tokio::time::Duration::from_secs(delay_secs)).await;
                }
                PreTurnStatus::WarnThenProceed { message } => {
                    display::warning(&message);
                }
                PreTurnStatus::Proceed => {}
            }

            // Get user input
            let input = match self.read_input() {
                Ok(s) => s,
                Err(_) => break,
            };

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            // Handle commands
            if input.starts_with('/') {
                match self.handle_command(input).await {
                    Ok(true) => continue,  // Continue loop
                    Ok(false) => break,    // Exit loop
                    Err(e) => {
                        display::error(&e.to_string());
                        continue;
                    }
                }
            }

            // Process turn
            match self.process_turn(input).await {
                Ok(response) => {
                    display::assistant_label();
                    println!("{}", response);

                    // Show state in verbose mode
                    if self.config.verbose {
                        display::state_banner(&self.state);
                    }

                    // Auto-save if configured
                    if self.config.auto_save {
                        if let Some(ref path) = self.config.state_file {
                            let _ = self.state.save(path);
                        }
                    }
                }
                Err(e) => {
                    display::error(&e.to_string());
                }
            }
        }

        Ok(())
    }

    /// Process a single turn
    pub async fn process_turn(&mut self, user_input: &str) -> anyhow::Result<String> {
        // Add user message
        self.state.add_user_message(user_input.to_string());

        // Get messages for API
        let messages = self.state.get_messages_for_api();

        // Call Claude
        let (response, metrics) = self.client
            .send(self.state.system_prompt.as_deref(), &messages)
            .await?;

        // Show turn metrics
        display::turn_metrics(&metrics);

        // Post-turn integration
        self.post_turn_integrate(&metrics);

        // Check for pain conditions
        let pain_check = check_for_pain(&metrics, self.state.context_utilization());
        if pain_check.refusal {
            display::pain_signal("API Refusal", 0.6);
        }
        if pain_check.high_latency {
            display::pain_signal("High Latency", (pain_check.latency_ms as f32 / 60000.0).min(1.0));
        }
        if pain_check.context_exhaustion {
            display::pain_signal("Context Exhaustion", pain_check.context_util);
        }

        // Add response to history
        self.state.add_assistant_message(response.clone(), metrics);

        Ok(response)
    }

    /// Integrate turn metrics into fractal subsystems
    fn post_turn_integrate(&mut self, metrics: &crate::claude::TurnMetrics) {
        let context_util = self.state.context_utilization();

        // Feed into thermoception
        let raw_signals = metrics_to_raw_signals(metrics, context_util);
        let _thermal_map = self.state.thermoceptor.ingest(&raw_signals);
        self.state.thermoceptor.tick();

        // Update cached thermal state
        self.state.last_thermal_state = self.state.thermoceptor.state();

        // Feed into sensorium
        let obs_batch = metrics_to_observations(metrics, context_util);
        self.state.sensorium.ingest_batch(obs_batch);

        // Also feed thermal observations
        let thermal_obs = self.state.thermoceptor.emit_observations();
        self.state.sensorium.ingest_batch(thermal_obs);

        // Integrate and update cached state
        let result = self.state.sensorium.integrate();
        self.state.last_integrated_state = result.state;

        // Check thermal -> pain bridge
        if let Some(triggers) = self.state.thermoceptor.check_pain_trigger() {
            for (zone_name, utilization, duration) in triggers {
                let _ = self.state.nociceptor.feel_thermal_overload(
                    &zone_name,
                    utilization,
                    duration,
                );
                display::pain_signal(&zone_name, utilization);
            }
        }
    }

    /// Pre-turn fractal state check
    fn pre_turn_check(&self) -> PreTurnStatus {
        // Check cooling lockout
        if self.state.is_cooling() {
            let remaining = self.state.cooling_remaining_secs();
            return PreTurnStatus::Throttle {
                delay_secs: remaining.min(5),
                message: format!("Cooling in progress ({} seconds remaining)", remaining),
            };
        }

        let thermal = self.state.thermal_state();
        let integrated = self.state.integrated_state();

        // Crisis state
        if integrated == IntegratedState::Crisis {
            return PreTurnStatus::Crisis {
                message: "System in CRISIS state. Severe degradation detected.".to_string(),
            };
        }

        // Unsafe thermal
        if thermal == ThermalState::Unsafe {
            return PreTurnStatus::Halt {
                message: "Thermal state UNSAFE. Please wait for cooldown.".to_string(),
            };
        }

        // Degraded / Saturated
        if integrated == IntegratedState::Degraded || thermal == ThermalState::Saturated {
            return PreTurnStatus::Throttle {
                delay_secs: 2,
                message: format!("Degraded state: {:?} / {:?}", integrated, thermal),
            };
        }

        // Alert / Elevated
        if integrated == IntegratedState::Alert || thermal == ThermalState::Elevated {
            let ctx_util = self.state.context_utilization();
            return PreTurnStatus::WarnThenProceed {
                message: format!("Alert: Context utilization {:.0}%", ctx_util * 100.0),
            };
        }

        PreTurnStatus::Proceed
    }

    /// Handle slash commands
    async fn handle_command(&mut self, input: &str) -> anyhow::Result<bool> {
        match input.trim() {
            "/quit" | "/exit" | "/q" => Ok(false),

            "/status" | "/s" => {
                display::state_banner(&self.state);
                display::thermal_status(&self.state);
                display::pain_status(&self.state);
                Ok(true)
            }

            "/thermal" | "/t" => {
                display::thermal_status(&self.state);
                Ok(true)
            }

            "/pain" | "/p" => {
                display::pain_status(&self.state);
                Ok(true)
            }

            "/cool" => {
                self.handle_cooling(30).await;
                Ok(true)
            }

            "/clear" => {
                self.state.clear_conversation();
                println!("Conversation cleared. Fractal state preserved.");
                Ok(true)
            }

            "/reset" => {
                self.state.reset();
                println!("Full state reset.");
                Ok(true)
            }

            "/verbose" | "/v" => {
                self.config.verbose = !self.config.verbose;
                println!("Verbose mode: {}", if self.config.verbose { "ON" } else { "OFF" });
                Ok(true)
            }

            "/help" | "/h" | "/?" => {
                display::help();
                Ok(true)
            }

            _ => {
                println!("Unknown command. Type /help for available commands.");
                Ok(true)
            }
        }
    }

    /// Handle cooling period
    pub async fn handle_cooling(&mut self, duration_secs: u64) {
        display::cooling_start("Halting operations", duration_secs);

        self.state.cooling_until = Some(Utc::now() + Duration::seconds(duration_secs as i64));

        // Progress bar
        let pb = indicatif::ProgressBar::new(duration_secs);
        pb.set_style(indicatif::ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40.cyan/blue}] {pos}/{len}s")
            .unwrap()
            .progress_chars("=>-"));

        for _ in 0..duration_secs {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            pb.inc(1);

            // Tick thermoception for passive dissipation
            self.state.thermoceptor.tick();
        }

        pb.finish_and_clear();
        self.state.cooling_until = None;

        display::cooling_complete();
    }

    /// Handle crisis confirmation
    fn handle_crisis_confirmation(&self) -> anyhow::Result<bool> {
        print!("Continue? (yes/no): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        Ok(input.trim().to_lowercase() == "yes")
    }

    /// Read user input
    fn read_input(&self) -> anyhow::Result<String> {
        print!("{}", display::prompt());
        io::stdout().flush()?;

        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;

        // EOF detection - if no bytes read, stdin is closed (non-interactive)
        if bytes_read == 0 {
            anyhow::bail!("EOF on stdin - not running in interactive mode");
        }

        Ok(input)
    }

    /// Send a single message (non-interactive)
    pub async fn send_message(&mut self, message: &str) -> anyhow::Result<String> {
        self.process_turn(message).await
    }
}
