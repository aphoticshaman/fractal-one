//! ═══════════════════════════════════════════════════════════════════════════════
//! DISPLAY — Terminal Output Formatting
//! ═══════════════════════════════════════════════════════════════════════════════

use colored::*;

use fractal::sensorium::IntegratedState;
use fractal::thermoception::ThermalState;

use crate::metrics::{format_latency, format_tokens};
use crate::claude::TurnMetrics;
use crate::state::AgentState;

/// Display welcome banner
pub fn welcome() {
    println!();
    println!("{}", "═".repeat(60).cyan());
    println!("{}", "  FRACTAL AGENT — Claude with Cognitive Monitoring".bright_white().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();
    println!("Commands: {} {} {} {} {}",
        "/status".yellow(),
        "/thermal".yellow(),
        "/pain".yellow(),
        "/cool".yellow(),
        "/quit".yellow()
    );
    println!();
}

/// Display state banner (shown in verbose mode after each turn)
pub fn state_banner(state: &AgentState) {
    let thermal = state.thermal_state();
    let integrated = state.integrated_state();

    let thermal_color = thermal_state_color(thermal);
    let integrated_color = integrated_state_color(integrated);

    println!();
    println!("{}", "─".repeat(60).bright_black());
    println!("{} {} │ {} {} │ {} {} │ {} {}",
        "Thermal:".bright_black(),
        format!("{:?}", thermal).color(thermal_color),
        "State:".bright_black(),
        format!("{:?}", integrated).color(integrated_color),
        "Turns:".bright_black(),
        state.turn_count.to_string().white(),
        "Tokens:".bright_black(),
        format_tokens((state.total_input_tokens + state.total_output_tokens) as u32).white(),
    );
    println!("{}", "─".repeat(60).bright_black());
}

/// Display turn metrics
pub fn turn_metrics(metrics: &TurnMetrics) {
    let latency_color = if metrics.latency_ms > 30000 {
        "red"
    } else if metrics.latency_ms > 10000 {
        "yellow"
    } else {
        "green"
    };

    println!();
    println!("{} {} │ {} in/{} out │ {}",
        "Latency:".bright_black(),
        format_latency(metrics.latency_ms).color(latency_color),
        format_tokens(metrics.input_tokens).bright_black(),
        format_tokens(metrics.output_tokens).bright_black(),
        if metrics.was_refusal { "REFUSAL".red().to_string() } else { "OK".green().to_string() },
    );
}

/// Display thermal zones status
pub fn thermal_status(state: &AgentState) {
    println!();
    println!("{}", "═══ THERMAL STATUS ═══".cyan().bold());
    println!();

    let thermal = state.thermal_state();
    let color = thermal_state_color(thermal);

    println!("{} {}", "State:".white(), format!("{:?}", thermal).color(color).bold());
    println!();

    // Show zone utilizations (we'd need to expose these from thermoceptor)
    // For now, show context utilization
    let ctx_util = state.context_utilization();
    thermal_bar("Context", ctx_util);

    println!();
}

/// Display a thermal utilization bar
pub fn thermal_bar(name: &str, utilization: f32) {
    let bar_width: usize = 30;
    let filled = (utilization * bar_width as f32) as usize;

    let color = if utilization > 0.9 {
        "red"
    } else if utilization > 0.7 {
        "yellow"
    } else if utilization > 0.5 {
        "bright_yellow"
    } else {
        "green"
    };

    let bar = format!(
        "[{}{}] {:5.1}%",
        "█".repeat(filled).color(color),
        "░".repeat(bar_width.saturating_sub(filled)).bright_black(),
        utilization * 100.0
    );

    println!("  {:12} {}", name.white(), bar);
}

/// Display pain/damage status
pub fn pain_status(state: &AgentState) {
    println!();
    println!("{}", "═══ PAIN STATUS ═══".red().bold());
    println!();

    let damage = state.nociceptor.damage_state();
    let in_pain = state.nociceptor.in_pain();

    println!("{} {}", "In Pain:".white(),
        if in_pain { "YES".red().bold() } else { "No".green() });
    println!("{} {:.1}%", "Total Damage:".white(),
        damage.total * 100.0);

    if let Some(loc) = &damage.worst_location {
        println!("{} {}", "Worst Location:".white(), loc.cyan());
    }

    println!();
}

/// Display warning message
pub fn warning(message: &str) {
    println!();
    println!("{} {}", "WARNING:".yellow().bold(), message.yellow());
}

/// Display throttle message
pub fn throttle(message: &str, delay_secs: u64) {
    println!();
    println!("{} {} ({}s delay)", "THROTTLE:".bright_yellow().bold(),
        message.yellow(), delay_secs);
}

/// Display halt message
pub fn halt(message: &str) {
    println!();
    println!("{}", "!!! HALT !!!".on_red().white().bold());
    println!("{}", message.red());
}

/// Display crisis warning
pub fn crisis(message: &str) {
    println!();
    println!("{}", "╔══════════════════════════════════════════╗".red());
    println!("{}", "║           !!! CRISIS !!!                 ║".red().bold());
    println!("{}", "╚══════════════════════════════════════════╝".red());
    println!();
    println!("{}", message.red());
    println!();
    println!("Type {} to continue despite crisis, or {} to abort.",
        "'yes'".green(), "'no'".yellow());
}

/// Display pain signal
pub fn pain_signal(location: &str, intensity: f32) {
    let intensity_bar = "▓".repeat((intensity * 10.0) as usize);
    println!("{} {} {} (intensity: {})",
        "⚠".yellow(),
        "PAIN:".red().bold(),
        location.cyan(),
        intensity_bar.red()
    );
}

/// Display cooling message
pub fn cooling_start(action: &str, duration_secs: u64) {
    println!();
    println!("{} {} ({}s)", "COOLING:".cyan().bold(), action.white(), duration_secs);
}

/// Display cooling complete
pub fn cooling_complete() {
    println!("{}", "✓ Cooled".green());
}

/// Display error
pub fn error(message: &str) {
    println!();
    println!("{} {}", "ERROR:".red().bold(), message.red());
}

/// Display help
pub fn help() {
    println!();
    println!("{}", "═══ COMMANDS ═══".cyan().bold());
    println!();
    println!("  {}   Show full fractal state", "/status".yellow());
    println!("  {}  Show thermal zones", "/thermal".yellow());
    println!("  {}     Show pain/damage", "/pain".yellow());
    println!("  {}     Force 30s cooldown", "/cool".yellow());
    println!("  {}    Clear conversation", "/clear".yellow());
    println!("  {}  Toggle verbose mode", "/verbose".yellow());
    println!("  {}     Show this help", "/help".yellow());
    println!("  {}     Exit agent", "/quit".yellow());
    println!();
}

/// Get color for thermal state
fn thermal_state_color(state: ThermalState) -> &'static str {
    match state {
        ThermalState::Nominal => "green",
        ThermalState::Elevated => "yellow",
        ThermalState::Saturated => "red",
        ThermalState::Unsafe => "bright_red",
    }
}

/// Get color for integrated state
fn integrated_state_color(state: IntegratedState) -> &'static str {
    match state {
        IntegratedState::Calm => "green",
        IntegratedState::Alert => "yellow",
        IntegratedState::Degraded => "red",
        IntegratedState::Crisis => "bright_red",
    }
}

/// User input prompt
pub fn prompt() -> String {
    format!("{} ", "You:".green().bold())
}

/// Assistant label
pub fn assistant_label() {
    println!();
    println!("{}", "Claude:".blue().bold());
}
