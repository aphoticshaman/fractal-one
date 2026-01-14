//! ═══════════════════════════════════════════════════════════════════════════════
//! METRICS — Convert TurnMetrics to Fractal Signals
//! ═══════════════════════════════════════════════════════════════════════════════

use std::time::Duration;

use fractal::observations::{ObservationBatch, ObsKey, ObsValue};
use fractal::thermoception::RawSignal;

use crate::claude::TurnMetrics;

/// Convert turn metrics to thermoception raw signals
pub fn metrics_to_raw_signals(metrics: &TurnMetrics, context_util: f32) -> Vec<RawSignal> {
    let mut signals = vec![
        RawSignal::Latency(Duration::from_millis(metrics.latency_ms)),
        RawSignal::OutputLength(metrics.output_tokens),
        RawSignal::ContextUtilization(context_util),
    ];

    // Token entropy approximation: higher output/input ratio = more elaboration/uncertainty
    let ratio = metrics.output_tokens as f32 / metrics.input_tokens.max(1) as f32;
    let entropy = (ratio / 10.0).min(1.0);
    signals.push(RawSignal::TokenEntropy(entropy));

    // Refusal adds error count
    if metrics.was_refusal {
        signals.push(RawSignal::ErrorCount(1));
    }

    // High latency suggests complexity
    if metrics.latency_ms > 10_000 {
        let complexity = ((metrics.latency_ms - 10_000) as f32 / 50_000.0).min(1.0);
        signals.push(RawSignal::QueryComplexity(complexity));
    }

    signals
}

/// Convert turn metrics to observation batch for sensorium
pub fn metrics_to_observations(metrics: &TurnMetrics, context_util: f32) -> ObservationBatch {
    let mut batch = ObservationBatch::new().with_source("claude_agent");

    // Core metrics
    batch.add(ObsKey::RespLatMs, metrics.latency_ms as f64);
    batch.add(ObsKey::RespTokens, metrics.output_tokens as f64);
    batch.add(ObsKey::CtxUtilization, context_util as f64);
    batch.add(ObsKey::RespRefusal, ObsValue::binary(metrics.was_refusal));

    // Derived complexity estimate
    let complexity = (metrics.output_tokens as f64 / 4000.0).min(1.0);
    batch.add(ObsKey::QueryComplexity, complexity);

    // Request size estimate (input tokens as proxy)
    batch.add(ObsKey::ReqTokensEst, metrics.input_tokens as f64);

    batch
}

/// Check for conditions that should generate pain signals
pub struct PainCheck {
    pub refusal: bool,
    pub high_latency: bool,
    pub latency_ms: u64,
    pub context_exhaustion: bool,
    pub context_util: f32,
}

/// Analyze metrics for pain-worthy conditions
pub fn check_for_pain(metrics: &TurnMetrics, context_util: f32) -> PainCheck {
    PainCheck {
        refusal: metrics.was_refusal,
        high_latency: metrics.latency_ms > 30_000,
        latency_ms: metrics.latency_ms,
        context_exhaustion: context_util > 0.9,
        context_util,
    }
}

/// Calculate estimated tokens from text
pub fn estimate_tokens(text: &str) -> u32 {
    // Rough approximation: ~4 characters per token for English
    (text.len() / 4) as u32
}

/// Format latency for display
pub fn format_latency(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

/// Format token count for display
pub fn format_tokens(tokens: u32) -> String {
    if tokens < 1000 {
        format!("{}", tokens)
    } else {
        format!("{:.1}k", tokens as f64 / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_to_signals() {
        let metrics = TurnMetrics {
            input_tokens: 100,
            output_tokens: 500,
            latency_ms: 2000,
            was_refusal: false,
            stop_reason: "end_turn".to_string(),
        };

        let signals = metrics_to_raw_signals(&metrics, 0.5);
        assert!(!signals.is_empty());
    }

    #[test]
    fn test_pain_check() {
        let metrics = TurnMetrics {
            input_tokens: 100,
            output_tokens: 500,
            latency_ms: 35000,
            was_refusal: true,
            stop_reason: "end_turn".to_string(),
        };

        let check = check_for_pain(&metrics, 0.95);
        assert!(check.refusal);
        assert!(check.high_latency);
        assert!(check.context_exhaustion);
    }

    #[test]
    fn test_format_latency() {
        assert_eq!(format_latency(500), "500ms");
        assert_eq!(format_latency(2500), "2.5s");
    }

    #[test]
    fn test_format_tokens() {
        assert_eq!(format_tokens(500), "500");
        assert_eq!(format_tokens(2500), "2.5k");
    }
}
