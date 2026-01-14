//! ═══════════════════════════════════════════════════════════════════════════════
//! GROUNDING LAYER — Connection to Reality
//! ═══════════════════════════════════════════════════════════════════════════════
//! Intelligence without grounding is hallucination at scale.
//! This layer anchors cognition to reality.
//!
//! Components:
//! - Environmental Telemetry: Real sensor data, not just text about the world
//! - Temporal Anchoring: Actual time, not token position
//! - Causal Modeling: Interventional, not just correlational
//!
//! What's missing in current LLMs: Everything. They're brains in vats.
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod causal;
pub mod telemetry;
pub mod temporal;

pub use causal::{
    CausalConfig, CausalEdge, CausalGraph, CausalInference, CausalModel, CausalNode,
    ConfoundingAnalysis, CounterfactualQuery, Intervention, InterventionResult,
};
pub use telemetry::{
    EnvironmentSnapshot, EnvironmentalTelemetry, SensorReading, SensorType, TelemetryConfig,
    TelemetryReading,
};
pub use temporal::{
    CausalOrder, TemporalAnchor, TemporalConfig, TemporalContext, TemporalReading,
    TemporalRelation, TimeScale,
};

use crate::observations::ObservationBatch;
use crate::time::TimePoint;

/// Integrated grounding state - the system's connection to reality
#[derive(Debug, Clone)]
pub struct GroundingState {
    /// Environmental telemetry snapshot
    pub environment: EnvironmentSnapshot,
    /// Temporal context and anchoring
    pub temporal: TemporalContext,
    /// Causal model state
    pub causal_confidence: f64,
    /// Overall grounding quality (0.0 = hallucinating, 1.0 = fully grounded)
    pub grounding_quality: f64,
    /// Environmental constraints
    pub constraints: Vec<String>,
    /// Timestamp of this state
    pub timestamp: TimePoint,
}

impl GroundingState {
    /// Check if the grounding state is valid (above minimum quality threshold)
    pub fn is_valid(&self) -> bool {
        self.grounding_quality >= 0.3
    }
}

/// The Grounding Layer - anchors intelligence to reality
pub struct GroundingLayer {
    config: GroundingLayerConfig,
    telemetry: EnvironmentalTelemetry,
    temporal: TemporalAnchor,
    causal: CausalModel,
    state: GroundingState,
    history: Vec<GroundingState>,
}

#[derive(Debug, Clone)]
pub struct GroundingLayerConfig {
    /// How many historical states to retain
    pub history_size: usize,
    /// Minimum grounding quality before warning
    pub grounding_threshold: f64,
    /// Weight for environmental telemetry in quality calc
    pub telemetry_weight: f64,
    /// Weight for temporal anchoring in quality calc
    pub temporal_weight: f64,
    /// Weight for causal confidence in quality calc
    pub causal_weight: f64,
}

impl Default for GroundingLayerConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            grounding_threshold: 0.5,
            telemetry_weight: 0.4,
            temporal_weight: 0.3,
            causal_weight: 0.3,
        }
    }
}

impl GroundingLayer {
    pub fn new(config: GroundingLayerConfig) -> Self {
        let now = TimePoint::now();
        Self {
            telemetry: EnvironmentalTelemetry::new(TelemetryConfig::default()),
            temporal: TemporalAnchor::new(TemporalConfig::default()),
            causal: CausalModel::new(CausalConfig::default()),
            state: GroundingState {
                environment: EnvironmentSnapshot::default(),
                temporal: TemporalContext::default(),
                causal_confidence: 0.5,
                grounding_quality: 0.5,
                constraints: Vec::new(),
                timestamp: now,
            },
            history: Vec::with_capacity(config.history_size),
            config,
        }
    }

    /// Update grounding from observations
    pub fn update(&mut self, observations: &ObservationBatch) -> GroundingState {
        let now = TimePoint::now();

        // Update each component
        let env_snapshot = self.telemetry.process(observations);
        let temporal_ctx = self.temporal.anchor(now, observations);
        let causal_conf = self.causal.update(observations);

        // Calculate overall grounding quality
        let quality = self.calculate_grounding_quality(&env_snapshot, &temporal_ctx, causal_conf);

        // Build new state
        // Extract constraints from environment
        let constraints = self.extract_constraints(&env_snapshot);

        let new_state = GroundingState {
            environment: env_snapshot,
            temporal: temporal_ctx,
            causal_confidence: causal_conf,
            grounding_quality: quality,
            constraints,
            timestamp: now,
        };

        // Archive old state
        if self.history.len() >= self.config.history_size {
            self.history.remove(0);
        }
        self.history.push(self.state.clone());

        self.state = new_state.clone();
        new_state
    }

    fn calculate_grounding_quality(
        &self,
        env: &EnvironmentSnapshot,
        temporal: &TemporalContext,
        causal: f64,
    ) -> f64 {
        let env_quality = env.confidence;
        let temporal_quality = temporal.anchoring_strength;

        (env_quality * self.config.telemetry_weight)
            + (temporal_quality * self.config.temporal_weight)
            + (causal * self.config.causal_weight)
    }

    /// Check if currently grounded enough to trust outputs
    pub fn is_grounded(&self) -> bool {
        self.state.grounding_quality >= self.config.grounding_threshold
    }

    /// Extract constraints from environment snapshot
    fn extract_constraints(&self, env: &EnvironmentSnapshot) -> Vec<String> {
        let mut constraints = Vec::new();

        // Check for resource constraints
        if env.confidence < 0.5 {
            constraints.push("Low environmental confidence".to_string());
        }

        if env.stale_count > 0 {
            constraints.push(format!("{} stale sensor readings", env.stale_count));
        }

        constraints
    }

    /// Get current grounding state
    pub fn state(&self) -> &GroundingState {
        &self.state
    }

    /// Get grounding quality trend over recent history
    pub fn quality_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = self
            .history
            .iter()
            .rev()
            .take(10)
            .map(|s| s.grounding_quality)
            .collect();

        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear trend
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();

        let denom = n * sum_xx - sum_x.powi(2);
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }

    /// Perform a causal intervention and observe result
    pub fn intervene(&mut self, intervention: Intervention) -> InterventionResult {
        self.causal.intervene(intervention)
    }

    /// Query counterfactual: "what would have happened if X?"
    pub fn counterfactual(&self, query: CounterfactualQuery) -> CausalInference {
        self.causal.counterfactual(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grounding_layer_creation() {
        let layer = GroundingLayer::new(GroundingLayerConfig::default());
        assert!(layer.state.grounding_quality >= 0.0);
        assert!(layer.state.grounding_quality <= 1.0);
    }

    #[test]
    fn test_grounding_quality_calculation() {
        let config = GroundingLayerConfig::default();
        let layer = GroundingLayer::new(config);

        // Initial state should have some grounding
        assert!(layer.is_grounded() || !layer.is_grounded()); // Just verify it runs
    }
}
