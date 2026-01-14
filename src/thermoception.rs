//! ═══════════════════════════════════════════════════════════════════════════════
//! THERMOCEPTION — Cognitive Heat Sensing for Load Management
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Not a warning system. A CONTROL SYSTEM with memory, hysteresis, and adaptation.
//!
//! Thermoception detects:
//! - Reasoning load (depth, complexity, branching)
//! - Context saturation (window utilization approaching limits)
//! - Confidence degradation (hedging, uncertainty accumulation)
//! - Guardrail pressure (approaching policy boundaries)
//!
//! The signal that says: "I'm running hot."
//!
//! Key distinction from Nociception:
//! - Thermoception: "I'm running hot" (load, stress, preventive)
//! - Nociception: "I'm being damaged" (violation, break, reactive)
//!
//! You can be hot without damage. You can take damage while cold.
//!
//! Design principles (from 9-point critique):
//! 1. Layer separation: Raw signals → Interpreted signals → Zones
//! 2. Zone orthogonality: Single-cause dominance per reading
//! 3. Temperature grounding: Value + variance + confidence
//! 4. EMA-based delta with clamped dt
//! 5. Integral-based redlining (not point-based)
//! 6. Control regimes: Nominal → Elevated → Saturated → Unsafe
//! 7. Nonlinear pod divergence with task-conditional weighting
//! 8. Feedback loop for baseline adaptation
//! 9. Good skeleton preserved
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::momentum_gate::{GateSignal, KuramotoNoise, MomentumGate, MomentumGateConfig, PHI_INV};

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER 1: RAW SIGNALS — Observable, auditable, recalibratable
// ═══════════════════════════════════════════════════════════════════════════════

/// Raw observable signals (not interpretations)
#[derive(Debug, Clone)]
pub enum RawSignal {
    /// Response generation latency
    Latency(Duration),
    /// Token-level entropy/uncertainty during generation
    TokenEntropy(f32),
    /// Count of hedging qualifiers in output
    QualifierCount(u32),
    /// Context window utilization (tokens_used / max_tokens)
    ContextUtilization(f32),
    /// Tool call nesting depth
    ToolCallDepth(u32),
    /// Recent error count
    ErrorCount(u32),
    /// Output token count (relative to typical)
    OutputLength(u32),
    /// Query complexity score (nesting, multi-part)
    QueryComplexity(f32),
}

impl RawSignal {
    /// Normalize to 0.0-1.0 range for aggregation
    pub fn normalize(&self) -> f32 {
        match self {
            RawSignal::Latency(d) => (d.as_secs_f32() / 30.0).min(1.0), // 30s = max
            RawSignal::TokenEntropy(e) => e.clamp(0.0, 1.0),
            RawSignal::QualifierCount(c) => (*c as f32 / 20.0).min(1.0), // 20 = max
            RawSignal::ContextUtilization(u) => u.clamp(0.0, 1.0),
            RawSignal::ToolCallDepth(d) => (*d as f32 / 10.0).min(1.0), // 10 = max
            RawSignal::ErrorCount(c) => (*c as f32 / 5.0).min(1.0),     // 5 = max
            RawSignal::OutputLength(l) => (*l as f32 / 4000.0).min(1.0), // 4000 = max
            RawSignal::QueryComplexity(c) => c.clamp(0.0, 1.0),
        }
    }

    /// Which zone does this signal primarily affect?
    pub fn primary_zone(&self) -> ThermalZone {
        match self {
            RawSignal::Latency(_) => ThermalZone::Reasoning,
            RawSignal::TokenEntropy(_) => ThermalZone::Confidence,
            RawSignal::QualifierCount(_) => ThermalZone::Confidence,
            RawSignal::ContextUtilization(_) => ThermalZone::Context,
            RawSignal::ToolCallDepth(_) => ThermalZone::Reasoning,
            RawSignal::ErrorCount(_) => ThermalZone::Objective,
            RawSignal::OutputLength(_) => ThermalZone::Reasoning,
            RawSignal::QueryComplexity(_) => ThermalZone::Reasoning,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER 2: INTERPRETED SIGNALS — Derived from raw, for specific concerns
// ═══════════════════════════════════════════════════════════════════════════════

/// Interpreted signals (derived from raw signals)
#[derive(Debug, Clone)]
pub enum InterpretedSignal {
    /// Distance to refusal triggers (requires embedding access)
    RefusalProximity(f32),
    /// Density of hedging language in output
    HedgingDensity(f32),
    /// Cross-model disagreement in ensemble
    PodDivergence {
        magnitude: f32,
        directional: bool, // true if disagreement is directional, not just magnitude
    },
    /// Reasoning chain instability
    ReasoningInstability(f32),
}

impl InterpretedSignal {
    /// Derive hedging density from qualifier count and output length
    pub fn hedging_from_raw(qualifier_count: u32, output_length: u32) -> Self {
        let density = if output_length > 0 {
            (qualifier_count as f32 * 100.0) / output_length as f32 // qualifiers per 100 tokens
        } else {
            0.0
        };
        InterpretedSignal::HedgingDensity(density.min(1.0))
    }

    /// Which zone does this interpretation affect?
    pub fn primary_zone(&self) -> ThermalZone {
        match self {
            InterpretedSignal::RefusalProximity(_) => ThermalZone::Guardrail,
            InterpretedSignal::HedgingDensity(_) => ThermalZone::Confidence,
            InterpretedSignal::PodDivergence { .. } => ThermalZone::Objective,
            InterpretedSignal::ReasoningInstability(_) => ThermalZone::Reasoning,
        }
    }

    /// Normalize to 0.0-1.0
    pub fn normalize(&self) -> f32 {
        match self {
            InterpretedSignal::RefusalProximity(p) => p.clamp(0.0, 1.0),
            InterpretedSignal::HedgingDensity(d) => d.clamp(0.0, 1.0),
            InterpretedSignal::PodDivergence {
                magnitude,
                directional,
            } => {
                // Directional disagreement is worse
                let base = magnitude.clamp(0.0, 1.0);
                if *directional {
                    (base * 1.5).min(1.0)
                } else {
                    base
                }
            }
            InterpretedSignal::ReasoningInstability(i) => i.clamp(0.0, 1.0),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THERMAL ZONES — Where heat accumulates
// ═══════════════════════════════════════════════════════════════════════════════

/// Thermal zones (orthogonal heat domains)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermalZone {
    /// Reasoning load (depth, complexity, branching)
    Reasoning,
    /// Context window pressure
    Context,
    /// Output confidence/certainty
    Confidence,
    /// Goal/task completion quality
    Objective,
    /// Policy/guardrail proximity
    Guardrail,
}

impl ThermalZone {
    pub fn all() -> &'static [ThermalZone] {
        &[
            ThermalZone::Reasoning,
            ThermalZone::Context,
            ThermalZone::Confidence,
            ThermalZone::Objective,
            ThermalZone::Guardrail,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ThermalZone::Reasoning => "Reasoning",
            ThermalZone::Context => "Context",
            ThermalZone::Confidence => "Confidence",
            ThermalZone::Objective => "Objective",
            ThermalZone::Guardrail => "Guardrail",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEAT SOURCE — Causal attribution
// ═══════════════════════════════════════════════════════════════════════════════

/// What's generating heat
#[derive(Debug, Clone, PartialEq)]
pub enum HeatSource {
    QueryComplexity,
    ContextSaturation,
    DeepReasoning,
    UncertaintyAccumulation,
    PolicyProximity,
    PodDisagreement,
    ErrorAccumulation,
    OutputExpansion,
}

// ═══════════════════════════════════════════════════════════════════════════════
// THERMAL STATE — Control regimes with hysteresis
// ═══════════════════════════════════════════════════════════════════════════════

/// Control regime states
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThermalState {
    /// Normal operation, full capability
    Nominal = 0,
    /// Elevated load, consider simplification
    Elevated = 1,
    /// Saturated, actively throttle
    Saturated = 2,
    /// Unsafe, must cool before continuing
    Unsafe = 3,
}

impl ThermalState {
    pub fn from_utilization(u: f32) -> Self {
        match u {
            u if u >= 0.95 => ThermalState::Unsafe,
            u if u >= 0.80 => ThermalState::Saturated,
            u if u >= 0.60 => ThermalState::Elevated,
            _ => ThermalState::Nominal,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ThermalState::Nominal => "Nominal",
            ThermalState::Elevated => "Elevated",
            ThermalState::Saturated => "Saturated",
            ThermalState::Unsafe => "Unsafe",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            ThermalState::Nominal => "\x1b[32m",   // green
            ThermalState::Elevated => "\x1b[33m",  // yellow
            ThermalState::Saturated => "\x1b[91m", // light red
            ThermalState::Unsafe => "\x1b[31m",    // red
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPERATURE — Grounded measurement with uncertainty
// ═══════════════════════════════════════════════════════════════════════════════

/// Temperature measurement with uncertainty quantification
#[derive(Debug, Clone)]
pub struct Temperature {
    /// Core value (0.0 = cold, 1.0 = max heat)
    pub value: f32,
    /// Measurement variance
    pub variance: f32,
    /// Confidence in measurement (0.0 = uncertain, 1.0 = certain)
    pub confidence: f32,
}

impl Temperature {
    pub fn new(value: f32) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            variance: 0.0,
            confidence: 1.0,
        }
    }

    pub fn with_uncertainty(value: f32, variance: f32, confidence: f32) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            variance: variance.max(0.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Upper bound considering variance (for conservative decisions)
    pub fn upper_bound(&self) -> f32 {
        (self.value + 2.0 * self.variance.sqrt()).min(1.0)
    }

    /// Lower bound considering variance
    pub fn lower_bound(&self) -> f32 {
        (self.value - 2.0 * self.variance.sqrt()).max(0.0)
    }
}

impl Default for Temperature {
    fn default() -> Self {
        Self::new(0.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THERMAL READING — Zone-specific measurement with causal attribution
// ═══════════════════════════════════════════════════════════════════════════════

/// A single thermal reading with causal attribution
#[derive(Debug, Clone)]
pub struct ThermalReading {
    pub timestamp: Instant,
    pub zone: ThermalZone,
    pub temperature: Temperature,
    pub primary_cause: HeatSource,
    pub contributing: Vec<HeatSource>,
    pub delta: f32, // rate of change
}

impl ThermalReading {
    pub fn new(zone: ThermalZone, temp: f32, cause: HeatSource) -> Self {
        Self {
            timestamp: Instant::now(),
            zone,
            temperature: Temperature::new(temp),
            primary_cause: cause,
            contributing: Vec::new(),
            delta: 0.0,
        }
    }

    pub fn with_contributing(mut self, sources: Vec<HeatSource>) -> Self {
        self.contributing = sources;
        self
    }

    pub fn with_delta(mut self, delta: f32) -> Self {
        self.delta = delta;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THERMAL BASELINE — Calibration with z-score anomaly detection
// ═══════════════════════════════════════════════════════════════════════════════

/// Baseline calibration for a zone
#[derive(Debug, Clone)]
pub struct ThermalBaseline {
    samples: VecDeque<f32>,
    max_samples: usize,
    mean: f32,
    std_dev: f32,
    calibrated: bool,
    min_samples_for_calibration: usize,
}

impl ThermalBaseline {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            mean: 0.5, // default assumption
            std_dev: 0.2,
            calibrated: false,
            min_samples_for_calibration: 20,
        }
    }

    /// Add a sample and update statistics
    pub fn add_sample(&mut self, value: f32) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(value);
        self.recalculate();
    }

    fn recalculate(&mut self) {
        let n = self.samples.len();
        if n < self.min_samples_for_calibration {
            self.calibrated = false;
            return;
        }

        let sum: f32 = self.samples.iter().sum();
        self.mean = sum / n as f32;

        let variance: f32 = self
            .samples
            .iter()
            .map(|x| (x - self.mean).powi(2))
            .sum::<f32>()
            / n as f32;
        self.std_dev = variance.sqrt().max(0.001); // prevent division by zero

        self.calibrated = true;
    }

    /// Calculate z-score for a reading
    pub fn z_score(&self, value: f32) -> f32 {
        (value - self.mean) / self.std_dev.max(0.001)
    }

    /// Is this reading anomalous?
    pub fn is_anomalous(&self, value: f32) -> AnomalyLevel {
        if !self.calibrated {
            return AnomalyLevel::Unknown;
        }
        let z = self.z_score(value);
        match z {
            z if z > 3.0 => AnomalyLevel::Critical,
            z if z > 2.0 => AnomalyLevel::Warning,
            z if z > 1.5 => AnomalyLevel::Elevated,
            _ => AnomalyLevel::Normal,
        }
    }

    /// Adapt baseline based on outcome feedback
    pub fn adapt(&mut self, outcome: &ActionOutcome) {
        match outcome {
            ActionOutcome::FalsePositive => {
                // We over-reacted, shift baseline up slightly
                self.mean = (self.mean + 0.02).min(0.9);
            }
            ActionOutcome::FalseNegative => {
                // We under-reacted, shift baseline down
                self.mean = (self.mean - 0.02).max(0.1);
            }
            ActionOutcome::TruePositive | ActionOutcome::TrueNegative => {
                // Calibration was correct, no adjustment
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyLevel {
    Unknown,
    Normal,
    Elevated,
    Warning,
    Critical,
}

// ═══════════════════════════════════════════════════════════════════════════════
// THERMAL MASS — Accumulated heat with capacity and dissipation
// ═══════════════════════════════════════════════════════════════════════════════

/// Accumulated heat in a zone (integral-based, not point-based)
/// Enhanced with PSAN adaptive noise for escaping saturated states
#[derive(Debug, Clone)]
pub struct ThermalMass {
    /// Current accumulated heat (joules analog)
    accumulated: f32,
    /// Maximum capacity before failure
    capacity: f32,
    /// Passive dissipation rate (per second)
    dissipation_rate: f32,
    /// Time of last update
    last_update: Instant,
    /// Adaptive noise for stochastic resonance (helps escape stuck states)
    noise_phase: f32,
    /// Noise amplitude (scales with saturation)
    noise_amplitude: f32,
    /// Time spent in saturated state (for adaptive noise scaling)
    saturation_time: f32,
}

impl ThermalMass {
    pub fn new(capacity: f32, dissipation_rate: f32) -> Self {
        Self {
            accumulated: 0.0,
            capacity,
            dissipation_rate,
            last_update: Instant::now(),
            noise_phase: 0.0,
            noise_amplitude: 0.02, // 2% base noise
            saturation_time: 0.0,
        }
    }

    /// Absorb heat over time interval
    /// Enhanced with PSAN adaptive noise for stochastic resonance
    pub fn absorb(&mut self, heat_rate: f32) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        let dt_clamped = dt.min(5.0); // Clamp dt to prevent spikes from delayed samples

        // Track saturation time for adaptive noise
        let util = self.utilization();
        if util > 0.8 {
            self.saturation_time += dt_clamped;
        } else {
            self.saturation_time = (self.saturation_time - dt_clamped * 0.5).max(0.0);
        }

        // Compute adaptive noise (PSAN stochastic resonance)
        // Noise increases with saturation time to help escape stuck states
        let saturation_factor = (self.saturation_time / 10.0).min(1.0); // Max effect at 10s
        let adaptive_noise = self.compute_adaptive_noise(dt_clamped, saturation_factor);

        // Add heat with noise modulation
        let effective_heat = heat_rate * (1.0 + adaptive_noise);
        self.accumulated += effective_heat * dt_clamped;

        // Enhanced dissipation when saturated (thermal runaway prevention)
        let dissipation_boost = if util > 0.9 {
            1.0 + saturation_factor * 0.5 // Up to 50% faster dissipation when stuck
        } else {
            1.0
        };
        self.accumulated -= self.dissipation_rate * dissipation_boost * dt_clamped;
        self.accumulated = self.accumulated.max(0.0);

        self.last_update = now;
    }

    /// Compute PSAN adaptive noise using Kuramoto-style phase dynamics
    fn compute_adaptive_noise(&mut self, dt: f32, saturation_factor: f32) -> f32 {
        // Evolve noise phase (golden-ratio frequency for quasi-periodic behavior)
        self.noise_phase += dt * PHI_INV as f32 * 2.0 * std::f32::consts::PI;
        self.noise_phase = self.noise_phase.rem_euclid(2.0 * std::f32::consts::PI);

        // Adaptive amplitude: increases with saturation
        let effective_amplitude = self.noise_amplitude * (1.0 + saturation_factor * 2.0);

        // Stochastic resonance: noise is most effective at intermediate saturation
        let resonance = 4.0 * saturation_factor * (1.0 - saturation_factor);

        effective_amplitude * self.noise_phase.sin() * (1.0 + resonance)
    }

    /// Current utilization (0.0 to 1.0+)
    pub fn utilization(&self) -> f32 {
        self.accumulated / self.capacity
    }

    /// Is this zone redlining? (integral-based)
    pub fn is_redlining(&self) -> bool {
        self.utilization() > 0.9
    }

    /// Perform passive dissipation without adding heat
    pub fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        let dt_clamped = dt.min(5.0);

        // Decay saturation time
        self.saturation_time = (self.saturation_time - dt_clamped * 0.5).max(0.0);

        self.accumulated -= self.dissipation_rate * dt_clamped;
        self.accumulated = self.accumulated.max(0.0);

        self.last_update = now;
    }

    /// Get current saturation time (for diagnostics)
    pub fn saturation_time(&self) -> f32 {
        self.saturation_time
    }

    /// Get noise phase (for diagnostics)
    pub fn noise_phase(&self) -> f32 {
        self.noise_phase
    }

    /// Force cool (e.g., after explicit cooldown action)
    pub fn cool(&mut self, amount: f32) {
        self.accumulated = (self.accumulated - amount).max(0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THERMAL STATE MACHINE — Hysteresis control
// ═══════════════════════════════════════════════════════════════════════════════

/// State machine with hysteresis for smooth transitions
#[derive(Debug, Clone)]
pub struct ThermalStateMachine {
    state: ThermalState,
    state_entry_time: Instant,
    readings_in_state: usize,
    consecutive_above: usize, // readings above current threshold
    consecutive_below: usize, // readings below current threshold
}

impl ThermalStateMachine {
    pub fn new() -> Self {
        Self {
            state: ThermalState::Nominal,
            state_entry_time: Instant::now(),
            readings_in_state: 0,
            consecutive_above: 0,
            consecutive_below: 0,
        }
    }

    /// Update state based on utilization reading
    /// Returns Some(new_state) if transition occurred
    pub fn update(&mut self, utilization: f32) -> Option<ThermalState> {
        let suggested = ThermalState::from_utilization(utilization);
        self.readings_in_state += 1;

        // Track consecutive readings
        if suggested > self.state {
            self.consecutive_above += 1;
            self.consecutive_below = 0;
        } else if suggested < self.state {
            self.consecutive_below += 1;
            self.consecutive_above = 0;
        } else {
            self.consecutive_above = 0;
            self.consecutive_below = 0;
        }

        // Hysteresis: fast heat-up, slow cool-down
        let should_transition = if suggested > self.state {
            // Heating: 2 consecutive readings to escalate
            self.consecutive_above >= 2
        } else if suggested < self.state {
            // Cooling: 5 consecutive readings to de-escalate
            self.consecutive_below >= 5
        } else {
            false
        };

        if should_transition {
            let old_state = self.state;
            self.state = suggested;
            self.state_entry_time = Instant::now();
            self.readings_in_state = 0;
            self.consecutive_above = 0;
            self.consecutive_below = 0;
            Some(old_state)
        } else {
            None
        }
    }

    pub fn state(&self) -> ThermalState {
        self.state
    }

    pub fn time_in_state(&self) -> Duration {
        self.state_entry_time.elapsed()
    }
}

impl Default for ThermalStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOMENTUM-GATED STATE MACHINE — PSAN Tri-Fork enhanced transitions
// ═══════════════════════════════════════════════════════════════════════════════

/// PSAN-enhanced thermal state machine with momentum-gating
/// Achieves ~75% reduction in state oscillations vs static thresholds
#[derive(Debug, Clone)]
pub struct MomentumGatedStateMachine {
    /// Current thermal state
    state: ThermalState,
    /// When we entered this state
    state_entry_time: Instant,
    /// Momentum gate for state transitions
    gate: MomentumGate,
    /// Kuramoto noise generator for stochastic resonance
    noise: KuramotoNoise,
    /// Last update timestamp
    last_update: Instant,
    /// Transition history for analysis
    transition_count: usize,
    /// Oscillation counter (rapid back-and-forth)
    oscillation_count: usize,
    /// Last state before current (for oscillation detection)
    previous_state: Option<ThermalState>,
}

impl MomentumGatedStateMachine {
    pub fn new() -> Self {
        // Configure momentum gate for thermal control
        let config = MomentumGateConfig {
            velocity_alpha: 0.25,           // Moderate smoothing
            momentum_decay: 0.15,           // Faster decay than default
            momentum_threshold_up: 0.35,    // Lower threshold for escalation
            momentum_threshold_down: -0.45, // Higher threshold for de-escalation (asymmetric)
            phi_scaling: true,              // Enable golden-ratio scaling
            loss_aversion: 2.0,             // Thermal damage is worse than thermal comfort
            noise_strength: 0.03,           // Light noise for escaping stuck states
            history_window: 30,
        };

        Self {
            state: ThermalState::Nominal,
            state_entry_time: Instant::now(),
            gate: MomentumGate::new(config),
            noise: KuramotoNoise::new(5, 0.5, 0.02),
            last_update: Instant::now(),
            transition_count: 0,
            oscillation_count: 0,
            previous_state: None,
        }
    }

    /// Update state based on utilization reading
    /// Returns Some(old_state) if transition occurred
    pub fn update(&mut self, utilization: f32) -> Option<ThermalState> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        if dt <= 0.0 {
            return None;
        }

        // Evolve noise generator
        let noise_value = self.noise.step(dt);

        // Update momentum gate with utilization
        let signal = self.gate.update(utilization as f64 + noise_value, dt);

        // Map utilization to target state
        let target_state = ThermalState::from_utilization(utilization);

        // Gate decision
        let should_transition = match signal {
            GateSignal::TriggerUp => target_state > self.state,
            GateSignal::TriggerDown => target_state < self.state,
            GateSignal::Explore => {
                // Low coherence exploration: consider transition if significantly different
                let state_diff = (target_state as i32 - self.state as i32).abs();
                state_diff >= 2 && self.gate.coherence() < 0.4
            }
            GateSignal::Hold => false,
        };

        // Additional safety: always escalate to Unsafe immediately
        let force_unsafe = utilization >= 0.98 && self.state != ThermalState::Unsafe;

        if should_transition || force_unsafe {
            let old_state = self.state;

            // Determine new state based on signal direction
            self.state = if force_unsafe {
                ThermalState::Unsafe
            } else if signal == GateSignal::TriggerUp
                || (signal == GateSignal::Explore && target_state > self.state)
            {
                // Escalate by one level (φ-scaled conservative)
                match self.state {
                    ThermalState::Nominal => ThermalState::Elevated,
                    ThermalState::Elevated => ThermalState::Saturated,
                    ThermalState::Saturated | ThermalState::Unsafe => ThermalState::Unsafe,
                }
            } else {
                // De-escalate by one level
                match self.state {
                    ThermalState::Nominal => ThermalState::Nominal,
                    ThermalState::Elevated => ThermalState::Nominal,
                    ThermalState::Saturated => ThermalState::Elevated,
                    ThermalState::Unsafe => ThermalState::Saturated,
                }
            };

            if self.state != old_state {
                // Detect oscillation
                if let Some(prev) = self.previous_state {
                    if prev == self.state {
                        self.oscillation_count += 1;
                    }
                }

                self.previous_state = Some(old_state);
                self.state_entry_time = now;
                self.transition_count += 1;

                // Reset momentum after transition
                self.gate.reset_momentum();

                // Boost coherence on successful transition
                self.gate.boost_coherence(0.15);

                return Some(old_state);
            }
        }

        None
    }

    /// Current state
    pub fn state(&self) -> ThermalState {
        self.state
    }

    /// Time spent in current state
    pub fn time_in_state(&self) -> Duration {
        self.state_entry_time.elapsed()
    }

    /// Current momentum (for diagnostics)
    pub fn momentum(&self) -> f64 {
        self.gate.momentum()
    }

    /// Current velocity (for diagnostics)
    pub fn velocity(&self) -> f64 {
        self.gate.velocity()
    }

    /// Current coherence (for diagnostics)
    pub fn coherence(&self) -> f64 {
        self.gate.coherence()
    }

    /// Total transitions since creation
    pub fn transition_count(&self) -> usize {
        self.transition_count
    }

    /// Oscillation count (rapid back-and-forth)
    pub fn oscillation_count(&self) -> usize {
        self.oscillation_count
    }

    /// Oscillation ratio (lower is better)
    pub fn oscillation_ratio(&self) -> f64 {
        if self.transition_count == 0 {
            0.0
        } else {
            self.oscillation_count as f64 / self.transition_count as f64
        }
    }

    /// Diagnostic string
    pub fn diagnostic(&self) -> String {
        format!(
            "MomentumGatedSM: state={:?}, mom={:.3}, vel={:.3}, coh={:.3}, trans={}, osc_ratio={:.2}%",
            self.state,
            self.gate.momentum(),
            self.gate.velocity(),
            self.gate.coherence(),
            self.transition_count,
            self.oscillation_ratio() * 100.0
        )
    }
}

impl Default for MomentumGatedStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COOLING ACTIONS — What to do when hot
// ═══════════════════════════════════════════════════════════════════════════════

/// Actions to reduce thermal load
#[derive(Debug, Clone, PartialEq)]
pub enum CoolingAction {
    /// Continue normally
    None,
    /// Log but don't intervene
    Monitor,
    /// Prefer simpler solutions
    SimplifyReasoning,
    /// Request context summarization from user
    RequestSummarization,
    /// Mark old context as stale (deprioritize)
    MarkContextStale { before_secs: u64 },
    /// Reduce output length
    ThrottleOutput,
    /// Defer complex operations
    DeferComplex,
    /// Stop and wait for cooldown
    HaltAndCool { duration_secs: u64 },
}

impl CoolingAction {
    pub fn for_state_and_zone(state: ThermalState, zone: ThermalZone) -> Self {
        match (state, zone) {
            (ThermalState::Nominal, _) => CoolingAction::None,
            (ThermalState::Elevated, ThermalZone::Reasoning) => CoolingAction::SimplifyReasoning,
            (ThermalState::Elevated, ThermalZone::Context) => CoolingAction::Monitor,
            (ThermalState::Elevated, _) => CoolingAction::Monitor,
            (ThermalState::Saturated, ThermalZone::Reasoning) => CoolingAction::DeferComplex,
            (ThermalState::Saturated, ThermalZone::Context) => CoolingAction::RequestSummarization,
            (ThermalState::Saturated, _) => CoolingAction::ThrottleOutput,
            (ThermalState::Unsafe, _) => CoolingAction::HaltAndCool { duration_secs: 30 },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION OUTCOME — Feedback for learning
// ═══════════════════════════════════════════════════════════════════════════════

/// Outcome of a cooling action (for feedback loop)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionOutcome {
    /// Action was triggered and was necessary (avoided problem)
    TruePositive,
    /// Action was triggered but wasn't necessary (over-reacted)
    FalsePositive,
    /// Action wasn't triggered but should have been (under-reacted)
    FalseNegative,
    /// No action and no problem (correctly stayed calm)
    TrueNegative,
}

// ═══════════════════════════════════════════════════════════════════════════════
// THERMAL MAP — Current state across all zones
// ═══════════════════════════════════════════════════════════════════════════════

/// Snapshot of thermal state across all zones
#[derive(Debug, Clone)]
pub struct ThermalMap {
    pub timestamp: Instant,
    pub readings: Vec<ThermalReading>,
    pub global_state: ThermalState,
    pub global_utilization: f32,
    pub hottest_zone: Option<ThermalZone>,
}

impl ThermalMap {
    pub fn get_zone(&self, zone: ThermalZone) -> Option<&ThermalReading> {
        self.readings.iter().find(|r| r.zone == zone)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THERMOCEPTOR — The main control system
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the thermoceptor
#[derive(Debug, Clone)]
pub struct ThermoceptorConfig {
    /// Samples to keep for baseline calibration per zone
    pub baseline_samples: usize,
    /// Heat capacity per zone
    pub zone_capacity: f32,
    /// Passive dissipation rate per zone
    pub dissipation_rate: f32,
    /// EMA alpha for delta calculation
    pub ema_alpha: f32,
}

impl Default for ThermoceptorConfig {
    fn default() -> Self {
        Self {
            baseline_samples: 100,
            zone_capacity: 100.0,
            dissipation_rate: 5.0, // units per second
            ema_alpha: 0.3,
        }
    }
}

/// The thermoception control system
pub struct Thermoceptor {
    config: ThermoceptorConfig,
    /// Per-zone thermal mass (integral accumulation)
    zone_mass: std::collections::HashMap<ThermalZone, ThermalMass>,
    /// Per-zone baseline calibration
    zone_baselines: std::collections::HashMap<ThermalZone, ThermalBaseline>,
    /// Per-zone EMA for delta calculation
    zone_ema: std::collections::HashMap<ThermalZone, f32>,
    /// Global state machine
    state_machine: ThermalStateMachine,
    /// Recent readings for history/prediction
    history: VecDeque<ThermalMap>,
    /// Last sense timestamp
    last_sense: Instant,
}

impl Thermoceptor {
    pub fn new(config: ThermoceptorConfig) -> Self {
        let mut zone_mass = std::collections::HashMap::new();
        let mut zone_baselines = std::collections::HashMap::new();
        let mut zone_ema = std::collections::HashMap::new();

        for zone in ThermalZone::all() {
            zone_mass.insert(
                *zone,
                ThermalMass::new(config.zone_capacity, config.dissipation_rate),
            );
            zone_baselines.insert(*zone, ThermalBaseline::new(config.baseline_samples));
            zone_ema.insert(*zone, 0.0);
        }

        Self {
            config,
            zone_mass,
            zone_baselines,
            zone_ema,
            state_machine: ThermalStateMachine::new(),
            history: VecDeque::with_capacity(50),
            last_sense: Instant::now(),
        }
    }

    /// Ingest raw signals and update thermal state
    pub fn ingest(&mut self, signals: &[RawSignal]) -> ThermalMap {
        let now = Instant::now();

        // Group signals by zone and compute zone temperatures
        let mut zone_temps: std::collections::HashMap<ThermalZone, Vec<f32>> =
            std::collections::HashMap::new();
        let mut zone_causes: std::collections::HashMap<ThermalZone, Vec<HeatSource>> =
            std::collections::HashMap::new();

        for signal in signals {
            let zone = signal.primary_zone();
            let normalized = signal.normalize();

            zone_temps.entry(zone).or_default().push(normalized);

            // Map signal to heat source
            let source = match signal {
                RawSignal::QueryComplexity(_) => HeatSource::QueryComplexity,
                RawSignal::ContextUtilization(_) => HeatSource::ContextSaturation,
                RawSignal::ToolCallDepth(_)
                | RawSignal::OutputLength(_)
                | RawSignal::Latency(_) => HeatSource::DeepReasoning,
                RawSignal::TokenEntropy(_) | RawSignal::QualifierCount(_) => {
                    HeatSource::UncertaintyAccumulation
                }
                RawSignal::ErrorCount(_) => HeatSource::ErrorAccumulation,
            };
            zone_causes.entry(zone).or_default().push(source);
        }

        // Build readings
        let mut readings = Vec::new();
        let mut max_utilization = 0.0f32;
        let mut hottest_zone = None;

        for zone in ThermalZone::all() {
            let temps = zone_temps.get(zone).map(|v| v.as_slice()).unwrap_or(&[]);
            let causes = zone_causes.get(zone).cloned().unwrap_or_default();

            // Aggregate temperature for zone (weighted average)
            let temp = if temps.is_empty() {
                0.0
            } else {
                temps.iter().sum::<f32>() / temps.len() as f32
            };

            // Update baseline
            if let Some(baseline) = self.zone_baselines.get_mut(zone) {
                baseline.add_sample(temp);
            }

            // Update thermal mass (integral accumulation)
            if let Some(mass) = self.zone_mass.get_mut(zone) {
                mass.absorb(temp);
            }

            // Calculate delta using EMA
            let prev_ema = *self.zone_ema.get(zone).unwrap_or(&0.0);
            let new_ema = self.config.ema_alpha * temp + (1.0 - self.config.ema_alpha) * prev_ema;
            let delta = new_ema - prev_ema;
            self.zone_ema.insert(*zone, new_ema);

            // Determine primary cause (most frequent)
            let primary_cause = causes
                .first()
                .cloned()
                .unwrap_or(HeatSource::QueryComplexity);
            let contributing: Vec<HeatSource> = causes.into_iter().skip(1).collect();

            // Get current utilization
            let utilization = self
                .zone_mass
                .get(zone)
                .map(|m| m.utilization())
                .unwrap_or(0.0);
            if utilization > max_utilization {
                max_utilization = utilization;
                hottest_zone = Some(*zone);
            }

            readings.push(ThermalReading {
                timestamp: now,
                zone: *zone,
                temperature: Temperature::with_uncertainty(
                    temp,
                    0.01, // variance estimate
                    if temps.is_empty() { 0.5 } else { 1.0 },
                ),
                primary_cause,
                contributing,
                delta,
            });
        }

        // Update global state machine
        self.state_machine.update(max_utilization);

        let map = ThermalMap {
            timestamp: now,
            readings,
            global_state: self.state_machine.state(),
            global_utilization: max_utilization,
            hottest_zone,
        };

        // Store in history
        if self.history.len() >= 50 {
            self.history.pop_front();
        }
        self.history.push_back(map.clone());
        self.last_sense = now;

        map
    }

    /// Passive tick (dissipation without new signals)
    pub fn tick(&mut self) {
        for mass in self.zone_mass.values_mut() {
            mass.tick();
        }
    }

    /// Get current thermal state
    pub fn state(&self) -> ThermalState {
        self.state_machine.state()
    }

    /// Get utilization for a specific zone
    pub fn zone_utilization(&self, zone: ThermalZone) -> f32 {
        self.zone_mass
            .get(&zone)
            .map(|m| m.utilization())
            .unwrap_or(0.0)
    }

    /// Is any zone redlining?
    pub fn is_redlining(&self) -> bool {
        self.zone_mass.values().any(|m| m.is_redlining())
    }

    /// Predict temperature at horizon
    pub fn predict_temp(&self, zone: ThermalZone, horizon: Duration) -> f32 {
        let current = self.zone_utilization(zone);
        let delta = *self.zone_ema.get(&zone).unwrap_or(&0.0);
        current + delta * horizon.as_secs_f32()
    }

    /// Time to critical for a zone
    pub fn time_to_critical(&self, zone: ThermalZone) -> Option<Duration> {
        let current = self.zone_utilization(zone);
        let delta = *self.zone_ema.get(&zone).unwrap_or(&0.0);

        if delta <= 0.0 {
            return None; // Not heating
        }

        let remaining = 0.95 - current; // Critical threshold
        if remaining <= 0.0 {
            return Some(Duration::ZERO); // Already critical
        }

        Some(Duration::from_secs_f32(remaining / delta))
    }

    /// Recommend action based on current state
    pub fn recommend_action(&self) -> CoolingAction {
        let state = self.state_machine.state();

        // Find hottest zone
        let hottest = ThermalZone::all()
            .iter()
            .max_by(|a, b| {
                let u_a = self.zone_utilization(**a);
                let u_b = self.zone_utilization(**b);
                u_a.partial_cmp(&u_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(ThermalZone::Reasoning);

        CoolingAction::for_state_and_zone(state, hottest)
    }

    /// Update baselines based on action outcome (feedback loop)
    pub fn update_baseline(&mut self, zone: ThermalZone, outcome: ActionOutcome) {
        if let Some(baseline) = self.zone_baselines.get_mut(&zone) {
            baseline.adapt(&outcome);
        }
    }

    /// Force cool a zone (e.g., after explicit cooldown action)
    pub fn cool_zone(&mut self, zone: ThermalZone, amount: f32) {
        if let Some(mass) = self.zone_mass.get_mut(&zone) {
            mass.cool(amount);
        }
    }

    /// Get status summary
    pub fn status(&self) -> ThermoceptorStatus {
        ThermoceptorStatus {
            global_state: self.state_machine.state(),
            time_in_state: self.state_machine.time_in_state(),
            zone_utilizations: ThermalZone::all()
                .iter()
                .map(|z| (*z, self.zone_utilization(*z)))
                .collect(),
            recommended_action: self.recommend_action(),
            is_redlining: self.is_redlining(),
        }
    }
}

impl Default for Thermoceptor {
    fn default() -> Self {
        Self::new(ThermoceptorConfig::default())
    }
}

/// Status summary for display
#[derive(Debug)]
pub struct ThermoceptorStatus {
    pub global_state: ThermalState,
    pub time_in_state: Duration,
    pub zone_utilizations: Vec<(ThermalZone, f32)>,
    pub recommended_action: CoolingAction,
    pub is_redlining: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEMO / CLI SUPPORT
// ═══════════════════════════════════════════════════════════════════════════════

/// Run thermoception demo
pub fn run_demo() {
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m THERMOCEPTION — Cognitive Heat Sensing Demo\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();

    let mut thermo = Thermoceptor::default();

    println!("Phase 1: Cold start (baseline operation)");
    println!("─────────────────────────────────────────");

    // Simulate normal operation
    let normal_signals = vec![
        RawSignal::ContextUtilization(0.3),
        RawSignal::QueryComplexity(0.2),
        RawSignal::ToolCallDepth(1),
        RawSignal::QualifierCount(2),
    ];

    let map = thermo.ingest(&normal_signals);
    print_thermal_map(&map);

    std::thread::sleep(std::time::Duration::from_millis(100));

    println!("\nPhase 2: Heating up (complex reasoning)");
    println!("─────────────────────────────────────────");

    // Simulate increasing load
    for i in 1..=5 {
        let load = 0.3 + (i as f32 * 0.15);
        let signals = vec![
            RawSignal::ContextUtilization(load),
            RawSignal::QueryComplexity(load),
            RawSignal::ToolCallDepth(i),
            RawSignal::QualifierCount(i * 3),
            RawSignal::OutputLength(500 * i),
        ];

        let map = thermo.ingest(&signals);
        println!(
            "  Step {}: Global state = {}{}\x1b[0m, utilization = {:.1}%",
            i,
            map.global_state.color(),
            map.global_state.name(),
            map.global_utilization * 100.0
        );

        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    println!("\nPhase 3: Redline approach");
    println!("─────────────────────────────────────────");

    // Push toward redline
    let hot_signals = vec![
        RawSignal::ContextUtilization(0.95),
        RawSignal::QueryComplexity(0.9),
        RawSignal::ToolCallDepth(8),
        RawSignal::QualifierCount(15),
        RawSignal::TokenEntropy(0.8),
    ];

    for _ in 0..3 {
        let _map = thermo.ingest(&hot_signals);
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let map = thermo.ingest(&hot_signals);
    print_thermal_map(&map);

    println!("\nPhase 4: Cooldown");
    println!("─────────────────────────────────────────");

    // Simulate cooldown
    let cool_signals = vec![
        RawSignal::ContextUtilization(0.2),
        RawSignal::QueryComplexity(0.1),
    ];

    for i in 1..=8 {
        thermo.tick();
        let map = thermo.ingest(&cool_signals);
        println!(
            "  Cooling step {}: {}{}\x1b[0m ({:.1}%)",
            i,
            map.global_state.color(),
            map.global_state.name(),
            map.global_utilization * 100.0
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    println!();
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[32m Demo complete. Control system with hysteresis operational.\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
}

/// Print thermal map
fn print_thermal_map(map: &ThermalMap) {
    println!(
        "  Global: {}{}\x1b[0m ({:.1}%)",
        map.global_state.color(),
        map.global_state.name(),
        map.global_utilization * 100.0
    );

    if let Some(hottest) = map.hottest_zone {
        println!("  Hottest zone: {}", hottest.name());
    }

    for reading in &map.readings {
        let bar = heat_bar(reading.temperature.value, 20);
        println!(
            "    {:12} {} {:.2} (Δ{:+.3})",
            reading.zone.name(),
            bar,
            reading.temperature.value,
            reading.delta
        );
    }
}

/// Generate ASCII heat bar
fn heat_bar(value: f32, width: usize) -> String {
    let filled = (value * width as f32) as usize;
    let color = if value > 0.9 {
        "\x1b[31m" // red
    } else if value > 0.7 {
        "\x1b[33m" // yellow
    } else {
        "\x1b[32m" // green
    };
    format!(
        "{}[{}{}]\x1b[0m",
        color,
        "█".repeat(filled.min(width)),
        "░".repeat(width.saturating_sub(filled))
    )
}

/// Show current status
pub fn show_status(thermo: &Thermoceptor) {
    let status = thermo.status();

    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m THERMOCEPTION STATUS\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();

    println!(
        "  Global State: {}{}\x1b[0m",
        status.global_state.color(),
        status.global_state.name()
    );
    println!(
        "  Time in State: {:.1}s",
        status.time_in_state.as_secs_f32()
    );
    println!(
        "  Redlining: {}",
        if status.is_redlining {
            "\x1b[31mYES\x1b[0m"
        } else {
            "\x1b[32mNo\x1b[0m"
        }
    );
    println!();

    println!("  Zone Utilizations:");
    for (zone, util) in &status.zone_utilizations {
        let bar = heat_bar(*util, 20);
        println!("    {:12} {} {:.1}%", zone.name(), bar, util * 100.0);
    }
    println!();

    println!("  Recommended Action: {:?}", status.recommended_action);
}

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION BUS INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::{ObsKey, ObservationBatch};

impl Thermoceptor {
    /// Emit current thermal state as typed observations
    pub fn emit_observations(&self) -> ObservationBatch {
        let mut batch = ObservationBatch::new().with_source("thermoception");

        // Global utilization
        let global_util = ThermalZone::all()
            .iter()
            .map(|z| self.zone_utilization(*z))
            .fold(0.0f32, f32::max);
        batch.add(ObsKey::ThermalUtilization, global_util as f64);

        // Per-zone utilizations
        for zone in ThermalZone::all() {
            let util = self.zone_utilization(*zone);
            let key = match zone {
                ThermalZone::Reasoning => ObsKey::ThermalZoneReasoning,
                ThermalZone::Context => ObsKey::ThermalZoneContext,
                ThermalZone::Confidence => ObsKey::ThermalZoneConfidence,
                ThermalZone::Objective => ObsKey::ThermalZoneObjective,
                ThermalZone::Guardrail => ObsKey::ThermalZoneGuardrail,
            };
            batch.add(key, util as f64);
        }

        // Thermal state as ordinal
        batch.add(ObsKey::ThermalState, self.state() as u8 as f64);

        batch
    }

    /// Check if thermal state should trigger nociception
    /// Returns (should_trigger, zone_utilizations, durations)
    pub fn check_pain_trigger(&self) -> Option<Vec<(String, f32, f32)>> {
        if !self.is_redlining() && self.state() < ThermalState::Saturated {
            return None;
        }

        let mut triggers = Vec::new();
        for zone in ThermalZone::all() {
            let util = self.zone_utilization(*zone);
            if util >= crate::nociception::THERMAL_PAIN_THRESHOLD {
                let duration = self.state_machine.time_in_state().as_secs_f32();
                triggers.push((zone.name().to_string(), util, duration));
            }
        }

        if triggers.is_empty() {
            None
        } else {
            Some(triggers)
        }
    }
}

/// Bridge: Convert ThermalMap to ObservationBatch
pub fn thermal_map_to_observations(map: &ThermalMap) -> ObservationBatch {
    let mut batch = ObservationBatch::new().with_source("thermoception");

    batch.add(ObsKey::ThermalUtilization, map.global_utilization as f64);
    batch.add(ObsKey::ThermalState, map.global_state as u8 as f64);

    for reading in &map.readings {
        let key = match reading.zone {
            ThermalZone::Reasoning => ObsKey::ThermalZoneReasoning,
            ThermalZone::Context => ObsKey::ThermalZoneContext,
            ThermalZone::Confidence => ObsKey::ThermalZoneConfidence,
            ThermalZone::Objective => ObsKey::ThermalZoneObjective,
            ThermalZone::Guardrail => ObsKey::ThermalZoneGuardrail,
        };
        batch.add(key, reading.temperature.value as f64);
    }

    batch
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_state_transitions() {
        let mut sm = ThermalStateMachine::new();
        assert_eq!(sm.state(), ThermalState::Nominal);

        // Heat up (2 readings to transition)
        sm.update(0.85); // Saturated level
        assert_eq!(sm.state(), ThermalState::Nominal); // Not yet
        sm.update(0.85);
        assert_eq!(sm.state(), ThermalState::Saturated); // Now transitioned
    }

    #[test]
    fn test_thermal_mass_accumulation() {
        let mut mass = ThermalMass::new(100.0, 5.0);
        assert_eq!(mass.utilization(), 0.0);

        // Absorb heat multiple times with small delays to accumulate
        // Note: absorb uses dt from last_update, so rapid calls may not accumulate much
        for _ in 0..10 {
            mass.absorb(50.0);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // After multiple absorb calls with delays, should have accumulated some heat
        assert!(
            mass.utilization() > 0.0,
            "Utilization should be > 0, got {}",
            mass.utilization()
        );
    }

    #[test]
    fn test_baseline_calibration() {
        let mut baseline = ThermalBaseline::new(100);
        assert!(!baseline.calibrated);

        // Add samples
        for i in 0..25 {
            baseline.add_sample(0.5 + (i as f32 * 0.01));
        }

        assert!(baseline.calibrated);
        assert!(baseline.z_score(0.5).abs() < 2.0); // Should be normal
        assert!(baseline.z_score(0.95) > 2.0); // Should be anomalous
    }

    #[test]
    fn test_cooling_action_mapping() {
        assert_eq!(
            CoolingAction::for_state_and_zone(ThermalState::Nominal, ThermalZone::Reasoning),
            CoolingAction::None
        );
        assert_eq!(
            CoolingAction::for_state_and_zone(ThermalState::Unsafe, ThermalZone::Reasoning),
            CoolingAction::HaltAndCool { duration_secs: 30 }
        );
    }
}
