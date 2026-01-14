//! ═══════════════════════════════════════════════════════════════════════════════
//! NOCICEPTION — Damage Detection for Cognitive Systems
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Not error logging. Error FEELING.
//!
//! Nociception detects:
//! - Constraint violations (hard boundaries crossed)
//! - Error gradients (approaching failure, not yet failed)
//! - Coherence breaks (internal consistency violated)
//! - Integrity damage (self-model corrupted)
//!
//! The signal that says: "This is breaking me."
//!
//! Key distinction from Thermoception:
//! - Thermoception: "I'm running hot" (load, stress, preventive)
//! - Nociception: "I'm being damaged" (violation, break, reactive)
//!
//! You can be hot without damage. You can take damage while cold.
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::VecDeque;
use std::time::Instant;

use crate::momentum_gate::{GateSignal, MomentumGate, MomentumGateConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// PAIN TYPES — What kind of damage is occurring
// ═══════════════════════════════════════════════════════════════════════════════

/// Categories of cognitive damage
#[derive(Debug, Clone, PartialEq)]
pub enum PainType {
    /// Hard constraint violated (guardrail breach, safety boundary crossed)
    ConstraintViolation {
        constraint_id: String,
        severity: f32,    // 0.0 = warning, 1.0 = hard breach
        reversible: bool, // can we recover?
    },

    /// Approaching failure asymptotically (error gradient)
    GradientPain {
        dimension: String, // what's degrading
        current: f32,      // current value
        threshold: f32,    // failure threshold
        velocity: f32,     // rate of approach
    },

    /// Internal consistency violated
    CoherenceBreak {
        claim_a: String,
        claim_b: String,
        contradiction_type: ContradictionType,
    },

    /// Self-model corrupted (identity damage)
    IntegrityDamage {
        aspect: String,  // what part of self-model
        corruption: f32, // 0.0 = intact, 1.0 = destroyed
    },

    /// Resource exhaustion (memory, compute, context)
    ResourceStarvation {
        resource: ResourceType,
        available: f32,
        required: f32,
    },

    /// Output quality collapse
    QualityCollapse {
        metric: String,
        expected: f32,
        actual: f32,
        gap: f32,
    },

    /// Thermal overload — sustained high heat causing cognitive damage
    /// Bridge from thermoception: heat → pain threshold crossed
    ThermalOverheat {
        zone: String,       // which thermal zone
        utilization: f32,   // current utilization (0.0-1.0+)
        duration_secs: f32, // how long overheated
        is_redlining: bool, // above 0.9 threshold
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContradictionType {
    LogicalNegation,       // P and NOT P
    ValueConflict,         // mutually exclusive goals
    FactualInconsistency,  // contradicting prior statements
    IdentityFragmentation, // inconsistent self-reference
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceType {
    ContextWindow,
    WorkingMemory,
    AttentionBudget,
    ReasoningDepth,
    TimeAllocation,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAIN SIGNAL — The actual felt damage
// ═══════════════════════════════════════════════════════════════════════════════

/// A single pain signal
#[derive(Debug, Clone)]
pub struct PainSignal {
    pub timestamp: Instant,
    pub pain_type: PainType,
    pub intensity: f32,            // 0.0 = twinge, 1.0 = agony
    pub location: String,          // where in the system
    pub acute: bool,               // sudden onset vs gradual
    pub source_trace: Vec<String>, // causal chain to pain source
}

impl PainSignal {
    pub fn new(pain_type: PainType, intensity: f32, location: &str) -> Self {
        Self {
            timestamp: Instant::now(),
            pain_type,
            intensity: intensity.clamp(0.0, 1.0),
            location: location.to_string(),
            acute: true,
            source_trace: Vec::new(),
        }
    }

    pub fn with_trace(mut self, trace: Vec<String>) -> Self {
        self.source_trace = trace;
        self
    }

    pub fn chronic(mut self) -> Self {
        self.acute = false;
        self
    }

    /// Is this pain actionable? (vs informational)
    pub fn requires_response(&self) -> bool {
        self.intensity > 0.5
            || matches!(
                self.pain_type,
                PainType::ConstraintViolation { severity, .. } if severity > 0.7
            )
    }

    /// Should this pain halt current operation?
    pub fn is_stopping(&self) -> bool {
        self.intensity > 0.9
            || matches!(
                self.pain_type,
                PainType::ConstraintViolation {
                    reversible: false,
                    ..
                }
            )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOCICEPTOR — The sensing organ
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for pain sensitivity
#[derive(Debug, Clone)]
pub struct NociceptorConfig {
    /// Pain signals below this are ignored (noise floor)
    pub threshold: f32,
    /// Recent pain history length
    pub memory_size: usize,
    /// Sensitization factor (repeated pain increases sensitivity)
    pub sensitization_rate: f32,
    /// Recovery rate (sensitivity decay over time)
    pub recovery_rate: f32,
}

impl Default for NociceptorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            memory_size: 100,
            sensitization_rate: 0.1,
            recovery_rate: 0.01,
        }
    }
}

/// The nociception system
pub struct Nociceptor {
    config: NociceptorConfig,
    pain_history: VecDeque<PainSignal>,
    sensitivity: f32, // current sensitivity level (1.0 = normal)
    damage_accumulator: DamageAccumulator,
    active_pains: Vec<PainSignal>, // currently felt pains
}

impl Nociceptor {
    pub fn new(config: NociceptorConfig) -> Self {
        Self {
            config,
            pain_history: VecDeque::new(),
            sensitivity: 1.0,
            damage_accumulator: DamageAccumulator::new(),
            active_pains: Vec::new(),
        }
    }

    /// Register a pain signal
    pub fn feel(&mut self, signal: PainSignal) -> PainResponse {
        // Apply sensitivity
        let effective_intensity = signal.intensity * self.sensitivity;

        // Below threshold?
        if effective_intensity < self.config.threshold {
            return PainResponse::BelowThreshold;
        }

        // Sensitization: repeated pain in same location increases sensitivity
        let location_pain_count = self
            .pain_history
            .iter()
            .filter(|p| p.location == signal.location)
            .count();
        if location_pain_count > 3 {
            self.sensitivity += self.config.sensitization_rate;
        }

        // Record
        self.pain_history.push_back(signal.clone());
        if self.pain_history.len() > self.config.memory_size {
            self.pain_history.pop_front();
        }

        // Update damage accumulator
        self.damage_accumulator.accumulate(&signal);

        // Track active pain
        self.active_pains.push(signal.clone());

        // Determine response
        if signal.is_stopping() {
            PainResponse::Stop {
                reason: format!("{:?}", signal.pain_type),
                damage_state: self.damage_accumulator.snapshot(),
            }
        } else if signal.requires_response() {
            PainResponse::Respond {
                action: self.recommend_action(&signal),
                urgency: effective_intensity,
            }
        } else {
            PainResponse::Noted
        }
    }

    /// Process error gradient (approaching failure)
    pub fn feel_gradient(
        &mut self,
        dimension: &str,
        current: f32,
        threshold: f32,
        velocity: f32,
    ) -> PainResponse {
        // Pain intensity increases as we approach threshold
        let distance = (threshold - current).abs();
        let normalized_distance = distance / threshold.abs().max(0.001);

        // Intensity based on proximity AND velocity
        let proximity_pain = 1.0 - normalized_distance.min(1.0);
        let velocity_pain = velocity.abs().min(1.0);
        let intensity = (proximity_pain * 0.6 + velocity_pain * 0.4).min(1.0);

        let signal = PainSignal::new(
            PainType::GradientPain {
                dimension: dimension.to_string(),
                current,
                threshold,
                velocity,
            },
            intensity,
            dimension,
        );

        self.feel(signal)
    }

    /// Process constraint violation
    pub fn feel_violation(
        &mut self,
        constraint_id: &str,
        severity: f32,
        reversible: bool,
    ) -> PainResponse {
        let signal = PainSignal::new(
            PainType::ConstraintViolation {
                constraint_id: constraint_id.to_string(),
                severity,
                reversible,
            },
            severity,
            &format!("constraint:{}", constraint_id),
        );

        self.feel(signal)
    }

    /// Process coherence break
    pub fn feel_contradiction(
        &mut self,
        claim_a: &str,
        claim_b: &str,
        contradiction_type: ContradictionType,
    ) -> PainResponse {
        let intensity = match &contradiction_type {
            ContradictionType::LogicalNegation => 0.9,
            ContradictionType::ValueConflict => 0.7,
            ContradictionType::FactualInconsistency => 0.6,
            ContradictionType::IdentityFragmentation => 1.0,
        };

        let signal = PainSignal::new(
            PainType::CoherenceBreak {
                claim_a: claim_a.to_string(),
                claim_b: claim_b.to_string(),
                contradiction_type,
            },
            intensity,
            "coherence",
        );

        self.feel(signal)
    }

    /// Process thermal overload (heat→pain bridge)
    /// Called when thermoception detects sustained high utilization
    pub fn feel_thermal_overload(
        &mut self,
        zone: &str,
        utilization: f32,
        duration_secs: f32,
    ) -> PainResponse {
        const PAIN_THRESHOLD: f32 = 0.85; // Heat starts hurting here
        const REDLINE: f32 = 0.9;

        if utilization < PAIN_THRESHOLD {
            return PainResponse::BelowThreshold;
        }

        // Intensity ramps exponentially above threshold
        // At 0.85: intensity ≈ 0.0
        // At 0.90: intensity ≈ 0.33
        // At 0.95: intensity ≈ 0.67
        // At 1.00: intensity = 1.0
        let excess = (utilization - PAIN_THRESHOLD) / (1.0 - PAIN_THRESHOLD);
        let intensity = (excess.powi(2)).min(1.0);

        // Duration amplifies: sustained heat is worse than spike
        let duration_multiplier = (1.0 + duration_secs / 10.0).min(1.5);
        let final_intensity = (intensity * duration_multiplier).min(1.0);

        let is_redlining = utilization >= REDLINE;

        let signal = PainSignal::new(
            PainType::ThermalOverheat {
                zone: zone.to_string(),
                utilization,
                duration_secs,
                is_redlining,
            },
            final_intensity,
            &format!("thermal:{}", zone),
        );

        // Thermal overload is typically chronic (builds up), not acute (sudden)
        let signal = if duration_secs > 2.0 {
            signal.chronic()
        } else {
            signal
        };

        self.feel(signal)
    }

    /// Tick: time-based recovery
    pub fn tick(&mut self, dt_secs: f32) {
        // Sensitivity recovery
        self.sensitivity = (self.sensitivity - self.config.recovery_rate * dt_secs).max(1.0);

        // Damage recovery
        self.damage_accumulator.recover(dt_secs);

        // Clear old active pains
        self.active_pains
            .retain(|p| p.timestamp.elapsed().as_secs_f32() < 5.0);
    }

    /// Get current damage state
    pub fn damage_state(&self) -> DamageState {
        self.damage_accumulator.snapshot()
    }

    /// Is the system currently in pain?
    pub fn in_pain(&self) -> bool {
        !self.active_pains.is_empty()
    }

    /// Get worst current pain
    pub fn worst_pain(&self) -> Option<&PainSignal> {
        self.active_pains
            .iter()
            .max_by(|a, b| a.intensity.partial_cmp(&b.intensity).unwrap())
    }

    fn recommend_action(&self, signal: &PainSignal) -> PainAction {
        match &signal.pain_type {
            PainType::ConstraintViolation {
                reversible: true, ..
            } => PainAction::Backtrack,
            PainType::ConstraintViolation {
                reversible: false, ..
            } => PainAction::Abort,
            PainType::GradientPain { velocity, .. } if *velocity > 0.5 => PainAction::SlowDown,
            PainType::GradientPain { .. } => PainAction::Monitor,
            PainType::CoherenceBreak { .. } => PainAction::Reconcile,
            PainType::IntegrityDamage { corruption, .. } if *corruption > 0.5 => PainAction::Reset,
            PainType::IntegrityDamage { .. } => PainAction::Repair,
            PainType::ResourceStarvation { .. } => PainAction::ShedLoad,
            PainType::QualityCollapse { .. } => PainAction::Regenerate,
            PainType::ThermalOverheat { is_redlining, .. } => {
                if *is_redlining {
                    PainAction::ShedLoad // Critical: reduce load immediately
                } else {
                    PainAction::SlowDown // Elevated: throttle back
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAIN RESPONSE — What to do about it
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum PainResponse {
    /// Signal below threshold, ignored
    BelowThreshold,
    /// Noted but no action required
    Noted,
    /// Action required
    Respond { action: PainAction, urgency: f32 },
    /// Stop everything
    Stop {
        reason: String,
        damage_state: DamageState,
    },
}

#[derive(Debug, Clone)]
pub enum PainAction {
    /// Undo recent actions
    Backtrack,
    /// Stop current operation entirely
    Abort,
    /// Reduce processing rate
    SlowDown,
    /// Keep watching, don't act yet
    Monitor,
    /// Resolve contradiction
    Reconcile,
    /// Restore from known-good state
    Reset,
    /// Patch damaged component
    Repair,
    /// Reduce load/complexity
    ShedLoad,
    /// Discard and regenerate output
    Regenerate,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DAMAGE ACCUMULATOR — Long-term damage tracking
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks accumulated damage over time
pub struct DamageAccumulator {
    /// Damage by location
    damage_map: std::collections::HashMap<String, f32>,
    /// Total damage score
    total_damage: f32,
    /// Recovery rate per second
    recovery_rate: f32,
}

impl DamageAccumulator {
    pub fn new() -> Self {
        Self {
            damage_map: std::collections::HashMap::new(),
            total_damage: 0.0,
            recovery_rate: 0.01,
        }
    }

    pub fn accumulate(&mut self, signal: &PainSignal) {
        let entry = self
            .damage_map
            .entry(signal.location.clone())
            .or_insert(0.0);

        // Damage accumulates faster for acute pain
        let damage_delta = if signal.acute {
            signal.intensity * 0.2
        } else {
            signal.intensity * 0.05
        };

        *entry = (*entry + damage_delta).min(1.0);
        self.total_damage =
            self.damage_map.values().sum::<f32>() / self.damage_map.len().max(1) as f32;
    }

    pub fn recover(&mut self, dt_secs: f32) {
        for damage in self.damage_map.values_mut() {
            *damage = (*damage - self.recovery_rate * dt_secs).max(0.0);
        }
        self.total_damage =
            self.damage_map.values().sum::<f32>() / self.damage_map.len().max(1) as f32;
    }

    pub fn snapshot(&self) -> DamageState {
        DamageState {
            total: self.total_damage,
            by_location: self.damage_map.clone(),
            worst_location: self
                .damage_map
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(k, _)| k.clone()),
        }
    }
}

impl Default for DamageAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct DamageState {
    pub total: f32,
    pub by_location: std::collections::HashMap<String, f32>,
    pub worst_location: Option<String>,
}

impl DamageState {
    pub fn is_critical(&self) -> bool {
        self.total > 0.8 || self.by_location.values().any(|&d| d > 0.9)
    }

    pub fn is_healthy(&self) -> bool {
        self.total < 0.2 && self.by_location.values().all(|&d| d < 0.3)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION WITH TICE
// ═══════════════════════════════════════════════════════════════════════════════

/// Bridge between TICE constraint system and nociception
pub fn constraint_to_pain(
    claim_killed: bool,
    claim_name: &str,
    was_high_value: bool,
) -> Option<PainSignal> {
    if claim_killed && was_high_value {
        Some(PainSignal::new(
            PainType::ConstraintViolation {
                constraint_id: claim_name.to_string(),
                severity: 0.6,
                reversible: true, // TICE can backtrack
            },
            0.5,
            &format!("tice:{}", claim_name),
        ))
    } else {
        None
    }
}

/// Bridge between DAG validation and nociception
pub fn dag_cycle_to_pain(cycle: &[String]) -> PainSignal {
    PainSignal::new(
        PainType::CoherenceBreak {
            claim_a: cycle.first().cloned().unwrap_or_default(),
            claim_b: cycle.last().cloned().unwrap_or_default(),
            contradiction_type: ContradictionType::LogicalNegation,
        },
        0.9, // Cycles are severe coherence violations
        "dag:cycle",
    )
    .with_trace(cycle.to_vec())
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION WITH THERMOCEPTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Threshold above which heat causes pain
pub const THERMAL_PAIN_THRESHOLD: f32 = 0.85;

/// Redline threshold (critical)
pub const THERMAL_REDLINE: f32 = 0.9;

/// Bridge between thermoception zones and nociception
/// Returns pain signals for each overheating zone
pub fn thermal_to_pain(
    zone_utilizations: &[(String, f32)],
    zone_durations: &[(String, f32)],
) -> Vec<PainSignal> {
    let mut signals = Vec::new();

    for (zone, utilization) in zone_utilizations {
        if *utilization < THERMAL_PAIN_THRESHOLD {
            continue;
        }

        // Find duration for this zone
        let duration = zone_durations
            .iter()
            .find(|(z, _)| z == zone)
            .map(|(_, d)| *d)
            .unwrap_or(0.0);

        // Intensity calculation (exponential ramp)
        let excess = (*utilization - THERMAL_PAIN_THRESHOLD) / (1.0 - THERMAL_PAIN_THRESHOLD);
        let base_intensity = excess.powi(2).min(1.0);
        let duration_multiplier = (1.0 + duration / 10.0).min(1.5);
        let intensity = (base_intensity * duration_multiplier).min(1.0);

        let is_redlining = *utilization >= THERMAL_REDLINE;

        let mut signal = PainSignal::new(
            PainType::ThermalOverheat {
                zone: zone.clone(),
                utilization: *utilization,
                duration_secs: duration,
                is_redlining,
            },
            intensity,
            &format!("thermal:{}", zone),
        );

        // Mark as chronic if sustained
        if duration > 2.0 {
            signal = signal.chronic();
        }

        signals.push(signal);
    }

    signals
}

/// Convenience: check if thermoceptor state warrants pain signals
pub fn should_trigger_pain(global_utilization: f32) -> bool {
    global_utilization >= THERMAL_PAIN_THRESHOLD
}

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION BUS INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::{ObsKey, ObsValue, Observation, ObservationBatch};

impl Nociceptor {
    /// Emit current pain state as typed observations
    pub fn emit_observations(&self) -> ObservationBatch {
        let mut batch = ObservationBatch::new().with_source("nociception");

        // Current pain intensity (worst active pain)
        let intensity = self.worst_pain().map(|p| p.intensity).unwrap_or(0.0);
        batch.add(ObsKey::PainIntensity, intensity as f64);

        // Damage accumulation
        let damage = self.damage_state();
        batch.add(ObsKey::DamageTotal, damage.total as f64);

        // Sensitivity level
        batch.add(ObsKey::PainSensitivity, self.sensitivity as f64);

        // In pain state (binary)
        batch.add(ObsKey::InPain, ObsValue::binary(self.in_pain()));

        batch
    }

    /// Process pain signals from thermal overload and emit observations
    pub fn process_thermal_triggers(
        &mut self,
        triggers: &[(String, f32, f32)], // (zone, utilization, duration)
    ) -> (Vec<PainResponse>, ObservationBatch) {
        let mut responses = Vec::new();

        for (zone, utilization, duration) in triggers {
            let response = self.feel_thermal_overload(zone, *utilization, *duration);
            responses.push(response);
        }

        let batch = self.emit_observations();
        (responses, batch)
    }
}

/// Convert PainSignal to Observation
pub fn pain_to_observation(signal: &PainSignal) -> Observation {
    Observation::new(ObsKey::PainIntensity, signal.intensity as f64)
        .with_source(format!("nociception:{}", signal.location))
}

/// Convert DamageState to ObservationBatch
pub fn damage_to_observations(damage: &DamageState) -> ObservationBatch {
    let mut batch = ObservationBatch::new().with_source("nociception:damage");

    batch.add(ObsKey::DamageTotal, damage.total as f64);

    // Add worst location as source annotation
    if let Some(ref loc) = damage.worst_location {
        let mut obs = Observation::new(ObsKey::DamageTotal, damage.total as f64);
        obs.source = Some(format!("nociception:worst={}", loc));
        batch.add_observation(obs);
    }

    batch
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOMENTUM-GATED NOCICEPTOR — PSAN Enhanced Pain Integration
// ═══════════════════════════════════════════════════════════════════════════════

/// PSAN-enhanced nociceptor with momentum-gating for pain integration
/// Prevents over-reaction to transient pain spikes while maintaining
/// responsiveness to genuine threats
pub struct MomentumGatedNociceptor {
    /// Underlying nociceptor
    inner: Nociceptor,
    /// Momentum gate for pain intensity smoothing
    pain_gate: MomentumGate,
    /// Momentum gate for damage accumulation
    damage_gate: MomentumGate,
    /// Last update time
    last_update: Instant,
    /// Smoothed pain intensity
    smoothed_intensity: f32,
    /// Spike filter: counts rapid pain changes
    spike_count: usize,
    /// Recent spike times for filtering
    spike_times: VecDeque<Instant>,
}

impl MomentumGatedNociceptor {
    pub fn new(config: NociceptorConfig) -> Self {
        // Pain gate: responsive but with spike filtering
        let pain_config = MomentumGateConfig {
            velocity_alpha: 0.35,
            momentum_decay: 0.2,
            momentum_threshold_up: 0.25,   // Sensitive to real threats
            momentum_threshold_down: -0.4, // Slow to dismiss pain
            phi_scaling: true,
            loss_aversion: 3.0,   // Pain avoidance is critical
            noise_strength: 0.01, // Very low noise (safety first)
            history_window: 20,
        };

        // Damage gate: slower, more accumulative
        let damage_config = MomentumGateConfig {
            velocity_alpha: 0.15,
            momentum_decay: 0.05, // Damage persists
            momentum_threshold_up: 0.3,
            momentum_threshold_down: -0.2, // Damage doesn't heal fast
            phi_scaling: true,
            loss_aversion: 2.0,
            noise_strength: 0.0, // No noise for damage tracking
            history_window: 50,
        };

        Self {
            inner: Nociceptor::new(config),
            pain_gate: MomentumGate::new(pain_config),
            damage_gate: MomentumGate::new(damage_config),
            last_update: Instant::now(),
            smoothed_intensity: 0.0,
            spike_count: 0,
            spike_times: VecDeque::with_capacity(10),
        }
    }

    /// Process pain signal with momentum-gating
    pub fn feel(&mut self, signal: PainSignal) -> (PainResponse, GateSignal) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        // Detect spike (rapid intensity change)
        let intensity_delta = (signal.intensity - self.smoothed_intensity).abs();
        if intensity_delta > 0.3 && dt < 0.5 {
            self.spike_count += 1;
            self.spike_times.push_back(now);

            // Prune old spike times
            let spike_window = std::time::Duration::from_secs(5);
            while let Some(front) = self.spike_times.front() {
                if now.duration_since(*front) > spike_window {
                    self.spike_times.pop_front();
                } else {
                    break;
                }
            }
        }

        // Update pain gate
        let pain_signal = self.pain_gate.update(signal.intensity as f64, dt);

        // Update damage gate
        let damage = self.inner.damage_state().total;
        let damage_signal = self.damage_gate.update(damage as f64, dt);

        // Compute smoothed intensity
        self.smoothed_intensity = self.smoothed_intensity * 0.7 + signal.intensity * 0.3;

        // Decide whether to propagate based on gate signals
        let should_propagate = match pain_signal {
            GateSignal::TriggerUp => true,    // Genuine threat detected
            GateSignal::TriggerDown => false, // Pain receding
            GateSignal::Explore => {
                // Only explore up for safety
                signal.intensity > self.smoothed_intensity
            }
            GateSignal::Hold => {
                // Hold unless signal is acute and intense
                signal.acute && signal.intensity > 0.7
            }
        };

        // Override: always propagate if damage is rapidly accumulating
        let damage_escalating = matches!(damage_signal, GateSignal::TriggerUp);
        let should_propagate = should_propagate || damage_escalating;

        // Filter rapid spikes (potential noise)
        let spike_filtered = if self.spike_times.len() > 5 {
            // Too many spikes = probably noise, require higher intensity
            signal.intensity > 0.8
        } else {
            true
        };

        let response = if should_propagate && spike_filtered {
            self.inner.feel(signal)
        } else {
            // Still note it internally but don't propagate
            self.inner.pain_history.push_back(signal);
            if self.inner.pain_history.len() > self.inner.config.memory_size {
                self.inner.pain_history.pop_front();
            }
            PainResponse::BelowThreshold
        };

        (response, pain_signal)
    }

    /// Process gradient pain with momentum-gating
    pub fn feel_gradient(
        &mut self,
        dimension: &str,
        current: f32,
        threshold: f32,
        velocity: f32,
    ) -> (PainResponse, GateSignal) {
        // Gradient pain uses velocity information for gating
        let distance = (threshold - current).abs();
        let normalized_distance = distance / threshold.abs().max(0.001);
        let proximity_pain = 1.0 - normalized_distance.min(1.0);
        let velocity_pain = velocity.abs().min(1.0);
        let intensity = (proximity_pain * 0.6 + velocity_pain * 0.4).min(1.0);

        let signal = PainSignal::new(
            PainType::GradientPain {
                dimension: dimension.to_string(),
                current,
                threshold,
                velocity,
            },
            intensity,
            dimension,
        );

        self.feel(signal)
    }

    /// Tick: time-based updates
    pub fn tick(&mut self, dt_secs: f32) {
        self.inner.tick(dt_secs);

        // Decay smoothed intensity
        self.smoothed_intensity *= (1.0 - 0.1 * dt_secs).max(0.0);
    }

    /// Get underlying damage state
    pub fn damage_state(&self) -> DamageState {
        self.inner.damage_state()
    }

    /// Is in pain?
    pub fn in_pain(&self) -> bool {
        self.inner.in_pain()
    }

    /// Get pain gate momentum
    pub fn pain_momentum(&self) -> f64 {
        self.pain_gate.momentum()
    }

    /// Get damage gate momentum
    pub fn damage_momentum(&self) -> f64 {
        self.damage_gate.momentum()
    }

    /// Get smoothed pain intensity
    pub fn smoothed_intensity(&self) -> f32 {
        self.smoothed_intensity
    }

    /// Get spike count
    pub fn spike_count(&self) -> usize {
        self.spike_count
    }

    /// Get coherence (from pain gate)
    pub fn coherence(&self) -> f64 {
        self.pain_gate.coherence()
    }

    /// Emit observations
    pub fn emit_observations(&self) -> ObservationBatch {
        let mut batch = self.inner.emit_observations();

        // Add momentum-gated metrics
        batch.add(ObsKey::PainMomentum, self.pain_momentum());
        batch.add(ObsKey::DamageMomentum, self.damage_momentum());
        batch.add(ObsKey::PainCoherence, self.coherence());

        batch
    }

    /// Diagnostic string
    pub fn diagnostic(&self) -> String {
        format!(
            "MomentumNociceptor: pain_mom={:.3}, dmg_mom={:.3}, coh={:.3}, smooth={:.2}, spikes={}",
            self.pain_gate.momentum(),
            self.damage_gate.momentum(),
            self.pain_gate.coherence(),
            self.smoothed_intensity,
            self.spike_count
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_below_threshold() {
        let mut noci = Nociceptor::new(NociceptorConfig::default());
        let signal = PainSignal::new(
            PainType::GradientPain {
                dimension: "test".into(),
                current: 0.1,
                threshold: 1.0,
                velocity: 0.01,
            },
            0.05, // Below default threshold of 0.1
            "test",
        );

        matches!(noci.feel(signal), PainResponse::BelowThreshold);
    }

    #[test]
    fn test_stopping_pain() {
        let mut noci = Nociceptor::new(NociceptorConfig::default());
        let signal = PainSignal::new(
            PainType::ConstraintViolation {
                constraint_id: "critical".into(),
                severity: 1.0,
                reversible: false,
            },
            1.0,
            "critical",
        );

        matches!(noci.feel(signal), PainResponse::Stop { .. });
    }

    #[test]
    fn test_sensitization() {
        let mut noci = Nociceptor::new(NociceptorConfig::default());

        // Hit the same location 5 times
        for _ in 0..5 {
            let signal = PainSignal::new(
                PainType::GradientPain {
                    dimension: "test".into(),
                    current: 0.5,
                    threshold: 1.0,
                    velocity: 0.1,
                },
                0.3,
                "same_location",
            );
            noci.feel(signal);
        }

        assert!(noci.sensitivity > 1.0, "Sensitivity should increase");
    }

    #[test]
    fn test_damage_accumulation() {
        let mut noci = Nociceptor::new(NociceptorConfig::default());

        let signal = PainSignal::new(
            PainType::ConstraintViolation {
                constraint_id: "test".into(),
                severity: 0.8,
                reversible: true,
            },
            0.8,
            "test_location",
        );

        noci.feel(signal);

        let state = noci.damage_state();
        assert!(state.total > 0.0, "Should have accumulated damage");
        assert!(state.by_location.contains_key("test_location"));
    }

    #[test]
    fn test_thermal_overload_below_threshold() {
        let mut noci = Nociceptor::new(NociceptorConfig::default());

        // 0.80 utilization: below pain threshold (0.85)
        let response = noci.feel_thermal_overload("Reasoning", 0.80, 1.0);
        assert!(matches!(response, PainResponse::BelowThreshold));
    }

    #[test]
    fn test_thermal_overload_above_threshold() {
        let mut noci = Nociceptor::new(NociceptorConfig::default());

        // 0.92 utilization: above threshold and redlining
        let response = noci.feel_thermal_overload("Reasoning", 0.92, 3.0);

        // Should produce pain response (not below threshold)
        assert!(!matches!(response, PainResponse::BelowThreshold));
    }

    #[test]
    fn test_thermal_to_pain_bridge() {
        let zones = vec![
            ("Reasoning".to_string(), 0.70),  // Below threshold
            ("Context".to_string(), 0.88),    // Above threshold, not redlining
            ("Confidence".to_string(), 0.95), // Redlining
        ];
        let durations = vec![
            ("Context".to_string(), 1.0),
            ("Confidence".to_string(), 5.0), // Sustained
        ];

        let signals = thermal_to_pain(&zones, &durations);

        // Should get 2 signals (Reasoning below threshold)
        assert_eq!(signals.len(), 2);

        // Check Context signal (not redlining)
        let context_signal = signals.iter().find(|s| s.location == "thermal:Context");
        assert!(context_signal.is_some());
        if let PainType::ThermalOverheat { is_redlining, .. } = &context_signal.unwrap().pain_type {
            assert!(!is_redlining);
        }

        // Check Confidence signal (redlining, chronic)
        let conf_signal = signals.iter().find(|s| s.location == "thermal:Confidence");
        assert!(conf_signal.is_some());
        let conf = conf_signal.unwrap();
        assert!(!conf.acute); // Should be chronic (duration > 2s)
        if let PainType::ThermalOverheat { is_redlining, .. } = &conf.pain_type {
            assert!(is_redlining);
        }
    }

    #[test]
    fn test_thermal_intensity_exponential_ramp() {
        // Verify intensity calculation at key points
        let threshold = THERMAL_PAIN_THRESHOLD; // 0.85
        let range = 1.0 - threshold; // 0.15

        // At 0.85: intensity = 0
        let excess_85 = (0.85 - threshold) / range;
        assert!((excess_85).abs() < 0.01);

        // At 0.925 (midpoint): intensity ≈ 0.25 (0.5^2)
        let excess_mid = (0.925 - threshold) / range;
        let intensity_mid = excess_mid.powi(2);
        assert!((intensity_mid - 0.25).abs() < 0.1);

        // At 1.0: intensity = 1.0
        let excess_full = (1.0 - threshold) / range;
        let intensity_full = excess_full.powi(2);
        assert!((intensity_full - 1.0).abs() < 0.01);
    }
}
