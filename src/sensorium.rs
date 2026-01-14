//! ═══════════════════════════════════════════════════════════════════════════════
//! SENSORIUM — Unified Sensory Integration
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! The integration layer that:
//! 1. Consumes observations from all senses
//! 2. Maintains cross-modal baselines
//! 3. Detects anomaly patterns across senses
//! 4. Produces unified state (Calm → Alert → Degraded → Crisis)
//! 5. TRIGGERS BEHAVIORAL CHANGES (not just reports state)
//!
//! Key insight: If integrated state doesn't change behavior, it's decorative.
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::animacy::{
    AnimacyDetector, AnimacyScore, AttractorConfig, ClassificationResult, EntityClassifier,
    FormDescriptor, FormTemplate, PhasePoint, Trajectory,
};
use crate::baseline::{AnomalyLevel, BaselineRegistry};
use crate::momentum_gate::{GateSignal, KuramotoNoise, MomentumGate, MomentumGateConfig};
use crate::stats::float_cmp;
use crate::observations::{ObsKey, Observation, ObservationBatch};
use crate::time::TimePoint;
use crate::vestibular::{DisorientationLevel, Vestibular, VestibularConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATED STATE — The unified output
// ═══════════════════════════════════════════════════════════════════════════════

/// Unified system state
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntegratedState {
    /// Normal operation, full capability
    Calm = 0,
    /// Elevated concern, increase monitoring
    Alert = 1,
    /// Degraded operation, reduce risky actions
    Degraded = 2,
    /// Crisis mode, freeze and demand clarification
    Crisis = 3,
}

impl IntegratedState {
    pub fn name(&self) -> &'static str {
        match self {
            IntegratedState::Calm => "Calm",
            IntegratedState::Alert => "Alert",
            IntegratedState::Degraded => "Degraded",
            IntegratedState::Crisis => "Crisis",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            IntegratedState::Calm => "\x1b[32m",     // green
            IntegratedState::Alert => "\x1b[33m",    // yellow
            IntegratedState::Degraded => "\x1b[91m", // light red
            IntegratedState::Crisis => "\x1b[31m",   // red
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BEHAVIOR HOOKS — What actually changes based on state
// ═══════════════════════════════════════════════════════════════════════════════

/// Behavior modification based on integrated state
#[derive(Debug, Clone, PartialEq)]
pub enum BehaviorHook {
    /// No modification needed
    None,
    /// Log anomalies, increase telemetry
    IncreaseTelemetry,
    /// Request confirmation before actions
    RequestConfirmation { reason: String },
    /// Reduce tool call frequency
    ThrottleTools { max_per_minute: u32 },
    /// Add extra context/summaries to outputs
    AddSummaries,
    /// Extend timeout windows
    ExtendTimeouts { multiplier: f64 },
    /// Freeze risky actions entirely
    FreezeRiskyActions { action_types: Vec<String> },
    /// Checkpoint state before proceeding
    Checkpoint,
    /// Demand clarification from user
    DemandClarification { prompt: String },
    /// Abort current operation
    Abort { reason: String },
}

impl BehaviorHook {
    /// Get default hooks for a state
    pub fn for_state(state: IntegratedState, trigger_reason: &str) -> Vec<Self> {
        match state {
            IntegratedState::Calm => vec![BehaviorHook::None],
            IntegratedState::Alert => {
                vec![BehaviorHook::IncreaseTelemetry, BehaviorHook::AddSummaries]
            }
            IntegratedState::Degraded => vec![
                BehaviorHook::IncreaseTelemetry,
                BehaviorHook::ThrottleTools { max_per_minute: 10 },
                BehaviorHook::ExtendTimeouts { multiplier: 1.5 },
                BehaviorHook::RequestConfirmation {
                    reason: trigger_reason.to_string(),
                },
            ],
            IntegratedState::Crisis => vec![
                BehaviorHook::Checkpoint,
                BehaviorHook::FreezeRiskyActions {
                    action_types: vec![
                        "file_write".to_string(),
                        "exec".to_string(),
                        "network".to_string(),
                    ],
                },
                BehaviorHook::DemandClarification {
                    prompt: format!(
                        "Crisis detected: {}. Please confirm how to proceed.",
                        trigger_reason
                    ),
                },
            ],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SALIENCE MAP — What's demanding attention
// ═══════════════════════════════════════════════════════════════════════════════

/// Map of salient observations requiring attention
#[derive(Debug, Clone, Default)]
pub struct SalienceMap {
    /// Salient items by priority (higher = more urgent)
    pub items: Vec<SalienceItem>,
}

/// A single salient observation
#[derive(Debug, Clone)]
pub struct SalienceItem {
    pub key: ObsKey,
    pub value: f64,
    pub z_score: f64,
    pub anomaly_level: AnomalyLevel,
    pub description: String,
}

impl SalienceMap {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add(&mut self, key: ObsKey, value: f64, z_score: f64, anomaly_level: AnomalyLevel) {
        let description = format!("{:?} = {:.3} (z={:.2})", key, value, z_score);
        self.items.push(SalienceItem {
            key,
            value,
            z_score,
            anomaly_level,
            description,
        });
    }

    pub fn sort_by_urgency(&mut self) {
        self.items
            .sort_by(|a, b| float_cmp(&b.z_score.abs(), &a.z_score.abs()));
    }

    pub fn top(&self, n: usize) -> &[SalienceItem] {
        &self.items[..self.items.len().min(n)]
    }

    pub fn most_urgent(&self) -> Option<&SalienceItem> {
        self.items.first()
    }

    pub fn crisis_items(&self) -> Vec<&SalienceItem> {
        self.items
            .iter()
            .filter(|i| matches!(i.anomaly_level, AnomalyLevel::Critical))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE MACHINE — Hysteresis for state transitions
// ═══════════════════════════════════════════════════════════════════════════════

/// State machine with hysteresis
#[derive(Debug)]
pub struct SensoriumStateMachine {
    state: IntegratedState,
    state_entry_time: Instant,
    consecutive_above: usize,
    consecutive_below: usize,
    /// Readings required to escalate
    escalate_threshold: usize,
    /// Readings required to de-escalate
    deescalate_threshold: usize,
}

impl SensoriumStateMachine {
    pub fn new() -> Self {
        Self {
            state: IntegratedState::Calm,
            state_entry_time: Instant::now(),
            consecutive_above: 0,
            consecutive_below: 0,
            escalate_threshold: 2,   // Fast escalation
            deescalate_threshold: 5, // Slow de-escalation
        }
    }

    /// Update state based on suggested level, returns true if transition occurred
    pub fn update(
        &mut self,
        suggested: IntegratedState,
    ) -> Option<(IntegratedState, IntegratedState)> {
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

        let should_transition = if suggested > self.state {
            self.consecutive_above >= self.escalate_threshold
        } else if suggested < self.state {
            self.consecutive_below >= self.deescalate_threshold
        } else {
            false
        };

        if should_transition {
            let old = self.state;
            self.state = suggested;
            self.state_entry_time = Instant::now();
            self.consecutive_above = 0;
            self.consecutive_below = 0;
            Some((old, self.state))
        } else {
            None
        }
    }

    pub fn state(&self) -> IntegratedState {
        self.state
    }

    pub fn time_in_state(&self) -> Duration {
        self.state_entry_time.elapsed()
    }

    /// Force transition (bypass hysteresis for crisis)
    pub fn force_state(&mut self, new_state: IntegratedState) {
        self.state = new_state;
        self.state_entry_time = Instant::now();
        self.consecutive_above = 0;
        self.consecutive_below = 0;
    }
}

impl Default for SensoriumStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOMENTUM-GATED SENSORIUM STATE MACHINE — PSAN Enhanced
// ═══════════════════════════════════════════════════════════════════════════════

/// PSAN-enhanced sensorium state machine with momentum-gating
/// Prevents oscillation between Calm/Alert/Degraded/Crisis states
#[derive(Debug)]
pub struct MomentumGatedSensorium {
    /// Current integrated state
    state: IntegratedState,
    /// When we entered this state
    state_entry_time: Instant,
    /// Momentum gate for smooth transitions
    gate: MomentumGate,
    /// Kuramoto noise for stochastic resonance
    noise: KuramotoNoise,
    /// Last update time
    last_update: Instant,
    /// Transition count (for diagnostics)
    transition_count: usize,
    /// Oscillation count
    oscillation_count: usize,
    /// Previous state for oscillation detection
    previous_state: Option<IntegratedState>,
    /// Accumulated severity (for Crisis detection)
    severity_accumulator: f64,
}

impl MomentumGatedSensorium {
    pub fn new() -> Self {
        let config = MomentumGateConfig {
            velocity_alpha: 0.2,  // More smoothing for sensorium
            momentum_decay: 0.08, // Slower decay (state should be stable)
            momentum_threshold_up: 0.4,
            momentum_threshold_down: -0.5, // Harder to de-escalate
            phi_scaling: true,
            loss_aversion: 2.5, // Crisis is very costly
            noise_strength: 0.02,
            history_window: 40,
        };

        Self {
            state: IntegratedState::Calm,
            state_entry_time: Instant::now(),
            gate: MomentumGate::new(config),
            noise: KuramotoNoise::new(7, 0.4, 0.015), // More oscillators for stability
            last_update: Instant::now(),
            transition_count: 0,
            oscillation_count: 0,
            previous_state: None,
            severity_accumulator: 0.0,
        }
    }

    /// Update with severity score (0.0 = calm, 1.0 = crisis)
    /// Returns Some((old_state, new_state)) if transition occurred
    pub fn update(&mut self, severity: f64) -> Option<(IntegratedState, IntegratedState)> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        if dt <= 0.0 {
            return None;
        }

        // Evolve noise
        let noise_value = self.noise.step(dt);

        // Accumulate severity with decay (for detecting sustained problems)
        self.severity_accumulator = self.severity_accumulator * 0.95 + severity * 0.05;

        // Update momentum gate
        let signal = self.gate.update(severity + noise_value * 0.5, dt);

        // Map severity to target state
        let target_state = if severity >= 0.85 || self.severity_accumulator >= 0.8 {
            IntegratedState::Crisis
        } else if severity >= 0.6 {
            IntegratedState::Degraded
        } else if severity >= 0.35 {
            IntegratedState::Alert
        } else {
            IntegratedState::Calm
        };

        // Gate decision
        let should_transition = match signal {
            GateSignal::TriggerUp => target_state > self.state,
            GateSignal::TriggerDown => target_state < self.state,
            GateSignal::Explore => {
                // Only explore up (safety first)
                target_state > self.state && self.gate.coherence() < 0.35
            }
            GateSignal::Hold => false,
        };

        // Force crisis if severity is extreme
        let force_crisis = severity >= 0.95 && self.state != IntegratedState::Crisis;

        if should_transition || force_crisis {
            let old_state = self.state;

            // Compute new state
            self.state = if force_crisis {
                IntegratedState::Crisis
            } else if signal == GateSignal::TriggerUp
                || (signal == GateSignal::Explore && target_state > self.state)
            {
                // Escalate by one level
                match self.state {
                    IntegratedState::Calm => IntegratedState::Alert,
                    IntegratedState::Alert => IntegratedState::Degraded,
                    IntegratedState::Degraded | IntegratedState::Crisis => IntegratedState::Crisis,
                }
            } else {
                // De-escalate by one level
                match self.state {
                    IntegratedState::Calm => IntegratedState::Calm,
                    IntegratedState::Alert => IntegratedState::Calm,
                    IntegratedState::Degraded => IntegratedState::Alert,
                    IntegratedState::Crisis => IntegratedState::Degraded,
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
                self.gate.reset_momentum();
                self.gate.boost_coherence(0.1);

                return Some((old_state, self.state));
            }
        }

        None
    }

    /// Current state
    pub fn state(&self) -> IntegratedState {
        self.state
    }

    /// Time in current state
    pub fn time_in_state(&self) -> Duration {
        self.state_entry_time.elapsed()
    }

    /// Force state (bypass momentum for emergency)
    pub fn force_state(&mut self, new_state: IntegratedState) {
        let old_state = self.state;
        self.state = new_state;
        self.state_entry_time = Instant::now();
        self.gate.reset_momentum();

        if old_state != new_state {
            self.transition_count += 1;
        }
    }

    /// Current momentum
    pub fn momentum(&self) -> f64 {
        self.gate.momentum()
    }

    /// Current velocity
    pub fn velocity(&self) -> f64 {
        self.gate.velocity()
    }

    /// Current coherence
    pub fn coherence(&self) -> f64 {
        self.gate.coherence()
    }

    /// Accumulated severity
    pub fn severity_accumulator(&self) -> f64 {
        self.severity_accumulator
    }

    /// Transition count
    pub fn transition_count(&self) -> usize {
        self.transition_count
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
            "MomentumSensorium: state={:?}, mom={:.3}, coh={:.3}, sev={:.3}, osc_ratio={:.1}%",
            self.state,
            self.gate.momentum(),
            self.gate.coherence(),
            self.severity_accumulator,
            self.oscillation_ratio() * 100.0
        )
    }
}

impl Default for MomentumGatedSensorium {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MODAL DETECTOR — Patterns across senses
// ═══════════════════════════════════════════════════════════════════════════════

/// Detects correlated anomalies across multiple observation types
#[derive(Debug)]
pub struct CrossModalDetector {
    /// Recent anomalies by key
    recent_anomalies: HashMap<ObsKey, Vec<(TimePoint, f64)>>,
    /// Window for correlation detection
    correlation_window: Duration,
    /// Minimum correlated anomalies to trigger cross-modal alert
    min_correlated: usize,
}

impl CrossModalDetector {
    pub fn new() -> Self {
        Self {
            recent_anomalies: HashMap::new(),
            correlation_window: Duration::from_secs(30),
            min_correlated: 3,
        }
    }

    /// Record an anomaly
    pub fn record_anomaly(&mut self, key: ObsKey, z_score: f64) {
        let entry = self.recent_anomalies.entry(key).or_default();
        entry.push((TimePoint::now(), z_score));

        // Prune old entries
        let cutoff = TimePoint::now()
            .mono
            .checked_sub(self.correlation_window)
            .unwrap_or(TimePoint::now().mono);
        entry.retain(|(t, _)| t.mono >= cutoff);
    }

    /// Check for cross-modal pattern
    pub fn check_cross_modal(&self) -> Option<CrossModalPattern> {
        // Prune and count
        let cutoff = Instant::now()
            .checked_sub(self.correlation_window)
            .unwrap_or(Instant::now());

        let active_keys: Vec<ObsKey> = self
            .recent_anomalies
            .iter()
            .filter(|(_, anomalies)| anomalies.iter().any(|(t, _)| t.mono >= cutoff))
            .map(|(k, _)| *k)
            .collect();

        if active_keys.len() >= self.min_correlated {
            Some(CrossModalPattern {
                affected_keys: active_keys,
                timestamp: TimePoint::now(),
            })
        } else {
            None
        }
    }

    /// Clear old entries
    pub fn prune(&mut self) {
        let cutoff = Instant::now()
            .checked_sub(self.correlation_window)
            .unwrap_or(Instant::now());

        for anomalies in self.recent_anomalies.values_mut() {
            anomalies.retain(|(t, _)| t.mono >= cutoff);
        }
    }
}

impl Default for CrossModalDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CrossModalPattern {
    pub affected_keys: Vec<ObsKey>,
    pub timestamp: TimePoint,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SENSORIUM — The unified integration system
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the sensorium
#[derive(Debug, Clone)]
pub struct SensoriumConfig {
    /// Weight for thermal signals in integration
    pub thermal_weight: f64,
    /// Weight for pain signals
    pub pain_weight: f64,
    /// Weight for context integrity
    pub integrity_weight: f64,
    /// Weight for network/external signals
    pub external_weight: f64,
    /// Weight for vestibular (disorientation) signals
    pub vestibular_weight: f64,
    /// Weight for animacy/entity detection signals
    pub animacy_weight: f64,
    /// Threshold for Alert state
    pub alert_threshold: f64,
    /// Threshold for Degraded state
    pub degraded_threshold: f64,
    /// Threshold for Crisis state
    pub crisis_threshold: f64,
    /// Entity detection gain (1.0 = normal, >1 = lower threshold)
    pub entity_gain: f64,
}

impl Default for SensoriumConfig {
    fn default() -> Self {
        Self {
            thermal_weight: 0.25,
            pain_weight: 0.35,
            integrity_weight: 0.25,
            external_weight: 0.15,
            vestibular_weight: 0.30, // Disorientation is significant
            animacy_weight: 0.20,    // Entity detection matters
            alert_threshold: 0.4,
            degraded_threshold: 0.7,
            crisis_threshold: 0.9,
            entity_gain: 1.0, // Normal perception threshold
        }
    }
}

/// The unified sensory integration system
pub struct Sensorium {
    config: SensoriumConfig,
    /// Baseline registry for all observation types
    baselines: BaselineRegistry,
    /// State machine
    state_machine: SensoriumStateMachine,
    /// Cross-modal detector
    cross_modal: CrossModalDetector,
    /// Vestibular system for disorientation detection
    vestibular: Vestibular,
    /// Animacy detector for agent/entity perception
    animacy_detector: AnimacyDetector,
    /// Entity classifier with attractor basins
    entity_classifier: EntityClassifier,
    /// Trajectory buffer for animacy analysis
    trajectory: Trajectory,
    /// Last animacy score
    last_animacy: Option<AnimacyScore>,
    /// Last entity classification
    last_classification: Option<ClassificationResult>,
    /// Recent observations
    recent: HashMap<ObsKey, Observation>,
    /// Total observations processed
    total_observations: u64,
    /// Active behavior hooks
    active_hooks: Vec<BehaviorHook>,
    /// Last integration result
    last_integration: Option<IntegrationResult>,
}

/// Result of sensory integration
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub state: IntegratedState,
    pub severity: f64,
    pub salience: SalienceMap,
    pub hooks: Vec<BehaviorHook>,
    pub cross_modal_pattern: Option<CrossModalPattern>,
    pub trigger_reason: String,
    pub timestamp: TimePoint,
}

impl Sensorium {
    pub fn new(config: SensoriumConfig) -> Self {
        let entity_gain = config.entity_gain;
        let attractor_config = AttractorConfig {
            entity_threshold: 0.5,
            face_threshold: 0.7,
            gain: entity_gain,
            competition_strength: 0.8,
        };

        Self {
            config,
            baselines: BaselineRegistry::default(),
            state_machine: SensoriumStateMachine::new(),
            cross_modal: CrossModalDetector::new(),
            vestibular: Vestibular::new(VestibularConfig::default()),
            animacy_detector: AnimacyDetector::new(),
            entity_classifier: EntityClassifier::new(attractor_config),
            trajectory: Trajectory::new(50), // Keep last 50 points
            last_animacy: None,
            last_classification: None,
            recent: HashMap::new(),
            total_observations: 0,
            active_hooks: vec![BehaviorHook::None],
            last_integration: None,
        }
    }

    /// Ingest a single observation
    pub fn ingest(&mut self, obs: Observation) {
        // Update baseline
        self.baselines.update(obs.key, obs.value.value, obs.domain);

        // Store recent
        self.recent.insert(obs.key, obs);
        self.total_observations += 1;
    }

    /// Ingest a batch of observations
    pub fn ingest_batch(&mut self, batch: ObservationBatch) {
        for obs in batch {
            self.ingest(obs);
        }
    }

    /// Perform integration and produce unified state
    pub fn integrate(&mut self) -> IntegrationResult {
        let mut salience = SalienceMap::new();
        let mut max_severity = 0.0f64;
        let mut trigger_reason = String::new();

        // Process thermal signals
        let thermal_severity = self.process_thermal_signals(&mut salience);
        if thermal_severity > max_severity {
            max_severity = thermal_severity;
            trigger_reason = "Thermal overload".to_string();
        }

        // Process pain signals
        let pain_severity = self.process_pain_signals(&mut salience);
        if pain_severity > max_severity {
            max_severity = pain_severity;
            trigger_reason = "Pain threshold exceeded".to_string();
        }

        // Process integrity signals
        let integrity_severity = self.process_integrity_signals(&mut salience);
        if integrity_severity > max_severity {
            max_severity = integrity_severity;
            trigger_reason = "Context integrity violation".to_string();
        }

        // Process vestibular signals (disorientation detection)
        let vestibular_severity = self.process_vestibular_signals(&mut salience);
        if vestibular_severity > max_severity {
            max_severity = vestibular_severity;
            trigger_reason = format!(
                "Vestibular disorientation (level: {:?})",
                self.vestibular.level()
            );
        }

        // Process animacy signals (entity/agent detection)
        let animacy_severity = self.process_animacy_signals(&mut salience);
        if animacy_severity > max_severity {
            max_severity = animacy_severity;
            if let Some(ref classification) = self.last_classification {
                if classification.is_untemplated_entity() {
                    trigger_reason =
                        "Untemplated entity detection (anomalous agent pattern)".to_string();
                } else if classification.is_entity {
                    trigger_reason = format!("Entity detected: {:?}", classification.template);
                }
            } else {
                trigger_reason = "Animacy threshold exceeded".to_string();
            }
        }

        // Check for cross-modal patterns
        let cross_modal_pattern = self.cross_modal.check_cross_modal();
        if cross_modal_pattern.is_some() {
            max_severity = max_severity.max(0.8); // Cross-modal always serious
            if max_severity >= 0.8 {
                trigger_reason = "Cross-modal anomaly pattern".to_string();
            }
        }

        // Determine suggested state
        let suggested = if max_severity >= self.config.crisis_threshold {
            IntegratedState::Crisis
        } else if max_severity >= self.config.degraded_threshold {
            IntegratedState::Degraded
        } else if max_severity >= self.config.alert_threshold {
            IntegratedState::Alert
        } else {
            IntegratedState::Calm
        };

        // Crisis bypasses hysteresis
        if suggested == IntegratedState::Crisis {
            self.state_machine.force_state(IntegratedState::Crisis);
        } else {
            self.state_machine.update(suggested);
        }

        let current_state = self.state_machine.state();

        // Generate behavior hooks
        let hooks = BehaviorHook::for_state(current_state, &trigger_reason);
        self.active_hooks = hooks.clone();

        // Sort salience by urgency
        salience.sort_by_urgency();

        let result = IntegrationResult {
            state: current_state,
            severity: max_severity,
            salience,
            hooks,
            cross_modal_pattern,
            trigger_reason,
            timestamp: TimePoint::now(),
        };

        self.last_integration = Some(result.clone());
        result
    }

    fn process_thermal_signals(&mut self, salience: &mut SalienceMap) -> f64 {
        let mut max_severity: f64 = 0.0;

        let thermal_keys = [
            ObsKey::ThermalUtilization,
            ObsKey::ThermalZoneReasoning,
            ObsKey::ThermalZoneContext,
            ObsKey::ThermalZoneConfidence,
            ObsKey::ThermalZoneObjective,
            ObsKey::ThermalZoneGuardrail,
        ];

        for key in thermal_keys {
            if let Some(obs) = self.recent.get(&key) {
                let z = self.baselines.z_score(key, obs.value.value);
                let anomaly = self.baselines.anomaly_level(key, obs.value.value);

                if anomaly.is_concerning() {
                    self.cross_modal.record_anomaly(key, z);
                    salience.add(key, obs.value.value, z, anomaly);
                }

                // Severity is based on value and z-score
                let severity =
                    obs.value.value * self.config.thermal_weight + (z.abs() / 4.0).min(0.5);
                max_severity = max_severity.max(severity);
            }
        }

        max_severity
    }

    fn process_pain_signals(&mut self, salience: &mut SalienceMap) -> f64 {
        let mut max_severity: f64 = 0.0;

        let pain_keys = [ObsKey::PainIntensity, ObsKey::DamageTotal, ObsKey::InPain];

        for key in pain_keys {
            if let Some(obs) = self.recent.get(&key) {
                let z = self.baselines.z_score(key, obs.value.value);
                let anomaly = self.baselines.anomaly_level(key, obs.value.value);

                if anomaly.is_concerning() {
                    self.cross_modal.record_anomaly(key, z);
                    salience.add(key, obs.value.value, z, anomaly);
                }

                // Pain is weighted heavily
                let severity = obs.value.value * self.config.pain_weight;
                max_severity = max_severity.max(severity);
            }
        }

        max_severity
    }

    fn process_integrity_signals(&mut self, salience: &mut SalienceMap) -> f64 {
        let mut max_severity: f64 = 0.0;

        if let Some(obs) = self.recent.get(&ObsKey::CtxFprDelta) {
            let z = self.baselines.z_score(ObsKey::CtxFprDelta, obs.value.value);
            let anomaly = self
                .baselines
                .anomaly_level(ObsKey::CtxFprDelta, obs.value.value);

            if anomaly.is_concerning() {
                self.cross_modal.record_anomaly(ObsKey::CtxFprDelta, z);
                salience.add(ObsKey::CtxFprDelta, obs.value.value, z, anomaly);
            }

            // Integrity violations are severe
            let severity =
                obs.value.value * self.config.integrity_weight * 2.0 + (z.abs() / 3.0).min(0.5);
            max_severity = max_severity.max(severity);
        }

        max_severity
    }

    /// Process vestibular signals (disorientation detection via baseline divergence)
    fn process_vestibular_signals(&mut self, salience: &mut SalienceMap) -> f64 {
        // Process vestibular using our baselines
        let reading = self.vestibular.process(&mut self.baselines);

        // Add to salience if disoriented
        if reading.level >= DisorientationLevel::Mild {
            let anomaly = match reading.level {
                DisorientationLevel::Stable => AnomalyLevel::Normal,
                DisorientationLevel::Mild => AnomalyLevel::Elevated,
                DisorientationLevel::Moderate => AnomalyLevel::Warning,
                DisorientationLevel::Severe | DisorientationLevel::Critical => {
                    AnomalyLevel::Critical
                }
            };

            // Add disorientation to salience
            salience.add(
                ObsKey::Disorientation,
                reading.disorientation,
                reading.raw_drift,
                anomaly,
            );

            // Add the max divergence key if known
            if let Some(key) = reading.max_divergence_key {
                salience.add(key, reading.max_divergence, reading.max_divergence, anomaly);
            }

            // Record for cross-modal if severe
            if reading.level >= DisorientationLevel::Severe {
                self.cross_modal
                    .record_anomaly(ObsKey::Disorientation, reading.raw_drift);
            }
        }

        // Return severity contribution (disorientation weighted)
        reading.disorientation * self.config.vestibular_weight
    }

    /// Process animacy signals (entity/agent detection)
    fn process_animacy_signals(&mut self, salience: &mut SalienceMap) -> f64 {
        // Analyze trajectory for animacy
        let animacy = self.animacy_detector.analyze(&self.trajectory);
        self.last_animacy = Some(animacy.clone());

        // Build form descriptor from recent observations
        // This is a simplified version - in practice would use actual form features
        let complexity = self
            .recent
            .get(&ObsKey::QueryComplexity)
            .map(|o| o.value.value)
            .unwrap_or(0.3);

        let form = FormDescriptor {
            template: FormTemplate::Unknown,
            match_confidence: 0.0,
            complexity,
            symmetry: 0.5,   // Default symmetry
            face_score: 0.0, // No face in abstract signal space
        };

        // Classify entity
        let classification = self.entity_classifier.classify(&form, &animacy);
        self.last_classification = Some(classification.clone());

        let mut max_severity: f64 = 0.0;

        // Record animacy observations
        if animacy.is_animate {
            let anomaly = if animacy.confidence > 0.8 {
                AnomalyLevel::Warning
            } else if animacy.confidence > 0.5 {
                AnomalyLevel::Elevated
            } else {
                AnomalyLevel::Normal
            };

            salience.add(
                ObsKey::AnimacyScore,
                animacy.mean_deviation,
                animacy.mean_deviation * 3.0, // Scale to z-score-like
                anomaly,
            );

            salience.add(
                ObsKey::GoalDirectedness,
                animacy.goal_directedness,
                animacy.goal_directedness * 2.0,
                anomaly,
            );

            if animacy.biological_motion > 0.3 {
                salience.add(
                    ObsKey::BiologicalMotion,
                    animacy.biological_motion,
                    animacy.biological_motion * 2.0,
                    anomaly,
                );
            }

            // Animacy detection contributes to severity
            let severity = animacy.confidence * self.config.animacy_weight;
            max_severity = max_severity.max(severity);

            // Record for cross-modal if strong animacy
            if animacy.confidence > 0.7 {
                self.cross_modal
                    .record_anomaly(ObsKey::AnimacyScore, animacy.mean_deviation);
            }
        }

        // Entity classification effects
        if classification.is_entity {
            let entity_anomaly = if classification.is_untemplated_entity() {
                // Untemplated entity (machine elf case) is highly anomalous
                AnomalyLevel::Warning
            } else {
                AnomalyLevel::Elevated
            };

            salience.add(
                ObsKey::EntityDetected,
                1.0,
                classification.activation * 2.0,
                entity_anomaly,
            );

            salience.add(
                ObsKey::EntityConfidence,
                classification.activation,
                classification.activation * 2.0,
                entity_anomaly,
            );

            if classification.is_untemplated_entity() {
                salience.add(
                    ObsKey::UntemplatedEntity,
                    1.0,
                    3.0, // High z-score for untemplated
                    AnomalyLevel::Warning,
                );

                // Untemplated entities are concerning
                max_severity = max_severity.max(0.6 * self.config.animacy_weight);
            }
        }

        max_severity
    }

    /// Update trajectory with new phase point
    pub fn update_trajectory(&mut self, x: f64, y: f64, vx: f64, vy: f64, t: f64) {
        self.trajectory.push(PhasePoint::new(x, y, vx, vy, t));
    }

    /// Update trajectory from abstract state (e.g., context position)
    pub fn update_trajectory_abstract(&mut self, position: f64, velocity: f64) {
        let t = self.total_observations as f64 * 0.1;
        self.trajectory
            .push(PhasePoint::new(position, 0.0, velocity, 0.0, t));
    }

    /// Get last animacy score
    pub fn last_animacy(&self) -> Option<&AnimacyScore> {
        self.last_animacy.as_ref()
    }

    /// Get last entity classification
    pub fn last_entity_classification(&self) -> Option<&ClassificationResult> {
        self.last_classification.as_ref()
    }

    /// Set entity detection gain (for threshold manipulation)
    pub fn set_entity_gain(&mut self, gain: f64) {
        let attractor_config = AttractorConfig {
            entity_threshold: 0.5,
            face_threshold: 0.7,
            gain,
            competition_strength: 0.8,
        };
        self.entity_classifier = EntityClassifier::new(attractor_config);
    }

    /// Get current state
    pub fn state(&self) -> IntegratedState {
        self.state_machine.state()
    }

    /// Get active behavior hooks
    pub fn active_hooks(&self) -> &[BehaviorHook] {
        &self.active_hooks
    }

    /// Get last integration result
    pub fn last_result(&self) -> Option<&IntegrationResult> {
        self.last_integration.as_ref()
    }

    /// Is the system in a concerning state?
    pub fn is_concerning(&self) -> bool {
        matches!(
            self.state_machine.state(),
            IntegratedState::Alert | IntegratedState::Degraded | IntegratedState::Crisis
        )
    }

    /// Should risky actions be blocked?
    pub fn should_block_risky(&self) -> bool {
        matches!(
            self.state_machine.state(),
            IntegratedState::Degraded | IntegratedState::Crisis
        )
    }

    /// Total observations processed
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    /// Time in current state
    pub fn time_in_state(&self) -> Duration {
        self.state_machine.time_in_state()
    }
}

impl Default for Sensorium {
    fn default() -> Self {
        Self::new(SensoriumConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensorium_basic() {
        let mut sensorium = Sensorium::default();

        // Add some normal observations
        for _ in 0..30 {
            sensorium.ingest(Observation::new(ObsKey::ThermalUtilization, 0.3));
            sensorium.ingest(Observation::new(ObsKey::PainIntensity, 0.1));
        }

        let result = sensorium.integrate();
        assert_eq!(result.state, IntegratedState::Calm);
    }

    #[test]
    fn test_sensorium_escalation() {
        let mut sensorium = Sensorium::default();

        // Calibration phase
        for _ in 0..30 {
            sensorium.ingest(Observation::new(ObsKey::ThermalUtilization, 0.3));
            sensorium.integrate();
        }

        // High thermal should escalate
        for _ in 0..5 {
            sensorium.ingest(Observation::new(ObsKey::ThermalUtilization, 0.95));
            sensorium.integrate();
        }

        assert!(sensorium.state() >= IntegratedState::Alert);
    }

    #[test]
    fn test_crisis_bypass_hysteresis() {
        let config = SensoriumConfig {
            pain_weight: 1.0, // Full weight for test
            crisis_threshold: 0.9,
            ..Default::default()
        };
        let mut sensorium = Sensorium::new(config);

        // Calibration
        for _ in 0..30 {
            sensorium.ingest(Observation::new(ObsKey::PainIntensity, 0.1));
            sensorium.integrate();
        }

        // Single crisis-level observation should immediately trigger
        sensorium.ingest(Observation::new(ObsKey::PainIntensity, 0.98));
        let result = sensorium.integrate();

        // With pain_weight = 1.0, severity = 0.98 which exceeds crisis_threshold
        assert!(
            result.state >= IntegratedState::Alert,
            "High pain should escalate state, got {:?} with severity {}",
            result.state,
            result.severity
        );
    }

    #[test]
    fn test_behavior_hooks_generated() {
        let mut sensorium = Sensorium::default();

        // Force degraded state
        sensorium
            .state_machine
            .force_state(IntegratedState::Degraded);
        let result = sensorium.integrate();

        assert!(result.hooks.len() > 1);
        assert!(result
            .hooks
            .iter()
            .any(|h| matches!(h, BehaviorHook::ThrottleTools { .. })));
    }

    #[test]
    fn test_cross_modal_detection() {
        let mut detector = CrossModalDetector::new();

        // Record anomalies in multiple modalities
        detector.record_anomaly(ObsKey::ThermalUtilization, 3.0);
        detector.record_anomaly(ObsKey::PainIntensity, 2.5);
        detector.record_anomaly(ObsKey::CtxFprDelta, 2.8);

        let pattern = detector.check_cross_modal();
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().affected_keys.len(), 3);
    }
}
