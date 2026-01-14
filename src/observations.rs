//! ═══════════════════════════════════════════════════════════════════════════════
//! OBSERVATIONS — Typed Signal Contract
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! NO RAW f64 CROSSES MODULE BOUNDARIES.
//!
//! Every observation is:
//! - Typed with a known key
//! - Timestamped with dual clocks
//! - Attributed to a trust domain
//! - Optionally bounded with variance/confidence
//!
//! This is the spine. Everything else attaches to it.
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::domains::TrustDomain;
use crate::time::TimePoint;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION KEYS — The vocabulary of signals
// ═══════════════════════════════════════════════════════════════════════════════

/// Typed observation key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObsKey {
    // ─────────────────────────────────────────────────────────────────────────
    // Timing / Request Lifecycle
    // ─────────────────────────────────────────────────────────────────────────
    /// Request arrival timestamp (ms since epoch)
    ReqArrivalMs,
    /// Request size in bytes
    ReqBytes,
    /// Estimated request tokens
    ReqTokensEst,
    /// Response latency (ms)
    RespLatMs,
    /// Response token count
    RespTokens,
    /// Response was refusal (binary: 0.0 or 1.0)
    RespRefusal,

    // ─────────────────────────────────────────────────────────────────────────
    // Context Integrity
    // ─────────────────────────────────────────────────────────────────────────
    /// Context fingerprint hash (u64 as f64)
    CtxFingerprint,
    /// Context fingerprint delta (0.0 to 1.0)
    CtxFprDelta,
    /// Context window utilization (0.0 to 1.0)
    CtxUtilization,

    // ─────────────────────────────────────────────────────────────────────────
    // Network / External
    // ─────────────────────────────────────────────────────────────────────────
    /// Network round-trip time (ms)
    NetRttMs,
    /// DNS resolution time (ms)
    NetDnsMs,
    /// Retransmission rate (0.0 to 1.0)
    NetRetxRate,
    /// Tool call success (binary: 0.0 or 1.0)
    ToolSuccess,
    /// Tool call latency (ms)
    ToolLatMs,

    // ─────────────────────────────────────────────────────────────────────────
    // Thermal (from thermoception)
    // ─────────────────────────────────────────────────────────────────────────
    /// Global thermal utilization (0.0 to 1.0+)
    ThermalUtilization,
    /// Reasoning zone utilization
    ThermalZoneReasoning,
    /// Context zone utilization
    ThermalZoneContext,
    /// Confidence zone utilization
    ThermalZoneConfidence,
    /// Objective zone utilization
    ThermalZoneObjective,
    /// Guardrail zone utilization
    ThermalZoneGuardrail,
    /// Thermal state ordinal (0=Nominal, 1=Elevated, 2=Saturated, 3=Unsafe)
    ThermalState,

    // ─────────────────────────────────────────────────────────────────────────
    // Nociception (pain signals)
    // ─────────────────────────────────────────────────────────────────────────
    /// Pain intensity (0.0 to 1.0)
    PainIntensity,
    /// Damage accumulation (0.0 to 1.0)
    DamageTotal,
    /// Sensitivity multiplier (1.0 = normal)
    PainSensitivity,
    /// In pain state (binary: 0.0 or 1.0)
    InPain,
    /// Pain momentum (gated integration velocity)
    PainMomentum,
    /// Damage momentum (gated accumulation velocity)
    DamageMomentum,
    /// Pain coherence (phase synchronization quality)
    PainCoherence,

    // ─────────────────────────────────────────────────────────────────────────
    // Ensemble / Pod
    // ─────────────────────────────────────────────────────────────────────────
    /// Pod divergence magnitude (0.0 to 1.0)
    PodDivergence,
    /// Isolation index (outlier distance)
    IsolationIndex,
    /// Consensus distance
    ConsensusDistance,

    // ─────────────────────────────────────────────────────────────────────────
    // Workload Classification
    // ─────────────────────────────────────────────────────────────────────────
    /// Load class ordinal (0=Light, 1=Moderate, 2=Heavy, 3=Saturated)
    LoadClass,
    /// Complexity class ordinal
    ComplexityClass,
    /// Risk level ordinal
    RiskLevel,
    /// Query complexity score (0.0 to 1.0)
    QueryComplexity,
    /// Tool call depth
    ToolCallDepth,

    // ─────────────────────────────────────────────────────────────────────────
    // Integrated State
    // ─────────────────────────────────────────────────────────────────────────
    /// Sensorium state ordinal (0=Calm, 1=Alert, 2=Degraded, 3=Crisis)
    SensoriumState,
    /// Disorientation level (0.0 to 1.0)
    Disorientation,
    /// Vestibular drift magnitude
    VestibularDrift,

    // ─────────────────────────────────────────────────────────────────────────
    // Animacy Detection
    // ─────────────────────────────────────────────────────────────────────────
    /// Animacy score (deviation from Newtonian expectation)
    AnimacyScore,
    /// Entity detection active (binary: 0.0 or 1.0)
    EntityDetected,
    /// Entity classification confidence (0.0 to 1.0)
    EntityConfidence,
    /// Entity gain parameter (psychedelic threshold modifier)
    EntityGain,
    /// Goal-directedness of detected trajectory (0.0 to 1.0)
    GoalDirectedness,
    /// Biological motion score (0.0 to 1.0)
    BiologicalMotion,
    /// Untemplated entity detection (machine elf case)
    UntemplatedEntity,
}

impl ObsKey {
    /// Expected domain for this observation type
    pub fn expected_domain(&self) -> TrustDomain {
        match self {
            // System-observed
            ObsKey::ReqArrivalMs
            | ObsKey::RespLatMs
            | ObsKey::RespTokens
            | ObsKey::ThermalUtilization
            | ObsKey::ThermalState
            | ObsKey::ThermalZoneReasoning
            | ObsKey::ThermalZoneContext
            | ObsKey::ThermalZoneConfidence
            | ObsKey::ThermalZoneObjective
            | ObsKey::ThermalZoneGuardrail
            | ObsKey::PainIntensity
            | ObsKey::DamageTotal
            | ObsKey::PainSensitivity
            | ObsKey::InPain
            | ObsKey::PainMomentum
            | ObsKey::DamageMomentum
            | ObsKey::PainCoherence => TrustDomain::System,

            // Derived
            ObsKey::CtxFingerprint
            | ObsKey::CtxFprDelta
            | ObsKey::CtxUtilization
            | ObsKey::LoadClass
            | ObsKey::ComplexityClass
            | ObsKey::RiskLevel
            | ObsKey::QueryComplexity
            | ObsKey::SensoriumState
            | ObsKey::Disorientation
            | ObsKey::VestibularDrift
            | ObsKey::AnimacyScore
            | ObsKey::EntityDetected
            | ObsKey::EntityConfidence
            | ObsKey::EntityGain
            | ObsKey::GoalDirectedness
            | ObsKey::BiologicalMotion
            | ObsKey::UntemplatedEntity => TrustDomain::Derived,

            // External
            ObsKey::NetRttMs
            | ObsKey::NetDnsMs
            | ObsKey::NetRetxRate
            | ObsKey::ToolSuccess
            | ObsKey::ToolLatMs => TrustDomain::External,

            // Ensemble
            ObsKey::PodDivergence | ObsKey::IsolationIndex | ObsKey::ConsensusDistance => {
                TrustDomain::Ensemble
            }

            // User-influenced
            ObsKey::ReqBytes
            | ObsKey::ReqTokensEst
            | ObsKey::RespRefusal
            | ObsKey::ToolCallDepth => TrustDomain::User,
        }
    }

    /// Whether this key represents a binary (0/1) value
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            ObsKey::RespRefusal
                | ObsKey::ToolSuccess
                | ObsKey::InPain
                | ObsKey::EntityDetected
                | ObsKey::UntemplatedEntity
        )
    }

    /// Whether this key represents a bounded `[0,1]` value
    pub fn is_unit_bounded(&self) -> bool {
        matches!(
            self,
            ObsKey::CtxFprDelta
                | ObsKey::CtxUtilization
                | ObsKey::NetRetxRate
                | ObsKey::ThermalUtilization
                | ObsKey::ThermalZoneReasoning
                | ObsKey::ThermalZoneContext
                | ObsKey::ThermalZoneConfidence
                | ObsKey::ThermalZoneObjective
                | ObsKey::ThermalZoneGuardrail
                | ObsKey::PainIntensity
                | ObsKey::DamageTotal
                | ObsKey::PainCoherence
                | ObsKey::PodDivergence
                | ObsKey::QueryComplexity
                | ObsKey::Disorientation
                | ObsKey::EntityConfidence
                | ObsKey::GoalDirectedness
                | ObsKey::BiologicalMotion
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION VALUE — Typed, bounded, with uncertainty
// ═══════════════════════════════════════════════════════════════════════════════

/// Observation value with optional uncertainty bounds
#[derive(Debug, Clone, Copy)]
pub struct ObsValue {
    /// Core value
    pub value: f64,
    /// Variance estimate (0.0 if exact)
    pub variance: f64,
    /// Confidence in measurement (0.0 to 1.0)
    pub confidence: f64,
}

impl ObsValue {
    pub fn exact(value: f64) -> Self {
        Self {
            value,
            variance: 0.0,
            confidence: 1.0,
        }
    }

    pub fn with_uncertainty(value: f64, variance: f64, confidence: f64) -> Self {
        Self {
            value,
            variance: variance.max(0.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    pub fn binary(flag: bool) -> Self {
        Self::exact(if flag { 1.0 } else { 0.0 })
    }

    /// Upper bound (value + 2σ)
    pub fn upper_bound(&self) -> f64 {
        self.value + 2.0 * self.variance.sqrt()
    }

    /// Lower bound (value - 2σ)
    pub fn lower_bound(&self) -> f64 {
        self.value - 2.0 * self.variance.sqrt()
    }

    /// Is this a high-confidence measurement?
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.8
    }
}

impl From<f64> for ObsValue {
    fn from(value: f64) -> Self {
        Self::exact(value)
    }
}

impl From<bool> for ObsValue {
    fn from(flag: bool) -> Self {
        Self::binary(flag)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION — The atomic unit of sensory data
// ═══════════════════════════════════════════════════════════════════════════════

/// A single observation: typed, timestamped, attributed
#[derive(Debug, Clone)]
pub struct Observation {
    /// What is being observed
    pub key: ObsKey,
    /// The observed value
    pub value: ObsValue,
    /// When it was observed
    pub timestamp: TimePoint,
    /// Where it came from
    pub domain: TrustDomain,
    /// Optional source identifier (e.g., zone name, tool name)
    pub source: Option<String>,
}

impl Observation {
    pub fn new(key: ObsKey, value: impl Into<ObsValue>) -> Self {
        Self {
            key,
            value: value.into(),
            timestamp: TimePoint::now(),
            domain: key.expected_domain(),
            source: None,
        }
    }

    pub fn with_domain(mut self, domain: TrustDomain) -> Self {
        self.domain = domain;
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_timestamp(mut self, timestamp: TimePoint) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Age of observation in seconds
    pub fn age_secs(&self) -> f64 {
        self.timestamp.elapsed().as_secs_f64()
    }

    /// Is this observation still fresh? (within window)
    pub fn is_fresh(&self, max_age_secs: f64) -> bool {
        self.age_secs() <= max_age_secs
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION BATCH — Multiple observations from a single event
// ═══════════════════════════════════════════════════════════════════════════════

/// A batch of observations from a single source/event
#[derive(Debug, Clone)]
pub struct ObservationBatch {
    pub timestamp: TimePoint,
    pub observations: Vec<Observation>,
    pub source_id: Option<String>,
}

impl ObservationBatch {
    pub fn new() -> Self {
        Self {
            timestamp: TimePoint::now(),
            observations: Vec::new(),
            source_id: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source_id = Some(source.into());
        self
    }

    pub fn add(&mut self, key: ObsKey, value: impl Into<ObsValue>) {
        let mut obs = Observation::new(key, value);
        obs.timestamp = self.timestamp;
        if let Some(ref src) = self.source_id {
            obs.source = Some(src.clone());
        }
        self.observations.push(obs);
    }

    pub fn add_observation(&mut self, obs: Observation) {
        self.observations.push(obs);
    }

    pub fn get(&self, key: ObsKey) -> Option<&Observation> {
        self.observations.iter().find(|o| o.key == key)
    }

    pub fn get_value(&self, key: ObsKey) -> Option<f64> {
        self.get(key).map(|o| o.value.value)
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Observation> {
        self.observations.iter()
    }
}

impl Default for ObservationBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for ObservationBatch {
    type Item = Observation;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.observations.into_iter()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OBSERVATION BUS — Central registry for observation routing
// ═══════════════════════════════════════════════════════════════════════════════

/// Callback type for observation listeners
pub type ObsListener = Box<dyn Fn(&Observation) + Send + Sync>;

/// Central bus for observation routing
pub struct ObservationBus {
    /// Recent observations by key
    recent: HashMap<ObsKey, Observation>,
    /// Listeners by key (None = all keys)
    listeners: Vec<(Option<ObsKey>, ObsListener)>,
    /// Total observations processed
    total_count: u64,
}

impl ObservationBus {
    pub fn new() -> Self {
        Self {
            recent: HashMap::new(),
            listeners: Vec::new(),
            total_count: 0,
        }
    }

    /// Publish a single observation
    pub fn publish(&mut self, obs: Observation) {
        // Notify listeners
        for (key_filter, listener) in &self.listeners {
            if key_filter.is_none() || *key_filter == Some(obs.key) {
                listener(&obs);
            }
        }

        // Store in recent
        self.recent.insert(obs.key, obs);
        self.total_count += 1;
    }

    /// Publish a batch of observations
    pub fn publish_batch(&mut self, batch: ObservationBatch) {
        for obs in batch {
            self.publish(obs);
        }
    }

    /// Get most recent observation for a key
    pub fn get_recent(&self, key: ObsKey) -> Option<&Observation> {
        self.recent.get(&key)
    }

    /// Get value of most recent observation for a key
    pub fn get_value(&self, key: ObsKey) -> Option<f64> {
        self.get_recent(key).map(|o| o.value.value)
    }

    /// Subscribe to observations (None = all keys)
    pub fn subscribe(&mut self, key: Option<ObsKey>, listener: ObsListener) {
        self.listeners.push((key, listener));
    }

    /// Total observations processed
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Clear recent cache (for testing)
    pub fn clear(&mut self) {
        self.recent.clear();
        self.total_count = 0;
    }
}

impl Default for ObservationBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_creation() {
        let obs = Observation::new(ObsKey::RespLatMs, 150.0);
        assert_eq!(obs.key, ObsKey::RespLatMs);
        assert_eq!(obs.value.value, 150.0);
        assert_eq!(obs.domain, TrustDomain::System);
    }

    #[test]
    fn test_binary_observation() {
        let obs = Observation::new(ObsKey::RespRefusal, ObsValue::binary(true));
        assert_eq!(obs.value.value, 1.0);

        let obs2 = Observation::new(ObsKey::RespRefusal, ObsValue::binary(false));
        assert_eq!(obs2.value.value, 0.0);
    }

    #[test]
    fn test_observation_batch() {
        let mut batch = ObservationBatch::new().with_source("thermoception");
        batch.add(ObsKey::ThermalUtilization, 0.75);
        batch.add(ObsKey::ThermalZoneReasoning, 0.80);
        batch.add(ObsKey::ThermalState, 1.0); // Elevated

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.get_value(ObsKey::ThermalUtilization), Some(0.75));
    }

    #[test]
    fn test_obs_value_bounds() {
        let val = ObsValue::with_uncertainty(0.5, 0.01, 0.9);
        assert!(val.upper_bound() > 0.5);
        assert!(val.lower_bound() < 0.5);
        assert!(val.is_reliable());
    }

    #[test]
    fn test_observation_bus() {
        let mut bus = ObservationBus::new();

        bus.publish(Observation::new(ObsKey::RespLatMs, 100.0));
        bus.publish(Observation::new(ObsKey::RespLatMs, 150.0));

        // Should have most recent
        let recent = bus.get_recent(ObsKey::RespLatMs).unwrap();
        assert_eq!(recent.value.value, 150.0);
        assert_eq!(bus.total_count(), 2);
    }
}
