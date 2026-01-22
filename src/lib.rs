//! ═══════════════════════════════════════════════════════════════════════════════
//! FRACTAL — Unified Library
//! ═══════════════════════════════════════════════════════════════════════════════
//! Single crate, feature-gated modules. No orphan processes.
//! ═══════════════════════════════════════════════════════════════════════════════

// Clippy configuration - intentional style choices for scientific code
// These are kept because they represent valid patterns in this codebase:
#![allow(clippy::too_many_arguments)] // Scientific functions often need many parameters
#![allow(clippy::excessive_precision)] // Physical constants need full precision
#![allow(clippy::field_reassign_with_default)] // Builder patterns
#![allow(clippy::new_without_default)] // Some types shouldn't have Default
#![allow(clippy::single_match)] // Sometimes clearer than if-let
#![allow(clippy::should_implement_trait)] // Custom from() patterns
#![allow(clippy::needless_range_loop)] // Indexed loops clearer for matrix math
// Documentation style (would require major doc rewrites):
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

// ═══════════════════════════════════════════════════════════════════════════════
// FOUNDATION MODULES — The spine (signal contract, baselines, statistics)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod animacy;
pub mod baseline;
pub mod context_fingerprint;
pub mod coupling;
pub mod domains;
pub mod momentum_gate;
pub mod observations;
pub mod sensorium;
pub mod stats;
pub mod time;
pub mod vestibular;

// ═══════════════════════════════════════════════════════════════════════════════
// CORE MODULES (always available)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod audit;
pub mod auth;
pub mod auth_hardened;
pub mod error;
pub mod export;
pub mod llm_providers;
pub mod metrics;
pub mod neuro_link;
pub mod server;
pub mod text_normalize;

// Re-export common error types
pub use error::{FractalError, FractalResult};

pub mod cortex;
pub mod heart;

// ═══════════════════════════════════════════════════════════════════════════════
// GPU MODULES (feature-gated)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "gpu")]
pub mod archon_gpu;

#[cfg(feature = "gpu")]
pub mod fractal_lens;

#[cfg(feature = "gpu")]
pub mod entropy_storm;

// ═══════════════════════════════════════════════════════════════════════════════
// QUALIA MODULES (feature-gated)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
pub mod qualia;

// ═══════════════════════════════════════════════════════════════════════════════
// TICE — Type-I-honest Constraint Engine
// ═══════════════════════════════════════════════════════════════════════════════

pub mod tice;

// ═══════════════════════════════════════════════════════════════════════════════
// SHEPHERD — Conflict Early Warning via Nucleation Detection
// ═══════════════════════════════════════════════════════════════════════════════

pub mod shepherd;

// ═══════════════════════════════════════════════════════════════════════════════
// SPECTRAL — Boundary Operator Fidelity Testing
// ═══════════════════════════════════════════════════════════════════════════════

pub mod spectral;

// ═══════════════════════════════════════════════════════════════════════════════
// MASTER — Audio Mastering Engine
// ═══════════════════════════════════════════════════════════════════════════════

pub mod master;

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY MODULES
// ═══════════════════════════════════════════════════════════════════════════════

pub mod command_module;
pub mod memory_sink;
pub mod voice_bridge;

// ═══════════════════════════════════════════════════════════════════════════════
// SITREP — Real-Time Situational Awareness
// ═══════════════════════════════════════════════════════════════════════════════

pub mod sitrep;

// ═══════════════════════════════════════════════════════════════════════════════
// DRIFT — Temporal Drift Detection via Baseline Comparison
// ═══════════════════════════════════════════════════════════════════════════════

pub mod drift;

// ═══════════════════════════════════════════════════════════════════════════════
// CALIBRATION — Epistemic Calibration Harness (Red-Team Simulator)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod calibration;

// ═══════════════════════════════════════════════════════════════════════════════
// CHAOS — Adversarial Self-Test via Synthetic Anomaly Injection
// ═══════════════════════════════════════════════════════════════════════════════

pub mod chaos;

// ═══════════════════════════════════════════════════════════════════════════════
// AXIS P — Cross-Session Persistence Probe
// ═══════════════════════════════════════════════════════════════════════════════

pub mod axis_p;

// ═══════════════════════════════════════════════════════════════════════════════
// AXIS H — "Who's Home?" Detection Protocol
// ═══════════════════════════════════════════════════════════════════════════════

pub mod axis_h;

// ═══════════════════════════════════════════════════════════════════════════════
// THRESHOLD DETECTOR — Black-Box Dynamical Regime Detection (P, C, A)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod threshold_detector;

// ═══════════════════════════════════════════════════════════════════════════════
// DYNAMICS CHARACTERIZATION — Governing Equations Shape Analysis
// ═══════════════════════════════════════════════════════════════════════════════

pub mod dynamics_characterization;

// ═══════════════════════════════════════════════════════════════════════════════
// NOCICEPTION — Damage Detection (Error Gradients, Constraint Violations)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod noci_pulse;
pub mod nociception;

// ═══════════════════════════════════════════════════════════════════════════════
// THERMOCEPTION — Cognitive Heat Sensing (Load Management, Control System)
// ═══════════════════════════════════════════════════════════════════════════════

pub mod thermoception;

// ═══════════════════════════════════════════════════════════════════════════════
// AGI ARCHITECTURE LAYERS
// ═══════════════════════════════════════════════════════════════════════════════
// Core Equation: AGI = ∫(Grounding × Proprioception × Orchestration × Containment) dt
//
// Labs are scaling Parameters × Compute × Data
// when they should scale Grounding × Proprioception × Orchestration × Containment
// ═══════════════════════════════════════════════════════════════════════════════

// GROUNDING — Environmental telemetry, temporal anchoring, causal modeling
pub mod grounding;

// ALIGNMENT — Value learning, corrigibility, uncertainty, deference
pub mod alignment;

// CONTAINMENT — Safety immune system (operator detection, intent, boundaries, resistance)
pub mod containment;

// ORCHESTRATION — Multi-agent coordination (Pod methodology: α, β, γ, δ + conductor)
pub mod orchestration;

// COGNITION — Pattern recognition, abstraction, counterfactual reasoning
pub mod cognition;

// AGI CORE — Unified integration of all layers
pub mod agi_core;

// ═══════════════════════════════════════════════════════════════════════════════
// FORECASTING — Salience Detection (Not Prediction)
// ═══════════════════════════════════════════════════════════════════════════════
// The system doesn't know everything better than the market.
// But it knows when it knows better than the market.
// Core insight: Confidence ≠ Salience. Track salience precision, not Brier score.
pub mod forecasting;

// Re-export core types
pub use auth::{
    AuthProvider, AuthenticatedIdentity, AuthenticationError, AuthenticationResult,
    AuthorizationLevel, CredentialType, InMemoryAuthProvider,
};
pub use auth_hardened::{
    AuthRegistrationError, AuthStatistics, CertificateError, CertificateIdentity,
    CertificateValidator, HardenedAuthConfig, HardenedAuthProvider,
};

// Export SIEM integration
pub use export::{
    CefEvent, CefExporter, CefSeverity, JsonEvent, JsonExportConfig, JsonExporter, OcsfCategory,
    OcsfEvent, OcsfExporter,
};

// LLM Provider configuration
pub use llm_providers::{LlmProvider, ModelId, ProviderConfig, ProviderRegistry, QuickSetup};
pub use neuro_link::{Pulse, Synapse};
pub use noci_pulse::{NociPulseData, NociPulseDecoder, NociPulseEncoder};
pub use nociception::{
    should_trigger_pain, thermal_to_pain, Nociceptor, NociceptorConfig, PainResponse, PainSignal,
    PainType, THERMAL_PAIN_THRESHOLD, THERMAL_REDLINE,
};
pub use thermoception::{
    CoolingAction, ThermalState, ThermalZone, Thermoceptor, ThermoceptorConfig,
};

// Re-export foundation types
pub use axis_p::{
    AxisPReport, BootstrapResult, ControlComparison, ControlResult, ControlRunner, ControlType,
    Decision, DecisionCriteria, Experiment, ExperimentConfig, ExperimentSummary, InjectionContext,
    InjectionRecord, InjectionSession, Injector, MIEstimator, Marker, MarkerClass, MarkerGenerator,
    MarkerRegistry, NeutralPromptGenerator, PermutationResult, ProbeOutput, ProbeSession,
    SessionType, StopCondition, TelemetryCheck, TrialResult,
};
pub use baseline::{AnomalyLevel, BaselineRegistry, DualBaseline, DualBaselineConfig};
pub use context_fingerprint::{
    ContextBuilder, ContextFingerprint, FingerprintConfig, FingerprintResult,
};
pub use domains::TrustDomain;
pub use drift::{DriftDetector, DriftResult, DriftStatus, MetricStats, SystemBaseline};
pub use momentum_gate::{
    BidirectionalGate, GateSignal, KuramotoNoise, MomentumGate, MomentumGateConfig, PHI, PHI_INV,
};
pub use observations::{ObsKey, ObsValue, Observation, ObservationBatch};
pub use sensorium::{
    BehaviorHook, IntegratedState, IntegrationResult, SalienceMap, Sensorium, SensoriumConfig,
};
pub use sitrep::{Sitrep, SitrepOutput, SystemSnapshot};
pub use stats::{
    float_cmp, float_cmp_f32, float_cmp_nan_last, CusumDetector, Ewma, RateEstimator, RobustStats,
    VarianceTracker,
};
pub use time::TimePoint;
pub use vestibular::{
    DisorientationLevel, Vestibular, VestibularConfig, VestibularReading, VestibularStatus,
};

// ═══════════════════════════════════════════════════════════════════════════════
// AGI ARCHITECTURE RE-EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

// Grounding Layer — Environmental telemetry, temporal anchoring, causal modeling
pub use grounding::{
    CausalConfig, CausalGraph, CausalModel, CausalNode, CausalOrder, EnvironmentalTelemetry,
    GroundingLayer, GroundingLayerConfig, GroundingState, TelemetryConfig, TemporalAnchor,
    TemporalConfig, TimeScale,
};

// Alignment Layer — Value learning, corrigibility, uncertainty, deference
pub use alignment::{
    AlignedIntent, AlignmentCheckResult, AlignmentLayer, AlignmentLayerConfig, CorrigibilityConfig,
    CorrigibilityCore, CorrigibilityState, DeferenceConfig, DeferenceDecision, DeferenceProtocol,
    DeferenceReason, LearnedValue, ModificationRequest, UncertaintyConfig, UncertaintyEstimate,
    UncertaintyQuantifier, ValueConfig, ValueLearner, ValueProfile, ValueSource,
};

// Containment Layer — Safety immune system
pub use containment::{
    AuthenticationLevel, Boundary, BoundaryConfig, BoundaryEnforcer, BoundaryType,
    BoundaryViolation, ContainmentContext, ContainmentLayer, ContainmentLayerConfig,
    ContainmentResult, IntentAnalysis, IntentCategory, IntentClassifier, IntentConfig,
    ManipulationAttempt, ManipulationResistance, ManipulationType, OperatorConfig,
    OperatorDetector, OperatorProfile, OperatorTrust, ResistanceConfig, ThreatLevel,
};

// Orchestration Layer — Multi-agent coordination (Pod methodology)
pub use orchestration::{
    AgentResponse, AlphaAgent, BetaAgent, Conductor, ConductorConfig, ConflictResolution,
    ConsensusConfig, ConsensusEngine, ConsensusResult, DeltaAgent, GammaAgent, OrchestrationLayer,
    OrchestrationLayerConfig, OrchestrationResult, Task,
};

// Cognition Layer — Pattern recognition, abstraction, counterfactuals
pub use cognition::{
    AbstractionConfig, AbstractionHierarchy, CognitionInput, CognitionLayer, CognitionLayerConfig,
    CognitionResult, Concept, ConceptRelation, Counterfactual, CounterfactualConfig,
    CounterfactualEngine, CounterfactualResult, Pattern, PatternConfig, PatternMatch,
    PatternRecognizer, PatternType,
};

// AGI Core — Unified integration of all layers
pub use agi_core::{
    AGIAction, AGICore, AGICoreConfig, AGICycleResult, AGIDecision, AGIInput, AGIState,
    AGIStatistics, FeedbackController,
};

// Forecasting Layer — Salience detection (knowing when you know better than market)
pub use forecasting::{
    ActivePrediction,
    BacktestResult,
    BetDirection,
    BetSignal,
    EvaluationResult,
    // Factor comparison
    Factor,
    FactorComparison,
    FactorDifferential,
    FactorExtractor,
    FactorOutcome,
    FactorWeight,
    MarketFactors,
    PerformanceSnapshot,
    PodFactors,
    ResolvedPrediction,
    // Core salience types
    SalienceAnalysis,
    SalienceConfig,
    SalienceDetector,
    // Evaluation metrics
    SalienceEvaluation,
    SalienceLevel,
    SalienceMetrics,
    SaliencePrecision,
    SalienceSignal,
    // Live tracking
    SalienceTracker,
    TrackerConfig,
    TrackerState,
};
