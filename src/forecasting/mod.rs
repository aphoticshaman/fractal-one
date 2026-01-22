//! ═══════════════════════════════════════════════════════════════════════════════
//! FORECASTING — Salience Detection, Not Prediction
//! ═══════════════════════════════════════════════════════════════════════════════
//! The system doesn't know everything better than the market.
//! But it knows when it knows better than the market.
//!
//! Core insight from backtest (4/4 bet signals won):
//! - Confidence: P(outcome = X) — marginal improvement over market
//! - Salience: P(I see factor market is mispricing) — the actual signal
//!
//! These are orthogonal dimensions. You can be uncertain about the outcome
//! while being certain you see the game differently than the crowd.
//!
//! Architecture:
//! ```text
//! WRONG MODEL:
//!   Input: Question
//!   Process: Analyze → Predict
//!   Output: P(outcome)
//!   Metric: Brier score
//!
//! RIGHT MODEL:
//!   Input: Question + Market consensus
//!   Process: Analyze → Compare factor weights
//!   Output: Salience signal (+ incidental prediction)
//!   Metric: Salience precision
//! ```
//!
//! The system is a **differential factor detector** that emits forecasts as exhaust.
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod evaluation;
pub mod factor;
pub mod salience;
pub mod tracker;

pub use evaluation::{
    BacktestResult, EvaluationResult, FactorOutcome, SalienceEvaluation, SalienceMetrics,
    SaliencePrecision,
};
pub use factor::{
    Factor, FactorComparison, FactorDifferential, FactorExtractor, FactorWeight, MarketFactors,
    PodFactors,
};
pub use salience::{
    BetDirection, BetSignal, SalienceAnalysis, SalienceConfig, SalienceDetector, SalienceLevel,
    SalienceSignal,
};
pub use tracker::{
    ActivePrediction, PerformanceSnapshot, ResolvedPrediction, SalienceTracker, TrackerConfig,
    TrackerState,
};
