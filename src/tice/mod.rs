//! ═══════════════════════════════════════════════════════════════════════════════
//! TICE — Type-I-honest Constraint Engine
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Not "think better." Cut faster.
//!
//! Core loop:
//!   1. Select target (highest uncertainty × highest value)
//!   2. Extract crux (one fact that collapses branches)
//!   3. Generate predictions (what changes if each side true)
//!   4. Execute test
//!   5. Propagate constraints (kill branches)
//!   6. Commit or continue
//!
//! Progress = fewer worlds remain.
//!
//! Qualia integration:
//!   - Connects to operator state for fatigue gating
//!   - Defers when operator fatigue > threshold
//!   - Respects embodied constraints on cognition
//!
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod crux;
pub mod graph;
pub mod metrics;
pub mod noci_bridge;
pub mod predictions;
pub mod propagator;
pub mod representations;

pub use crux::{Crux, CruxExtractor, TestType};
pub use graph::{Claim, ClaimId, Constraint, ConstraintGraph, Observable};
pub use metrics::TICEMetrics;
pub use noci_bridge::{TiceEvent, TiceNociBridge, TiceNociception};
pub use predictions::{Prediction, PredictionHarness};
pub use propagator::ConstraintPropagator;
pub use representations::{Representation, RepresentationChurner};

use std::time::Instant;

#[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
use crate::qualia::OperatorState;
#[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
use parking_lot::RwLock;
#[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
use std::sync::Arc;

/// Outcome of a single TICE iteration
#[derive(Debug, Clone)]
pub enum Outcome {
    /// Decision reached, graph collapsed to single branch
    Commit(ClaimId),
    /// Continue iterating, branches remain
    Continue { kills: usize, remaining: usize },
    /// Deferred due to external constraint (e.g., operator fatigue)
    Defer(String),
    /// No crux found, problem may be ill-posed
    Stuck(String),
}

/// The Type-I-honest Constraint Engine
pub struct TICE {
    pub graph: ConstraintGraph,
    pub crux_extractor: CruxExtractor,
    pub propagator: ConstraintPropagator,
    pub predictions: PredictionHarness,
    pub representations: RepresentationChurner,
    pub metrics: TICEMetrics,
    iteration: u64,
    /// Fatigue threshold for deferring (default 0.7)
    pub fatigue_threshold: f64,
    /// Connected qualia operator state (for fatigue gating)
    #[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
    qualia_handle: Option<Arc<RwLock<OperatorState>>>,
}

impl TICE {
    pub fn new() -> Self {
        Self {
            graph: ConstraintGraph::new(),
            crux_extractor: CruxExtractor::new(),
            propagator: ConstraintPropagator::new(),
            predictions: PredictionHarness::new(),
            representations: RepresentationChurner::new(),
            metrics: TICEMetrics::new(),
            iteration: 0,
            fatigue_threshold: 0.7,
            #[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
            qualia_handle: None,
        }
    }

    /// Connect to qualia operator state for fatigue gating
    #[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
    pub fn connect_qualia(&mut self, handle: Arc<RwLock<OperatorState>>) {
        self.qualia_handle = Some(handle);
    }

    /// Check if qualia is connected
    #[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
    pub fn has_qualia(&self) -> bool {
        self.qualia_handle.is_some()
    }

    /// Read current operator fatigue from qualia (if connected)
    #[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
    fn read_qualia_fatigue(&self) -> Option<f64> {
        self.qualia_handle
            .as_ref()
            .map(|h| h.read().fatigue_estimate)
    }

    /// Read full operator state from qualia (if connected)
    #[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
    pub fn read_operator_state(&self) -> Option<OperatorState> {
        self.qualia_handle.as_ref().map(|h| h.read().clone())
    }

    /// Main iteration loop
    ///
    /// If operator_fatigue is None and qualia is connected, reads from qualia.
    /// Otherwise uses the provided value.
    pub fn iterate(&mut self, operator_fatigue: Option<f64>) -> Outcome {
        let start = Instant::now();
        self.iteration += 1;

        // Get fatigue: prefer explicit arg, fall back to qualia if connected
        #[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
        let fatigue = operator_fatigue.or_else(|| self.read_qualia_fatigue());
        #[cfg(not(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video")))]
        let fatigue = operator_fatigue;

        // Gate on operator fatigue
        if let Some(f) = fatigue {
            if f > self.fatigue_threshold {
                self.metrics.record_defer();
                return Outcome::Defer(format!(
                    "operator fatigue {:.2} > {:.2} threshold",
                    f, self.fatigue_threshold
                ));
            }
        }

        // 1. Select target: highest uncertainty × highest value
        let target = match self.graph.select_target() {
            Some(t) => t,
            None => {
                if self.graph.is_decided() {
                    // Handle edge case: decided but no claims left (all killed)
                    match self.graph.decision() {
                        Some(d) => return Outcome::Commit(d),
                        None => return Outcome::Stuck("all claims eliminated".into()),
                    }
                }
                return Outcome::Stuck("no actionable targets".into());
            }
        };

        // 2. Extract crux
        let crux = match self.crux_extractor.extract(&self.graph, target) {
            Some(c) => c,
            None => return Outcome::Stuck("no crux found for target".into()),
        };

        // 3. Generate predictions
        let predictions = self.predictions.generate(&crux);
        self.metrics.predictions_generated += predictions.len();

        // 4. Execute test (synchronous for now)
        let test_result = self.execute_test(&crux);

        // 5. Propagate constraints
        let kills = self
            .propagator
            .propagate(&mut self.graph, &crux, test_result);
        self.metrics.branches_killed += kills;
        self.metrics.tests_executed += 1;

        if kills > 0 {
            self.metrics.effective_tests += 1;
        }

        // 6. Check predictions
        for pred in &predictions {
            if self.predictions.check(pred, &self.graph) {
                self.metrics.predictions_correct += 1;
            }
        }

        // Record timing
        self.metrics
            .iteration_times
            .push(start.elapsed().as_millis() as u64);

        // 7. Commit or continue
        let remaining = self.graph.live_claims();
        if remaining <= 1 {
            Outcome::Commit(self.graph.decision().unwrap_or(ClaimId(0)))
        } else {
            Outcome::Continue { kills, remaining }
        }
    }

    /// Execute a test based on crux type
    fn execute_test(&self, crux: &Crux) -> bool {
        match &crux.test_type {
            TestType::CodeExecution(code) => {
                // Shell out and check exit code
                std::process::Command::new("sh")
                    .args(["-c", code])
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
            }
            TestType::FileCheck(pattern) => glob::glob(pattern)
                .map(|paths| paths.count() > 0)
                .unwrap_or(false),
            TestType::Lookup(key) => {
                // Placeholder: would query a knowledge store
                !key.is_empty()
            }
            TestType::Manual => {
                // Can't auto-execute, return false to force human review
                false
            }
        }
    }

    /// Run multiple iterations until decision or max reached
    pub fn run(&mut self, max_iterations: usize, operator_fatigue: Option<f64>) -> Outcome {
        for _ in 0..max_iterations {
            match self.iterate(operator_fatigue) {
                Outcome::Commit(id) => return Outcome::Commit(id),
                Outcome::Defer(reason) => return Outcome::Defer(reason),
                Outcome::Stuck(reason) => return Outcome::Stuck(reason),
                Outcome::Continue { .. } => continue,
            }
        }
        Outcome::Continue {
            kills: self.metrics.branches_killed,
            remaining: self.graph.live_claims(),
        }
    }

    /// Get current metrics
    pub fn metrics(&self) -> &TICEMetrics {
        &self.metrics
    }

    /// Get iteration count
    pub fn iteration(&self) -> u64 {
        self.iteration
    }
}

impl Default for TICE {
    fn default() -> Self {
        Self::new()
    }
}
