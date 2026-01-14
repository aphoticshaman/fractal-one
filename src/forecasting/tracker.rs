//! ═══════════════════════════════════════════════════════════════════════════════
//! TRACKER — Live Salience Tracking System
//! ═══════════════════════════════════════════════════════════════════════════════
//! Real-time tracking of salience signals and their outcomes.
//!
//! Responsibilities:
//! - Track active predictions awaiting resolution
//! - Record resolutions and evaluate salience accuracy
//! - Maintain rolling performance metrics
//! - Detect drift in salience detection quality
//! ═══════════════════════════════════════════════════════════════════════════════

use super::salience::{SalienceAnalysis, SalienceConfig};
use super::evaluation::{SalienceEvaluation, SaliencePrecision, BacktestResult, FactorOutcome};
use crate::time::TimePoint;
use crate::stats::Ewma;
use std::collections::{HashMap, VecDeque};

/// Configuration for the tracker
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Maximum active predictions to track
    pub max_active: usize,
    /// History size for resolved predictions
    pub history_size: usize,
    /// EWMA alpha for rolling metrics
    pub ewma_alpha: f64,
    /// Alert threshold for precision drop
    pub precision_alert_threshold: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_active: 1000,
            history_size: 500,
            ewma_alpha: 0.1,
            precision_alert_threshold: 0.5,
        }
    }
}

/// An active prediction awaiting resolution
#[derive(Debug, Clone)]
pub struct ActivePrediction {
    /// The salience analysis
    pub analysis: SalienceAnalysis,
    /// Expected resolution date
    pub resolution_date: Option<TimePoint>,
    /// Source market/platform
    pub source: String,
    /// Category
    pub category: String,
    /// Notes
    pub notes: String,
}

impl ActivePrediction {
    pub fn new(analysis: SalienceAnalysis, source: &str, category: &str) -> Self {
        Self {
            analysis,
            resolution_date: None,
            source: source.to_string(),
            category: category.to_string(),
            notes: String::new(),
        }
    }

    pub fn with_resolution_date(mut self, date: TimePoint) -> Self {
        self.resolution_date = Some(date);
        self
    }

    pub fn question_id(&self) -> &str {
        &self.analysis.question_id
    }

    pub fn has_bet_signal(&self) -> bool {
        self.analysis.should_bet()
    }
}

/// A resolved prediction with evaluation
#[derive(Debug, Clone)]
pub struct ResolvedPrediction {
    /// Original active prediction
    pub prediction: ActivePrediction,
    /// Actual outcome
    pub outcome: bool,
    /// Evaluation result
    pub evaluation: SalienceEvaluation,
    /// Resolution timestamp
    pub resolved_at: TimePoint,
}

/// Current state of the tracker
#[derive(Debug, Clone)]
pub struct TrackerState {
    /// Number of active predictions
    pub active_count: usize,
    /// Number with bet signals
    pub active_bets: usize,
    /// Total resolved
    pub total_resolved: usize,
    /// Current rolling precision
    pub rolling_precision: f64,
    /// Rolling bet precision
    pub rolling_bet_precision: f64,
    /// Is precision degraded?
    pub precision_alert: bool,
}

/// Snapshot of performance at a point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: TimePoint,
    /// Cumulative metrics
    pub metrics: SaliencePrecision,
    /// Rolling precision (EWMA)
    pub rolling_precision: f64,
    /// Rolling bet precision
    pub rolling_bet_precision: f64,
    /// Active predictions count
    pub active_count: usize,
}

/// The salience tracker - maintains live state and history
pub struct SalienceTracker {
    config: TrackerConfig,
    salience_config: SalienceConfig,
    /// Active predictions by question ID
    active: HashMap<String, ActivePrediction>,
    /// Resolved predictions (recent history)
    resolved: VecDeque<ResolvedPrediction>,
    /// Rolling precision tracker
    rolling_precision: Ewma,
    /// Rolling bet precision tracker
    rolling_bet_precision: Ewma,
    /// Cumulative precision stats
    cumulative: SaliencePrecision,
    /// Performance snapshots over time
    snapshots: Vec<PerformanceSnapshot>,
    /// Last snapshot time
    last_snapshot: Option<TimePoint>,
}

impl SalienceTracker {
    pub fn new(config: TrackerConfig, salience_config: SalienceConfig) -> Self {
        Self {
            rolling_precision: Ewma::new(config.ewma_alpha),
            rolling_bet_precision: Ewma::new(config.ewma_alpha),
            config,
            salience_config,
            active: HashMap::new(),
            resolved: VecDeque::new(),
            cumulative: SaliencePrecision::default(),
            snapshots: Vec::new(),
            last_snapshot: None,
        }
    }

    /// Register a new prediction
    pub fn track(&mut self, prediction: ActivePrediction) -> Result<(), TrackerError> {
        if self.active.len() >= self.config.max_active {
            return Err(TrackerError::CapacityExceeded);
        }

        let id = prediction.question_id().to_string();
        if self.active.contains_key(&id) {
            return Err(TrackerError::DuplicateQuestion(id));
        }

        self.active.insert(id, prediction);
        Ok(())
    }

    /// Resolve a prediction with outcome
    pub fn resolve(
        &mut self,
        question_id: &str,
        outcome: bool,
        factor_outcomes: Vec<FactorOutcome>,
    ) -> Result<ResolvedPrediction, TrackerError> {
        let prediction = self.active.remove(question_id)
            .ok_or_else(|| TrackerError::NotFound(question_id.to_string()))?;

        let evaluation = SalienceEvaluation::from_resolution(
            &prediction.analysis,
            outcome,
            factor_outcomes,
        );

        let resolved = ResolvedPrediction {
            prediction,
            outcome,
            evaluation: evaluation.clone(),
            resolved_at: TimePoint::now(),
        };

        // Update cumulative stats
        self.update_cumulative(&evaluation);

        // Update rolling metrics
        self.update_rolling(&evaluation);

        // Archive
        if self.resolved.len() >= self.config.history_size {
            self.resolved.pop_front();
        }
        self.resolved.push_back(resolved.clone());

        // Maybe take snapshot
        self.maybe_snapshot();

        Ok(resolved)
    }

    fn update_cumulative(&mut self, eval: &SalienceEvaluation) {
        self.cumulative.total += 1;

        if eval.had_salience {
            self.cumulative.salience_detected += 1;
            if eval.salience_success() {
                self.cumulative.salience_correct += 1;
            }
        }

        if eval.bet_signal_fired {
            self.cumulative.bets_fired += 1;
            if let Some(won) = eval.bet_won {
                if won {
                    self.cumulative.bets_won += 1;
                } else {
                    self.cumulative.bets_lost += 1;
                }
            }
        }
    }

    fn update_rolling(&mut self, eval: &SalienceEvaluation) {
        // Update salience precision EWMA
        if eval.had_salience {
            let success = if eval.salience_success() { 1.0 } else { 0.0 };
            self.rolling_precision.update(success);
        }

        // Update bet precision EWMA
        if eval.bet_signal_fired {
            if let Some(won) = eval.bet_won {
                let success = if won { 1.0 } else { 0.0 };
                self.rolling_bet_precision.update(success);
            }
        }
    }

    fn maybe_snapshot(&mut self) {
        let now = TimePoint::now();

        // Snapshot every hour or on first resolution
        let should_snapshot = match &self.last_snapshot {
            None => true,
            Some(last) => now.duration_since(last).as_secs() >= 3600,
        };

        if should_snapshot {
            self.snapshots.push(PerformanceSnapshot {
                timestamp: now.clone(),
                metrics: self.cumulative.clone(),
                rolling_precision: self.rolling_precision.value(),
                rolling_bet_precision: self.rolling_bet_precision.value(),
                active_count: self.active.len(),
            });
            self.last_snapshot = Some(now);
        }
    }

    /// Get current state
    pub fn state(&self) -> TrackerState {
        let active_bets = self.active.values()
            .filter(|p| p.has_bet_signal())
            .count();

        let precision_alert = self.rolling_bet_precision.value() < self.config.precision_alert_threshold
            && self.cumulative.bets_fired >= 5;

        TrackerState {
            active_count: self.active.len(),
            active_bets,
            total_resolved: self.cumulative.total,
            rolling_precision: self.rolling_precision.value(),
            rolling_bet_precision: self.rolling_bet_precision.value(),
            precision_alert,
        }
    }

    /// Get cumulative metrics
    pub fn metrics(&self) -> &SaliencePrecision {
        &self.cumulative
    }

    /// Generate backtest result from resolved history
    pub fn to_backtest_result(&self) -> BacktestResult {
        let evaluations: Vec<_> = self.resolved.iter()
            .map(|r| r.evaluation.clone())
            .collect();
        BacktestResult::from_evaluations(evaluations)
    }

    /// Get active prediction by ID
    pub fn get_active(&self, question_id: &str) -> Option<&ActivePrediction> {
        self.active.get(question_id)
    }

    /// Get all active predictions
    pub fn active_predictions(&self) -> impl Iterator<Item = &ActivePrediction> {
        self.active.values()
    }

    /// Get all active bet signals
    pub fn active_bets(&self) -> impl Iterator<Item = &ActivePrediction> {
        self.active.values().filter(|p| p.has_bet_signal())
    }

    /// Get recent resolutions
    pub fn recent_resolutions(&self, n: usize) -> impl Iterator<Item = &ResolvedPrediction> {
        self.resolved.iter().rev().take(n)
    }

    /// Get performance trend (is precision improving or degrading?)
    pub fn precision_trend(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }

        let recent: Vec<_> = self.snapshots.iter().rev().take(5).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let first = recent.last().unwrap().rolling_bet_precision;
        let last = recent.first().unwrap().rolling_bet_precision;

        last - first
    }

    /// Export state for persistence
    pub fn export_state(&self) -> TrackerExport {
        TrackerExport {
            active: self.active.values().cloned().collect(),
            cumulative: self.cumulative.clone(),
            rolling_precision: self.rolling_precision.value(),
            rolling_bet_precision: self.rolling_bet_precision.value(),
            exported_at: TimePoint::now(),
        }
    }

    /// Get salience config
    pub fn salience_config(&self) -> &SalienceConfig {
        &self.salience_config
    }
}

/// Exported tracker state for persistence
#[derive(Debug, Clone)]
pub struct TrackerExport {
    pub active: Vec<ActivePrediction>,
    pub cumulative: SaliencePrecision,
    pub rolling_precision: f64,
    pub rolling_bet_precision: f64,
    pub exported_at: TimePoint,
}

/// Tracker errors
#[derive(Debug, Clone)]
pub enum TrackerError {
    /// Capacity exceeded
    CapacityExceeded,
    /// Duplicate question ID
    DuplicateQuestion(String),
    /// Question not found
    NotFound(String),
    /// Invalid state
    InvalidState(String),
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapacityExceeded => write!(f, "Tracker capacity exceeded"),
            Self::DuplicateQuestion(id) => write!(f, "Duplicate question: {}", id),
            Self::NotFound(id) => write!(f, "Question not found: {}", id),
            Self::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for TrackerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::factor::{Factor, PodFactors, MarketFactors};
    use crate::forecasting::salience::BetDirection;

    fn make_test_analysis(question_id: &str, should_bet: bool) -> SalienceAnalysis {
        let mut pod = PodFactors::new();
        pod.add(Factor::new("key_factor", if should_bet { 0.8 } else { 0.3 }, BetDirection::Yes, ""));

        let mut market = MarketFactors::new();
        market.add(Factor::new("key_factor", 0.3, BetDirection::Yes, ""));

        let config = SalienceConfig {
            salience_threshold: 0.15,
            bet_threshold: 0.2,
            ..Default::default()
        };

        SalienceAnalysis::new(
            question_id.to_string(),
            pod,
            market,
            0.55,
            0.50,
            0.5,
            &config,
        )
    }

    #[test]
    fn test_tracker_basic_flow() {
        let mut tracker = SalienceTracker::new(
            TrackerConfig::default(),
            SalienceConfig::default(),
        );

        // Track a prediction
        let analysis = make_test_analysis("Q1", true);
        let prediction = ActivePrediction::new(analysis, "polymarket", "economics");

        tracker.track(prediction).unwrap();
        assert_eq!(tracker.active.len(), 1);

        // Resolve it
        let resolved = tracker.resolve("Q1", true, vec![]).unwrap();
        assert!(resolved.evaluation.bet_won.unwrap_or(false));

        // Check cumulative stats
        assert_eq!(tracker.cumulative.total, 1);
        assert_eq!(tracker.cumulative.bets_fired, 1);
        assert_eq!(tracker.cumulative.bets_won, 1);
    }

    #[test]
    fn test_tracker_precision_tracking() {
        let mut tracker = SalienceTracker::new(
            TrackerConfig::default(),
            SalienceConfig::default(),
        );

        // Add and resolve 4 winning bets
        for i in 0..4 {
            let id = format!("Q{}", i);
            let analysis = make_test_analysis(&id, true);
            let prediction = ActivePrediction::new(analysis, "test", "test");

            tracker.track(prediction).unwrap();
            tracker.resolve(&id, true, vec![]).unwrap();
        }

        // Should have 100% bet precision
        assert_eq!(tracker.cumulative.bet_precision(), 1.0);

        let state = tracker.state();
        assert!(!state.precision_alert);
    }

    #[test]
    fn test_duplicate_question_error() {
        let mut tracker = SalienceTracker::new(
            TrackerConfig::default(),
            SalienceConfig::default(),
        );

        let analysis = make_test_analysis("Q1", true);
        let prediction = ActivePrediction::new(analysis.clone(), "test", "test");

        tracker.track(prediction).unwrap();

        // Try to add duplicate
        let prediction2 = ActivePrediction::new(analysis, "test", "test");
        let result = tracker.track(prediction2);

        assert!(matches!(result, Err(TrackerError::DuplicateQuestion(_))));
    }
}
