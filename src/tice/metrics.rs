//! TICE Metrics — Measure what matters
//!
//! Progress = fewer worlds remain.
//! Quality = predictions match reality.
//! Speed = time to collapse.

use std::time::Duration;

/// Metrics for TICE operation
#[derive(Debug, Clone)]
pub struct TICEMetrics {
    // Effectiveness
    pub branches_killed: usize,
    pub tests_executed: usize,
    pub effective_tests: usize,

    // Predictions
    pub predictions_generated: usize,
    pub predictions_correct: usize,

    // Representations
    pub representation_switches: usize,

    // Timing
    pub iteration_times: Vec<u64>, // milliseconds
    pub total_runtime: Duration,

    // Decisions
    pub decisions_made: usize,
    pub decisions_deferred: usize,
    pub stuck_count: usize,
}

impl TICEMetrics {
    pub fn new() -> Self {
        Self {
            branches_killed: 0,
            tests_executed: 0,
            effective_tests: 0,
            predictions_generated: 0,
            predictions_correct: 0,
            representation_switches: 0,
            iteration_times: Vec::new(),
            total_runtime: Duration::ZERO,
            decisions_made: 0,
            decisions_deferred: 0,
            stuck_count: 0,
        }
    }

    /// Kill ratio: how effective are our tests?
    pub fn kill_ratio(&self) -> f64 {
        if self.tests_executed == 0 {
            return 0.0;
        }
        self.effective_tests as f64 / self.tests_executed as f64
    }

    /// Prediction accuracy
    pub fn prediction_accuracy(&self) -> f64 {
        if self.predictions_generated == 0 {
            return 1.0;
        }
        self.predictions_correct as f64 / self.predictions_generated as f64
    }

    /// Average iteration time in milliseconds
    pub fn avg_iteration_ms(&self) -> f64 {
        if self.iteration_times.is_empty() {
            return 0.0;
        }
        self.iteration_times.iter().sum::<u64>() as f64 / self.iteration_times.len() as f64
    }

    /// Median iteration time
    pub fn median_iteration_ms(&self) -> u64 {
        if self.iteration_times.is_empty() {
            return 0;
        }
        let mut sorted = self.iteration_times.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    /// Total branches killed per second
    pub fn kills_per_second(&self) -> f64 {
        let secs = self.total_runtime.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.branches_killed as f64 / secs
    }

    /// Decision rate (decisions / total attempts)
    pub fn decision_rate(&self) -> f64 {
        let total = self.decisions_made + self.decisions_deferred + self.stuck_count;
        if total == 0 {
            return 0.0;
        }
        self.decisions_made as f64 / total as f64
    }

    /// Summary string
    pub fn summary(&self) -> String {
        format!(
            "TICE Metrics:\n\
             - Tests: {} executed, {} effective ({:.1}% kill ratio)\n\
             - Branches killed: {}\n\
             - Predictions: {}/{} correct ({:.1}% accuracy)\n\
             - Avg iteration: {:.1}ms\n\
             - Decisions: {} made, {} deferred, {} stuck\n\
             - Representation switches: {}",
            self.tests_executed,
            self.effective_tests,
            self.kill_ratio() * 100.0,
            self.branches_killed,
            self.predictions_correct,
            self.predictions_generated,
            self.prediction_accuracy() * 100.0,
            self.avg_iteration_ms(),
            self.decisions_made,
            self.decisions_deferred,
            self.stuck_count,
            self.representation_switches,
        )
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Record a decision outcome
    pub fn record_decision(&mut self) {
        self.decisions_made += 1;
    }

    /// Record a deferral
    pub fn record_defer(&mut self) {
        self.decisions_deferred += 1;
    }

    /// Record getting stuck
    pub fn record_stuck(&mut self) {
        self.stuck_count += 1;
    }
}

impl Default for TICEMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated metrics across multiple TICE runs
#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    pub runs: Vec<TICEMetrics>,
}

impl AggregateMetrics {
    pub fn new() -> Self {
        Self { runs: Vec::new() }
    }

    pub fn add(&mut self, metrics: TICEMetrics) {
        self.runs.push(metrics);
    }

    pub fn total_branches_killed(&self) -> usize {
        self.runs.iter().map(|m| m.branches_killed).sum()
    }

    pub fn total_tests(&self) -> usize {
        self.runs.iter().map(|m| m.tests_executed).sum()
    }

    pub fn avg_kill_ratio(&self) -> f64 {
        if self.runs.is_empty() {
            return 0.0;
        }
        self.runs.iter().map(|m| m.kill_ratio()).sum::<f64>() / self.runs.len() as f64
    }

    pub fn avg_prediction_accuracy(&self) -> f64 {
        if self.runs.is_empty() {
            return 1.0;
        }
        self.runs
            .iter()
            .map(|m| m.prediction_accuracy())
            .sum::<f64>()
            / self.runs.len() as f64
    }
}

impl Default for AggregateMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_new() {
        let m = TICEMetrics::new();
        assert_eq!(m.branches_killed, 0);
        assert_eq!(m.kill_ratio(), 0.0);
        assert_eq!(m.prediction_accuracy(), 1.0); // No predictions = perfect
    }

    #[test]
    fn test_kill_ratio() {
        let mut m = TICEMetrics::new();
        m.tests_executed = 10;
        m.effective_tests = 7;

        assert!((m.kill_ratio() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_summary() {
        let mut m = TICEMetrics::new();
        m.tests_executed = 5;
        m.effective_tests = 4;
        m.branches_killed = 10;

        let summary = m.summary();
        assert!(summary.contains("Tests: 5"));
        assert!(summary.contains("80.0% kill ratio"));
    }
}
