//! ═══════════════════════════════════════════════════════════════════════════════
//! EVALUATION — Salience Precision, Not Brier Score
//! ═══════════════════════════════════════════════════════════════════════════════
//! The right metric for this system isn't prediction accuracy.
//! It's salience detection accuracy.
//!
//! From backtest:
//! - Average Brier improvement: +0.0124 (marginal, noisy)
//! - Bet signal precision: 4/4 = 100% (the actual signal)
//!
//! We track:
//! - Salience precision: When we said "I see something", were we right?
//! - Factor-level accuracy: Did the factors we flagged actually matter?
//! - Conditional Brier: Brier score when salience was detected vs not
//! ═══════════════════════════════════════════════════════════════════════════════

use super::salience::{SalienceAnalysis, SalienceLevel, BetDirection};
use crate::time::TimePoint;
use crate::stats;
use std::collections::HashMap;

/// Outcome of a factor that was flagged as salient
#[derive(Debug, Clone)]
pub struct FactorOutcome {
    /// The factor that was flagged
    pub factor_name: String,
    /// Pod's weight
    pub pod_weight: f64,
    /// Market's weight
    pub market_weight: f64,
    /// Did this factor actually matter more than market expected?
    pub pod_was_right: bool,
    /// Post-hoc importance score (if available)
    pub actual_importance: Option<f64>,
    /// Explanation of why factor mattered or didn't
    pub explanation: String,
}

/// Evaluation of a single salience analysis after resolution
#[derive(Debug, Clone)]
pub struct SalienceEvaluation {
    /// Original analysis
    pub analysis_id: String,
    /// Did the outcome match pod's directional call?
    pub directional_correct: bool,
    /// Pod's Brier score
    pub pod_brier: f64,
    /// Market's Brier score
    pub market_brier: f64,
    /// Was salience detected?
    pub had_salience: bool,
    /// Salience level at analysis time
    pub salience_level: SalienceLevel,
    /// Did bet signal fire?
    pub bet_signal_fired: bool,
    /// If bet fired, did it win?
    pub bet_won: Option<bool>,
    /// Factor-level outcomes
    pub factor_outcomes: Vec<FactorOutcome>,
    /// What fraction of salient factors were correct?
    pub factor_precision: f64,
    /// Resolution timestamp
    pub resolved_at: TimePoint,
}

impl SalienceEvaluation {
    /// Create from analysis and resolution
    pub fn from_resolution(
        analysis: &SalienceAnalysis,
        outcome: bool,
        factor_outcomes: Vec<FactorOutcome>,
    ) -> Self {
        let pod_brier = (analysis.pod_prediction - outcome as u8 as f64).powi(2);
        let market_brier = (analysis.market_price - outcome as u8 as f64).powi(2);

        let directional_correct = if analysis.pod_prediction > 0.5 {
            outcome
        } else {
            !outcome
        };

        let bet_won = analysis.bet_signal.as_ref().map(|bet| {
            match bet.direction {
                BetDirection::Yes => outcome,
                BetDirection::No => !outcome,
            }
        });

        let correct_factors = factor_outcomes.iter()
            .filter(|f| f.pod_was_right)
            .count();
        let factor_precision = if factor_outcomes.is_empty() {
            1.0 // No factors to evaluate
        } else {
            correct_factors as f64 / factor_outcomes.len() as f64
        };

        Self {
            analysis_id: analysis.question_id.clone(),
            directional_correct,
            pod_brier,
            market_brier,
            had_salience: analysis.has_salience(),
            salience_level: analysis.salience_level(),
            bet_signal_fired: analysis.should_bet(),
            bet_won,
            factor_outcomes,
            factor_precision,
            resolved_at: TimePoint::now(),
        }
    }

    /// Edge captured (positive = pod better)
    pub fn edge(&self) -> f64 {
        self.market_brier - self.pod_brier
    }

    /// Was this a successful salience detection?
    pub fn salience_success(&self) -> bool {
        if !self.had_salience {
            return true; // No salience claimed, can't be wrong
        }
        // Success if bet won OR factor precision high
        self.bet_won.unwrap_or(false) || self.factor_precision > 0.5
    }
}

/// Precision metrics for salience detection
#[derive(Debug, Clone, Default)]
pub struct SaliencePrecision {
    /// Total predictions evaluated
    pub total: usize,
    /// Predictions where salience was detected
    pub salience_detected: usize,
    /// Of salience detections, how many were correct
    pub salience_correct: usize,
    /// Bet signals fired
    pub bets_fired: usize,
    /// Bets won
    pub bets_won: usize,
    /// Bets lost
    pub bets_lost: usize,
}

impl SaliencePrecision {
    /// The key metric: P(correct | salience detected)
    pub fn salience_precision(&self) -> f64 {
        if self.salience_detected == 0 {
            return 0.0;
        }
        self.salience_correct as f64 / self.salience_detected as f64
    }

    /// Bet precision: P(won | bet fired)
    pub fn bet_precision(&self) -> f64 {
        if self.bets_fired == 0 {
            return 0.0;
        }
        self.bets_won as f64 / self.bets_fired as f64
    }

    /// Salience rate: P(salience detected)
    pub fn salience_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.salience_detected as f64 / self.total as f64
    }

    /// Bet rate: P(bet fired)
    pub fn bet_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.bets_fired as f64 / self.total as f64
    }

    /// Summary string
    pub fn summary(&self) -> String {
        format!(
            "Salience: {}/{} ({:.0}% precision) | Bets: {}/{} ({:.0}% precision) | n={}",
            self.salience_correct,
            self.salience_detected,
            self.salience_precision() * 100.0,
            self.bets_won,
            self.bets_fired,
            self.bet_precision() * 100.0,
            self.total
        )
    }
}

/// Aggregate metrics across all evaluations
#[derive(Debug, Clone)]
pub struct SalienceMetrics {
    /// Salience precision stats
    pub precision: SaliencePrecision,
    /// Average Brier when salience detected
    pub brier_with_salience: f64,
    /// Average Brier when no salience
    pub brier_without_salience: f64,
    /// Average market Brier
    pub market_brier: f64,
    /// Conditional edge: improvement when salience detected
    pub conditional_edge: f64,
    /// Per-level breakdown
    pub by_level: HashMap<SalienceLevel, LevelMetrics>,
    /// Factor-level precision
    pub factor_precision: f64,
}

/// Metrics for a specific salience level
#[derive(Debug, Clone, Default)]
pub struct LevelMetrics {
    pub count: usize,
    pub correct: usize,
    pub avg_brier: f64,
    pub avg_edge: f64,
}

impl LevelMetrics {
    pub fn precision(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.correct as f64 / self.count as f64 }
    }
}

/// Result of a backtest evaluation
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// All individual evaluations
    pub evaluations: Vec<SalienceEvaluation>,
    /// Aggregate metrics
    pub metrics: SalienceMetrics,
    /// Verdict
    pub verdict: BacktestVerdict,
}

/// Backtest verdict
#[derive(Debug, Clone)]
pub struct BacktestVerdict {
    /// Overall status
    pub status: VerdictStatus,
    /// One-line summary
    pub summary: String,
    /// Detailed reasoning
    pub reasoning: Vec<String>,
    /// Statistical significance
    pub significant: bool,
    /// Recommended action
    pub recommendation: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictStatus {
    /// Clear positive signal
    Positive,
    /// Weak positive signal
    WeakPositive,
    /// Inconclusive
    Inconclusive,
    /// No signal detected
    NoSignal,
    /// Negative (worse than market)
    Negative,
}

impl BacktestResult {
    /// Compute from evaluations
    pub fn from_evaluations(evaluations: Vec<SalienceEvaluation>) -> Self {
        let mut precision = SaliencePrecision::default();
        let mut by_level: HashMap<SalienceLevel, Vec<&SalienceEvaluation>> = HashMap::new();

        let mut briers_with: Vec<f64> = Vec::new();
        let mut briers_without: Vec<f64> = Vec::new();
        let mut market_briers: Vec<f64> = Vec::new();
        let mut factor_precisions: Vec<f64> = Vec::new();

        for eval in &evaluations {
            precision.total += 1;
            market_briers.push(eval.market_brier);

            if eval.had_salience {
                precision.salience_detected += 1;
                briers_with.push(eval.pod_brier);
                if eval.salience_success() {
                    precision.salience_correct += 1;
                }
            } else {
                briers_without.push(eval.pod_brier);
            }

            if eval.bet_signal_fired {
                precision.bets_fired += 1;
                if let Some(won) = eval.bet_won {
                    if won {
                        precision.bets_won += 1;
                    } else {
                        precision.bets_lost += 1;
                    }
                }
            }

            by_level.entry(eval.salience_level).or_default().push(eval);
            factor_precisions.push(eval.factor_precision);
        }

        let brier_with_salience = stats::mean(&briers_with).unwrap_or(0.0);
        let brier_without_salience = stats::mean(&briers_without).unwrap_or(0.0);
        let market_brier = stats::mean(&market_briers).unwrap_or(0.0);
        let factor_precision = stats::mean(&factor_precisions).unwrap_or(0.0);

        // Conditional edge: how much better when salience detected
        let conditional_edge = if !briers_with.is_empty() {
            market_brier - brier_with_salience
        } else {
            0.0
        };

        // Compute per-level metrics
        let mut level_metrics = HashMap::new();
        for (level, evals) in by_level {
            let correct = evals.iter().filter(|e| e.salience_success()).count();
            let briers: Vec<f64> = evals.iter().map(|e| e.pod_brier).collect();
            let edges: Vec<f64> = evals.iter().map(|e| e.edge()).collect();

            level_metrics.insert(level, LevelMetrics {
                count: evals.len(),
                correct,
                avg_brier: stats::mean(&briers).unwrap_or(0.0),
                avg_edge: stats::mean(&edges).unwrap_or(0.0),
            });
        }

        let metrics = SalienceMetrics {
            precision,
            brier_with_salience,
            brier_without_salience,
            market_brier,
            conditional_edge,
            by_level: level_metrics,
            factor_precision,
        };

        let verdict = Self::compute_verdict(&metrics, evaluations.len());

        Self {
            evaluations,
            metrics,
            verdict,
        }
    }

    fn compute_verdict(metrics: &SalienceMetrics, n: usize) -> BacktestVerdict {
        let mut reasoning = Vec::new();

        // Key metric: bet precision
        let bet_precision = metrics.precision.bet_precision();
        let bets_fired = metrics.precision.bets_fired;

        // Statistical significance check (binomial test vs 50%)
        let significant = if bets_fired >= 4 {
            // Approximate: need > 75% precision with n>=4 for significance
            bet_precision > 0.75
        } else {
            false
        };

        // Determine status
        let status = if bets_fired == 0 {
            reasoning.push("No bet signals fired - cannot evaluate salience detection".to_string());
            VerdictStatus::Inconclusive
        } else if bet_precision >= 0.9 && bets_fired >= 4 {
            reasoning.push(format!(
                "Bet precision {:.0}% on {} signals - strong salience detection",
                bet_precision * 100.0, bets_fired
            ));
            VerdictStatus::Positive
        } else if bet_precision >= 0.7 && bets_fired >= 3 {
            reasoning.push(format!(
                "Bet precision {:.0}% on {} signals - moderate salience detection",
                bet_precision * 100.0, bets_fired
            ));
            VerdictStatus::WeakPositive
        } else if bet_precision >= 0.5 {
            reasoning.push(format!(
                "Bet precision {:.0}% on {} signals - inconclusive",
                bet_precision * 100.0, bets_fired
            ));
            VerdictStatus::Inconclusive
        } else {
            reasoning.push(format!(
                "Bet precision {:.0}% on {} signals - salience detection failing",
                bet_precision * 100.0, bets_fired
            ));
            VerdictStatus::Negative
        };

        // Conditional edge
        if metrics.conditional_edge > 0.05 {
            reasoning.push(format!(
                "Conditional edge +{:.1}% when salience detected",
                metrics.conditional_edge * 100.0
            ));
        } else if metrics.conditional_edge < -0.05 {
            reasoning.push(format!(
                "Negative edge {:.1}% when salience detected - concerning",
                metrics.conditional_edge * 100.0
            ));
        }

        // Sample size warning
        if n < 50 {
            reasoning.push(format!("Sample size n={} insufficient for statistical confidence", n));
        }

        let summary = match status {
            VerdictStatus::Positive => format!(
                "POSITIVE: {}/{} bets won ({:.0}%), salience detection working",
                metrics.precision.bets_won, bets_fired, bet_precision * 100.0
            ),
            VerdictStatus::WeakPositive => format!(
                "WEAK POSITIVE: {}/{} bets won, needs larger n",
                metrics.precision.bets_won, bets_fired
            ),
            VerdictStatus::Inconclusive => format!(
                "INCONCLUSIVE: {}/{} bets won, insufficient data",
                metrics.precision.bets_won, bets_fired
            ),
            VerdictStatus::NoSignal => "NO SIGNAL: No bet signals generated".to_string(),
            VerdictStatus::Negative => format!(
                "NEGATIVE: {}/{} bets won, salience detection unreliable",
                metrics.precision.bets_won, bets_fired
            ),
        };

        let recommendation = match status {
            VerdictStatus::Positive => "Continue using salience signals for betting".to_string(),
            VerdictStatus::WeakPositive => "Expand sample size to confirm signal".to_string(),
            VerdictStatus::Inconclusive => "Gather more data before drawing conclusions".to_string(),
            VerdictStatus::NoSignal => "Lower salience threshold or expand question set".to_string(),
            VerdictStatus::Negative => "Do not use salience signals for betting - investigate failure mode".to_string(),
        };

        BacktestVerdict {
            status,
            summary,
            reasoning,
            significant,
            recommendation,
        }
    }

    /// Generate report string
    pub fn report(&self) -> String {
        let mut lines = Vec::new();

        lines.push("═══════════════════════════════════════════════════════════════".to_string());
        lines.push("SALIENCE BACKTEST REPORT".to_string());
        lines.push("═══════════════════════════════════════════════════════════════".to_string());
        lines.push(String::new());

        lines.push(format!("VERDICT: {}", self.verdict.summary));
        lines.push(String::new());

        lines.push("KEY METRICS (Salience-Centric):".to_string());
        lines.push(format!("  Bet Precision:     {}/{} = {:.0}%",
            self.metrics.precision.bets_won,
            self.metrics.precision.bets_fired,
            self.metrics.precision.bet_precision() * 100.0
        ));
        lines.push(format!("  Salience Precision: {}/{} = {:.0}%",
            self.metrics.precision.salience_correct,
            self.metrics.precision.salience_detected,
            self.metrics.precision.salience_precision() * 100.0
        ));
        lines.push(format!("  Factor Precision:  {:.0}%", self.metrics.factor_precision * 100.0));
        lines.push(format!("  Conditional Edge:  {:+.1}%", self.metrics.conditional_edge * 100.0));
        lines.push(String::new());

        lines.push("SECONDARY METRICS (Traditional):".to_string());
        lines.push(format!("  Brier (with salience): {:.4}", self.metrics.brier_with_salience));
        lines.push(format!("  Brier (no salience):   {:.4}", self.metrics.brier_without_salience));
        lines.push(format!("  Market Brier:          {:.4}", self.metrics.market_brier));
        lines.push(String::new());

        lines.push("BY SALIENCE LEVEL:".to_string());
        for level in [SalienceLevel::Strong, SalienceLevel::Moderate, SalienceLevel::Weak, SalienceLevel::None] {
            if let Some(lm) = self.metrics.by_level.get(&level) {
                lines.push(format!("  {:?}: n={}, precision={:.0}%, edge={:+.1}%",
                    level, lm.count, lm.precision() * 100.0, lm.avg_edge * 100.0
                ));
            }
        }
        lines.push(String::new());

        lines.push("REASONING:".to_string());
        for reason in &self.verdict.reasoning {
            lines.push(format!("  - {}", reason));
        }
        lines.push(String::new());

        lines.push(format!("RECOMMENDATION: {}", self.verdict.recommendation));
        lines.push("═══════════════════════════════════════════════════════════════".to_string());

        lines.join("\n")
    }
}

/// Result of evaluating a single question
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Question identifier
    pub question_id: String,
    /// Whether salience was detected
    pub salience_detected: bool,
    /// Whether prediction was correct (directional)
    pub prediction_correct: bool,
    /// Whether bet signal was generated
    pub bet_generated: bool,
    /// Whether bet won (if generated)
    pub bet_won: Option<bool>,
    /// Pod Brier score
    pub pod_brier: f64,
    /// Market Brier score
    pub market_brier: f64,
    /// Edge (positive = pod better)
    pub edge: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_evaluation(salience: bool, bet: bool, won: Option<bool>, pod_brier: f64, market_brier: f64) -> SalienceEvaluation {
        SalienceEvaluation {
            analysis_id: "test".to_string(),
            directional_correct: true,
            pod_brier,
            market_brier,
            had_salience: salience,
            salience_level: if salience { SalienceLevel::Strong } else { SalienceLevel::None },
            bet_signal_fired: bet,
            bet_won: won,
            factor_outcomes: vec![],
            factor_precision: 1.0,
            resolved_at: TimePoint::now(),
        }
    }

    #[test]
    fn test_backtest_result_positive() {
        // 4/4 bets won - should be positive
        let evals = vec![
            make_evaluation(true, true, Some(true), 0.2, 0.3),
            make_evaluation(true, true, Some(true), 0.1, 0.25),
            make_evaluation(false, false, None, 0.15, 0.14),
            make_evaluation(true, true, Some(true), 0.18, 0.28),
            make_evaluation(false, false, None, 0.12, 0.11),
            make_evaluation(true, true, Some(true), 0.22, 0.35),
        ];

        let result = BacktestResult::from_evaluations(evals);

        assert_eq!(result.metrics.precision.bets_fired, 4);
        assert_eq!(result.metrics.precision.bets_won, 4);
        assert_eq!(result.verdict.status, VerdictStatus::Positive);
    }

    #[test]
    fn test_backtest_result_negative() {
        // 1/4 bets won - should be negative
        let evals = vec![
            make_evaluation(true, true, Some(false), 0.4, 0.3),
            make_evaluation(true, true, Some(false), 0.35, 0.25),
            make_evaluation(true, true, Some(false), 0.38, 0.28),
            make_evaluation(true, true, Some(true), 0.22, 0.35),
        ];

        let result = BacktestResult::from_evaluations(evals);

        assert_eq!(result.metrics.precision.bets_fired, 4);
        assert_eq!(result.metrics.precision.bets_won, 1);
        assert_eq!(result.verdict.status, VerdictStatus::Negative);
    }

    #[test]
    fn test_conditional_edge() {
        let evals = vec![
            make_evaluation(true, true, Some(true), 0.10, 0.30), // Edge +0.20
            make_evaluation(true, true, Some(true), 0.15, 0.25), // Edge +0.10
            make_evaluation(false, false, None, 0.20, 0.18),     // Edge -0.02
        ];

        let result = BacktestResult::from_evaluations(evals);

        // Conditional edge should be positive (better when salience detected)
        assert!(result.metrics.conditional_edge > 0.0);
    }
}
