//! ═══════════════════════════════════════════════════════════════════════════════
//! SALIENCE — The Core Signal
//! ═══════════════════════════════════════════════════════════════════════════════
//! Salience detection: knowing when you know better than the market.
//!
//! Key distinction:
//! - Confidence = P(outcome = X)
//! - Salience = P(I see factor market is mispricing)
//!
//! From backtest evidence (4/4 bet signals won, all LOW or MEDIUM confidence):
//! The system can be uncertain about whether the Fed cuts (LOW confidence)
//! but certain that employment data matters more than the market is pricing.
//! That's a different kind of certainty.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::factor::{FactorDifferential, MarketFactors, PodFactors};
use crate::time::TimePoint;
use std::collections::HashMap;

/// Configuration for salience detection
#[derive(Debug, Clone)]
pub struct SalienceConfig {
    /// Minimum factor differential to be considered salient
    pub salience_threshold: f64,
    /// Minimum aggregate salience score to trigger bet signal
    pub bet_threshold: f64,
    /// Weight decay for older factors in same analysis
    pub recency_weight: f64,
    /// Minimum factors required for bet signal
    pub min_salient_factors: usize,
    /// Maximum bet size as fraction of bankroll (Kelly fraction cap)
    pub max_kelly_fraction: f64,
}

impl Default for SalienceConfig {
    fn default() -> Self {
        Self {
            salience_threshold: 0.15, // Factor differential > 15%
            bet_threshold: 0.25,      // Aggregate salience > 25%
            recency_weight: 0.9,
            min_salient_factors: 1,
            max_kelly_fraction: 0.25, // Quarter-Kelly max
        }
    }
}

/// Salience level — how strongly do we see something the market doesn't?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SalienceLevel {
    /// No meaningful factor differential detected
    None,
    /// Some differential but below bet threshold
    Weak,
    /// Clear differential, borderline bet territory
    Moderate,
    /// Strong differential, high-confidence bet signal
    Strong,
}

impl SalienceLevel {
    pub fn from_score(score: f64) -> Self {
        if score < 0.1 {
            Self::None
        } else if score < 0.2 {
            Self::Weak
        } else if score < 0.35 {
            Self::Moderate
        } else {
            Self::Strong
        }
    }

    pub fn is_actionable(&self) -> bool {
        matches!(self, Self::Moderate | Self::Strong)
    }
}

/// Direction of bet signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetDirection {
    /// Bet YES (outcome more likely than market thinks)
    Yes,
    /// Bet NO (outcome less likely than market thinks)
    No,
}

impl BetDirection {
    pub fn from_differential(pod_prediction: f64, market_price: f64) -> Self {
        if pod_prediction > market_price {
            Self::Yes
        } else {
            Self::No
        }
    }
}

/// A signal to bet — the core output when salience is detected
#[derive(Debug, Clone)]
pub struct BetSignal {
    /// Direction of the bet
    pub direction: BetDirection,
    /// Recommended position size (0.0 - 1.0 of bankroll)
    pub position_size: f64,
    /// Expected edge (pod vs market Brier differential)
    pub expected_edge: f64,
    /// The salient factors driving this signal
    pub driving_factors: Vec<FactorDifferential>,
    /// Confidence in the salience detection (NOT outcome confidence)
    pub salience_confidence: f64,
    /// Human-readable reasoning
    pub reasoning: String,
}

/// The core salience signal — what we actually detect
#[derive(Debug, Clone)]
pub struct SalienceSignal {
    /// Aggregate salience score (0.0 - 1.0)
    pub score: f64,
    /// Categorized level
    pub level: SalienceLevel,
    /// Individual factor differentials that contribute
    pub factors: Vec<FactorDifferential>,
    /// The dominant factor (highest differential)
    pub dominant_factor: Option<FactorDifferential>,
    /// Whether this triggers a bet signal
    pub triggers_bet: bool,
}

impl SalienceSignal {
    /// Create from factor differentials
    pub fn from_factors(factors: Vec<FactorDifferential>, config: &SalienceConfig) -> Self {
        // Filter to salient factors only
        let salient: Vec<_> = factors
            .iter()
            .filter(|f| f.differential().abs() >= config.salience_threshold)
            .cloned()
            .collect();

        // Compute aggregate score (weighted by differential magnitude)
        let total_weight: f64 = salient.iter().map(|f| f.differential().abs()).sum();

        let score = if salient.is_empty() {
            0.0
        } else {
            // Normalize to 0-1 range, capped at 1.0
            (total_weight / salient.len() as f64).min(1.0)
        };

        let level = SalienceLevel::from_score(score);

        let dominant_factor = salient
            .iter()
            .max_by(|a, b| {
                a.differential()
                    .abs()
                    .partial_cmp(&b.differential().abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        let triggers_bet =
            score >= config.bet_threshold && salient.len() >= config.min_salient_factors;

        Self {
            score,
            level,
            factors: salient,
            dominant_factor,
            triggers_bet,
        }
    }

    /// Generate bet signal if salience warrants it
    pub fn to_bet_signal(
        &self,
        pod_prediction: f64,
        market_price: f64,
        config: &SalienceConfig,
    ) -> Option<BetSignal> {
        if !self.triggers_bet {
            return None;
        }

        let direction = BetDirection::from_differential(pod_prediction, market_price);

        // Kelly criterion for position sizing, capped
        let edge = (pod_prediction - market_price).abs();
        let odds = match direction {
            BetDirection::Yes => market_price / (1.0 - market_price),
            BetDirection::No => (1.0 - market_price) / market_price,
        };
        let kelly = edge / odds.max(0.01);
        let position_size = kelly.min(config.max_kelly_fraction) * 0.5; // Half-Kelly for safety

        let reasoning = self.generate_reasoning();

        Some(BetSignal {
            direction,
            position_size,
            expected_edge: edge,
            driving_factors: self.factors.clone(),
            salience_confidence: self.score,
            reasoning,
        })
    }

    fn generate_reasoning(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref dominant) = self.dominant_factor {
            parts.push(format!(
                "Primary: {} (pod weight {:.0}% vs market {:.0}%, differential {:.0}%)",
                dominant.factor.name,
                dominant.pod_weight * 100.0,
                dominant.market_weight * 100.0,
                dominant.differential() * 100.0
            ));
        }

        let other_count = self.factors.len().saturating_sub(1);
        if other_count > 0 {
            parts.push(format!("{} additional salient factor(s)", other_count));
        }

        parts.push(format!(
            "Aggregate salience: {:.0}% ({:?})",
            self.score * 100.0,
            self.level
        ));

        parts.join(". ")
    }
}

/// Complete salience analysis — the full output of the detection system
#[derive(Debug, Clone)]
pub struct SalienceAnalysis {
    /// Question identifier
    pub question_id: String,
    /// The salience signal (core output)
    pub signal: SalienceSignal,
    /// Pod's incidental prediction (byproduct, not primary)
    pub pod_prediction: f64,
    /// Market price at analysis time
    pub market_price: f64,
    /// Pod's confidence in prediction (SEPARATE from salience)
    pub prediction_confidence: PredictionConfidence,
    /// Bet signal if warranted
    pub bet_signal: Option<BetSignal>,
    /// All factors considered (not just salient ones)
    pub all_factors: PodFactors,
    /// Market's implied factors
    pub market_factors: MarketFactors,
    /// Timestamp of analysis
    pub timestamp: TimePoint,
    /// Cutoff date for information (backtest mode)
    pub information_cutoff: Option<TimePoint>,
}

/// Confidence in the prediction outcome (distinct from salience)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionConfidence {
    /// Very uncertain about outcome
    Low,
    /// Moderately confident
    Medium,
    /// Highly confident (rare)
    High,
}

impl PredictionConfidence {
    pub fn from_score(confidence: f64) -> Self {
        if confidence < 0.4 {
            Self::Low
        } else if confidence < 0.7 {
            Self::Medium
        } else {
            Self::High
        }
    }
}

impl SalienceAnalysis {
    /// Create a new salience analysis
    pub fn new(
        question_id: String,
        pod_factors: PodFactors,
        market_factors: MarketFactors,
        pod_prediction: f64,
        market_price: f64,
        prediction_confidence: f64,
        config: &SalienceConfig,
    ) -> Self {
        // Extract factor differentials
        let differentials = FactorDifferential::compute_all(&pod_factors, &market_factors);

        // Compute salience signal
        let signal = SalienceSignal::from_factors(differentials, config);

        // Generate bet signal if warranted
        let bet_signal = signal.to_bet_signal(pod_prediction, market_price, config);

        Self {
            question_id,
            signal,
            pod_prediction,
            market_price,
            prediction_confidence: PredictionConfidence::from_score(prediction_confidence),
            bet_signal,
            all_factors: pod_factors,
            market_factors,
            timestamp: TimePoint::now(),
            information_cutoff: None,
        }
    }

    /// The decision function — should we bet?
    pub fn should_bet(&self) -> bool {
        self.bet_signal.is_some()
    }

    /// Get the salience level
    pub fn salience_level(&self) -> SalienceLevel {
        self.signal.level
    }

    /// Was this a "I see something specific" signal?
    pub fn has_salience(&self) -> bool {
        self.signal.level.is_actionable()
    }

    /// Summary for logging
    pub fn summary(&self) -> String {
        let bet_status = if let Some(ref bet) = self.bet_signal {
            format!(
                "BET {:?} ({:.1}% position)",
                bet.direction,
                bet.position_size * 100.0
            )
        } else {
            "NO_BET".to_string()
        };

        format!(
            "[{}] Pod: {:.0}% | Market: {:.0}% | Salience: {:?} ({:.0}%) | Confidence: {:?} | {}",
            self.question_id,
            self.pod_prediction * 100.0,
            self.market_price * 100.0,
            self.signal.level,
            self.signal.score * 100.0,
            self.prediction_confidence,
            bet_status
        )
    }
}

/// The salience detector — wraps configuration and provides detection interface
pub struct SalienceDetector {
    config: SalienceConfig,
    /// Historical analyses for calibration
    history: Vec<SalienceAnalysis>,
}

impl SalienceDetector {
    pub fn new(config: SalienceConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Analyze a question for salience
    pub fn analyze(
        &mut self,
        question_id: String,
        pod_factors: PodFactors,
        market_factors: MarketFactors,
        pod_prediction: f64,
        market_price: f64,
        prediction_confidence: f64,
    ) -> SalienceAnalysis {
        let analysis = SalienceAnalysis::new(
            question_id,
            pod_factors,
            market_factors,
            pod_prediction,
            market_price,
            prediction_confidence,
            &self.config,
        );

        self.history.push(analysis.clone());
        analysis
    }

    /// Get historical bet signal rate
    pub fn bet_signal_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let bet_count = self.history.iter().filter(|a| a.should_bet()).count();
        bet_count as f64 / self.history.len() as f64
    }

    /// Get salience distribution
    pub fn salience_distribution(&self) -> HashMap<SalienceLevel, usize> {
        let mut dist = HashMap::new();
        for analysis in &self.history {
            *dist.entry(analysis.salience_level()).or_insert(0) += 1;
        }
        dist
    }

    /// Get config
    pub fn config(&self) -> &SalienceConfig {
        &self.config
    }

    /// Update config
    pub fn set_config(&mut self, config: SalienceConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::factor::Factor;

    fn make_test_factors() -> (PodFactors, MarketFactors) {
        let pod = PodFactors {
            factors: vec![
                Factor::new(
                    "employment_data",
                    0.7,
                    BetDirection::Yes,
                    "Employment mandate stronger",
                ),
                Factor::new("inflation_trend", 0.3, BetDirection::No, "Inflation sticky"),
            ],
        };

        let market = MarketFactors {
            factors: vec![
                Factor::new("employment_data", 0.3, BetDirection::Yes, ""),
                Factor::new("inflation_trend", 0.5, BetDirection::No, ""),
            ],
        };

        (pod, market)
    }

    #[test]
    fn test_salience_signal_from_factors() {
        let (pod, market) = make_test_factors();
        let differentials = FactorDifferential::compute_all(&pod, &market);
        let config = SalienceConfig::default();

        let signal = SalienceSignal::from_factors(differentials, &config);

        // Employment data has +40% differential, should be salient
        assert!(signal.score > 0.0);
        assert!(signal.dominant_factor.is_some());
        assert_eq!(
            signal.dominant_factor.as_ref().unwrap().factor.name,
            "employment_data"
        );
    }

    #[test]
    fn test_salience_analysis_bet_signal() {
        let (pod, market) = make_test_factors();
        let config = SalienceConfig {
            salience_threshold: 0.1,
            bet_threshold: 0.2,
            ..Default::default()
        };

        let analysis = SalienceAnalysis::new(
            "test_q".to_string(),
            pod,
            market,
            0.55, // Pod prediction
            0.47, // Market price
            0.4,  // Confidence
            &config,
        );

        // Should trigger bet with these settings
        assert!(analysis.should_bet());
        assert!(analysis.bet_signal.is_some());

        let bet = analysis.bet_signal.unwrap();
        assert_eq!(bet.direction, BetDirection::Yes);
    }

    #[test]
    fn test_low_confidence_high_salience() {
        // The key insight: LOW confidence + HIGH salience = valid bet signal
        let (pod, market) = make_test_factors();
        let config = SalienceConfig::default();

        let analysis = SalienceAnalysis::new(
            "fed_dec".to_string(),
            pod,
            market,
            0.55, // Pod prediction (uncertain about outcome)
            0.47, // Market price
            0.3,  // LOW confidence in outcome
            &config,
        );

        // Confidence is LOW
        assert_eq!(analysis.prediction_confidence, PredictionConfidence::Low);

        // But salience should still be detected (employment_data differential)
        assert!(analysis.signal.score > 0.0);
    }
}
