//! Prediction Harness — Calibrate beliefs against reality
//!
//! Generate predictions before tests, score after.
//! Systematically wrong predictions → bad model → switch representations.

use super::crux::Crux;
use super::graph::{ClaimId, ConstraintGraph};

/// A prediction about what will happen
#[derive(Debug, Clone)]
pub struct Prediction {
    /// What we're predicting about
    pub claim_id: ClaimId,
    /// The predicted outcome
    pub expected: bool,
    /// Confidence in this prediction (0.0 - 1.0)
    pub confidence: f64,
    /// What happens if prediction is correct
    pub if_correct: String,
    /// What happens if prediction is wrong
    pub if_wrong: String,
    /// Was this prediction checked?
    pub checked: bool,
    /// Was this prediction correct?
    pub correct: Option<bool>,
}

impl Prediction {
    pub fn new(claim_id: ClaimId, expected: bool, confidence: f64) -> Self {
        Self {
            claim_id,
            expected,
            confidence,
            if_correct: String::new(),
            if_wrong: String::new(),
            checked: false,
            correct: None,
        }
    }

    pub fn with_outcomes(mut self, if_correct: &str, if_wrong: &str) -> Self {
        self.if_correct = if_correct.to_string();
        self.if_wrong = if_wrong.to_string();
        self
    }
}

/// Harness for generating and checking predictions
pub struct PredictionHarness {
    /// Historical predictions for calibration
    history: Vec<Prediction>,
    /// Window size for calibration scoring
    window_size: usize,
}

impl PredictionHarness {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            window_size: 100,
        }
    }

    /// Generate predictions for a crux
    pub fn generate(&mut self, crux: &Crux) -> Vec<Prediction> {
        let mut predictions = Vec::new();

        // Primary prediction: what we expect the test to show
        let expected = crux.expected.unwrap_or(true);
        let confidence = if crux.expected.is_some() { 0.7 } else { 0.5 };

        let mut pred = Prediction::new(crux.target, expected, confidence);
        pred.if_correct = crux.if_true.clone();
        pred.if_wrong = crux.if_false.clone();

        predictions.push(pred);

        predictions
    }

    /// Check a prediction against the graph state
    pub fn check(&mut self, prediction: &Prediction, graph: &ConstraintGraph) -> bool {
        let claim = match graph.get(prediction.claim_id) {
            Some(c) => c,
            None => return false,
        };

        // Prediction was about whether the claim would survive
        let actual = claim.alive && claim.tested;
        let correct = prediction.expected == actual;

        // Record for calibration
        let mut recorded = prediction.clone();
        recorded.checked = true;
        recorded.correct = Some(correct);
        self.history.push(recorded);

        // Trim history to window
        if self.history.len() > self.window_size * 2 {
            self.history = self
                .history
                .split_off(self.history.len() - self.window_size);
        }

        correct
    }

    /// Get calibration score (how well-calibrated are our predictions?)
    ///
    /// Perfect calibration: 70% confident predictions are correct 70% of the time
    pub fn calibration_score(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0; // No data, assume calibrated
        }

        let checked: Vec<_> = self
            .history
            .iter()
            .filter(|p| p.checked && p.correct.is_some())
            .collect();

        if checked.is_empty() {
            return 1.0;
        }

        // Bin predictions by confidence
        let mut bins: Vec<(f64, usize, usize)> = vec![
            (0.5, 0, 0), // 50% confident
            (0.6, 0, 0), // 60% confident
            (0.7, 0, 0), // 70% confident
            (0.8, 0, 0), // 80% confident
            (0.9, 0, 0), // 90% confident
        ];

        for pred in checked {
            let bin_idx = ((pred.confidence - 0.5) * 10.0).floor() as usize;
            let bin_idx = bin_idx.min(bins.len() - 1);
            bins[bin_idx].1 += 1; // total
            if pred.correct.unwrap_or(false) {
                bins[bin_idx].2 += 1; // correct
            }
        }

        // Calculate calibration error
        let mut total_error = 0.0;
        let mut total_count = 0;

        for (expected_rate, count, correct) in bins {
            if count > 0 {
                let actual_rate = correct as f64 / count as f64;
                let error = (expected_rate - actual_rate).abs();
                total_error += error * count as f64;
                total_count += count;
            }
        }

        if total_count == 0 {
            return 1.0;
        }

        // Return 1 - average calibration error (1.0 = perfect, 0.0 = terrible)
        1.0 - (total_error / total_count as f64)
    }

    /// Get accuracy (raw correct/total)
    pub fn accuracy(&self) -> f64 {
        let checked: Vec<_> = self
            .history
            .iter()
            .filter(|p| p.checked && p.correct.is_some())
            .collect();

        if checked.is_empty() {
            return 1.0;
        }

        let correct = checked
            .iter()
            .filter(|p| p.correct.unwrap_or(false))
            .count();
        correct as f64 / checked.len() as f64
    }

    /// Get number of predictions made
    pub fn count(&self) -> usize {
        self.history.len()
    }

    /// Signal that we need to switch representations
    /// (calibration is bad, predictions systematically wrong)
    pub fn needs_representation_switch(&self) -> bool {
        let calibration = self.calibration_score();
        let accuracy = self.accuracy();

        // Bad calibration OR very low accuracy
        calibration < 0.6 || accuracy < 0.4
    }
}

impl Default for PredictionHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::crux::TestType;
    use super::*;

    #[test]
    fn test_prediction_generation() {
        let crux = Crux {
            target: ClaimId(1),
            falsifier: "test".into(),
            test_type: TestType::Manual,
            if_true: "accept".into(),
            if_false: "reject".into(),
            expected: Some(true),
        };

        let mut harness = PredictionHarness::new();
        let preds = harness.generate(&crux);

        assert_eq!(preds.len(), 1);
        assert!(preds[0].expected);
        assert_eq!(preds[0].confidence, 0.7);
    }

    #[test]
    fn test_calibration_starts_perfect() {
        let harness = PredictionHarness::new();
        assert_eq!(harness.calibration_score(), 1.0);
        assert_eq!(harness.accuracy(), 1.0);
    }
}
