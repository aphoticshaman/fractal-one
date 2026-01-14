//! Phase transition detection for entity detection systems.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Monitor for entity detection phase transitions
#[derive(Debug, Clone)]
pub struct EntityPhaseMonitor {
    /// History of entity detection rates
    detection_history: VecDeque<f64>,

    /// History of gain values
    gain_history: VecDeque<f64>,

    /// Window size for analysis
    window_size: usize,

    /// Critical gain threshold (estimated)
    estimated_mu_crit: f64,
}

impl EntityPhaseMonitor {
    pub fn new(window_size: usize) -> Self {
        Self {
            detection_history: VecDeque::with_capacity(window_size),
            gain_history: VecDeque::with_capacity(window_size),
            window_size,
            estimated_mu_crit: 1.5, // Initial estimate
        }
    }

    /// Record an observation
    pub fn record(&mut self, detection_rate: f64, gain: f64) {
        if self.detection_history.len() >= self.window_size {
            self.detection_history.pop_front();
            self.gain_history.pop_front();
        }
        self.detection_history.push_back(detection_rate);
        self.gain_history.push_back(gain);

        // Update critical point estimate
        self.update_mu_crit_estimate();
    }

    /// Estimate critical gain (μ_crit) from data
    fn update_mu_crit_estimate(&mut self) {
        if self.detection_history.len() < 5 {
            return;
        }

        // Find the gain where detection rate changes most rapidly
        let mut max_derivative = 0.0;
        let mut max_deriv_gain = self.estimated_mu_crit;

        let detections: Vec<f64> = self.detection_history.iter().cloned().collect();
        let gains: Vec<f64> = self.gain_history.iter().cloned().collect();

        for i in 1..detections.len() {
            let d_detect = detections[i] - detections[i - 1];
            let d_gain = gains[i] - gains[i - 1];

            if d_gain.abs() > 0.01 {
                let derivative = d_detect / d_gain;
                if derivative > max_derivative {
                    max_derivative = derivative;
                    max_deriv_gain = (gains[i] + gains[i - 1]) / 2.0;
                }
            }
        }

        if max_derivative > 0.1 {
            // Significant phase transition detected
            self.estimated_mu_crit = max_deriv_gain;
        }
    }

    /// Current phase based on gain
    pub fn current_phase(&self, gain: f64) -> Phase {
        if gain < self.estimated_mu_crit * 0.7 {
            Phase::Normal
        } else if gain < self.estimated_mu_crit {
            Phase::NearCritical
        } else if gain < self.estimated_mu_crit * 1.5 {
            Phase::Supercritical
        } else {
            Phase::EntitySaturated
        }
    }

    /// Get estimated critical point
    pub fn mu_crit(&self) -> f64 {
        self.estimated_mu_crit
    }
}

/// Phase of the entity detection system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Normal: Appropriate thresholding
    Normal,
    /// Near critical: Approaching phase transition
    NearCritical,
    /// Above critical: Expanded entity detection
    Supercritical,
    /// Saturated: Everything is entity
    EntitySaturated,
}

impl Phase {
    pub fn description(&self) -> &'static str {
        match self {
            Phase::Normal => "Normal entity detection: appropriate thresholds",
            Phase::NearCritical => "Near critical: approaching phase transition",
            Phase::Supercritical => "Supercritical: expanded entity basins, lowered thresholds",
            Phase::EntitySaturated => "Entity-saturated: all inputs classified as agents",
        }
    }
}
