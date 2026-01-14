//! Animacy detection: deviation from Newtonian expectation.

use serde::{Deserialize, Serialize};

use super::physics::PhysicsModel;
use super::trajectory::{PhasePoint, Trajectory};

/// Result of animacy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimacyScore {
    /// Total integrated deviation from expected physics
    /// A(trajectory) = ∫₀ᵀ ‖a(t) - a_expected(t)‖² dt
    pub total_deviation: f64,

    /// Mean deviation per timestep
    pub mean_deviation: f64,

    /// Peak deviation (single largest unexplained acceleration)
    pub peak_deviation: f64,

    /// Number of "decision points" — sudden acceleration changes
    pub decision_points: usize,

    /// Goal-directedness score (trajectory aims at something)
    pub goal_directedness: f64,

    /// Contingency score (responds to environment)
    pub contingency: f64,

    /// Biological motion score (kinematics match living things)
    pub biological_motion: f64,

    /// Final animacy classification
    pub is_animate: bool,

    /// Confidence in classification
    pub confidence: f64,
}

impl AnimacyScore {
    /// Interpretation of the score
    pub fn interpretation(&self) -> &'static str {
        if !self.is_animate {
            if self.mean_deviation < 0.01 {
                "INANIMATE: Following physical law precisely"
            } else {
                "INANIMATE: Minor deviations within noise threshold"
            }
        } else if self.goal_directedness > 0.7 {
            "ANIMATE: Strong goal-directed behavior"
        } else if self.decision_points > 3 {
            "ANIMATE: Multiple decision points detected"
        } else if self.biological_motion > 0.5 {
            "ANIMATE: Biological motion pattern"
        } else {
            "ANIMATE: Self-propelled, unexplained by physics"
        }
    }
}

/// Computes animacy from trajectory deviation from physics
#[derive(Debug, Clone)]
pub struct AnimacyDetector {
    /// Physics model for expected behavior
    physics: PhysicsModel,

    /// Threshold for classifying as animate
    animacy_threshold: f64,

    /// Threshold for detecting "decision points" (sudden acceleration changes)
    decision_threshold: f64,

    /// Smoothing window for acceleration estimation
    smoothing_window: usize,
}

impl AnimacyDetector {
    pub fn new() -> Self {
        Self {
            physics: PhysicsModel::default(),
            animacy_threshold: 0.5,
            decision_threshold: 2.0, // >2σ acceleration change = decision
            smoothing_window: 3,
        }
    }

    pub fn with_physics(mut self, physics: PhysicsModel) -> Self {
        self.physics = physics;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.animacy_threshold = threshold;
        self
    }

    /// Set smoothing window for acceleration estimation
    pub fn with_smoothing_window(mut self, window: usize) -> Self {
        self.smoothing_window = window.max(1);
        self
    }

    /// Get current smoothing window size
    pub fn smoothing_window(&self) -> usize {
        self.smoothing_window
    }

    /// Smooth a sequence of values using the configured window
    fn smooth(&self, values: &[[f64; 2]]) -> Vec<[f64; 2]> {
        if values.len() < self.smoothing_window || self.smoothing_window <= 1 {
            return values.to_vec();
        }

        let mut smoothed = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            let start = i.saturating_sub(self.smoothing_window / 2);
            let end = (i + self.smoothing_window / 2 + 1).min(values.len());
            let window = &values[start..end];

            let avg_x = window.iter().map(|v| v[0]).sum::<f64>() / window.len() as f64;
            let avg_y = window.iter().map(|v| v[1]).sum::<f64>() / window.len() as f64;
            smoothed.push([avg_x, avg_y]);
        }
        smoothed
    }

    /// Compute animacy score for a trajectory
    pub fn analyze(&self, trajectory: &Trajectory) -> AnimacyScore {
        if trajectory.len() < 3 {
            return AnimacyScore {
                total_deviation: 0.0,
                mean_deviation: 0.0,
                peak_deviation: 0.0,
                decision_points: 0,
                goal_directedness: 0.0,
                contingency: 0.0,
                biological_motion: 0.0,
                is_animate: false,
                confidence: 0.0,
            };
        }

        // Compute observed accelerations (with smoothing if configured)
        let raw_acc = trajectory.accelerations();
        let observed_acc = self.smooth(&raw_acc);

        // Compute expected accelerations
        let expected_acc: Vec<[f64; 2]> = trajectory
            .points
            .iter()
            .take(trajectory.len() - 1)
            .map(|p| self.physics.expected_acceleration(p))
            .collect();

        // Compute deviations
        let deviations: Vec<f64> = observed_acc
            .iter()
            .zip(expected_acc.iter())
            .map(|(obs, exp)| {
                let dx = obs[0] - exp[0];
                let dy = obs[1] - exp[1];
                (dx * dx + dy * dy).sqrt()
            })
            .collect();

        let total_deviation: f64 = deviations.iter().sum();
        let mean_deviation = if deviations.is_empty() {
            0.0
        } else {
            total_deviation / deviations.len() as f64
        };
        let peak_deviation = deviations.iter().cloned().fold(0.0, f64::max);

        // Count decision points (sudden acceleration changes)
        let decision_points = self.count_decision_points(&observed_acc);

        // Goal-directedness: does trajectory aim at something?
        let goal_directedness = self.compute_goal_directedness(trajectory);

        // Biological motion: characteristic oscillation patterns
        let biological_motion = self.compute_biological_motion(trajectory);

        // Contingency: would need external reference, set to 0 for now
        let contingency = 0.0;

        // Classification
        let animacy_evidence = mean_deviation / self.animacy_threshold.max(0.01);
        let is_animate = animacy_evidence > 1.0
            || decision_points >= 2
            || goal_directedness > 0.6
            || biological_motion > 0.5;

        let confidence = if is_animate {
            (animacy_evidence.min(3.0) / 3.0)
                .max(goal_directedness)
                .max(biological_motion)
        } else {
            1.0 - animacy_evidence.min(1.0)
        };

        AnimacyScore {
            total_deviation,
            mean_deviation,
            peak_deviation,
            decision_points,
            goal_directedness,
            contingency,
            biological_motion,
            is_animate,
            confidence,
        }
    }

    /// Count points where acceleration changes abruptly (decision points)
    fn count_decision_points(&self, accelerations: &[[f64; 2]]) -> usize {
        if accelerations.len() < 2 {
            return 0;
        }

        // Compute acceleration change magnitudes
        let acc_changes: Vec<f64> = accelerations
            .windows(2)
            .map(|w| {
                let dx = w[1][0] - w[0][0];
                let dy = w[1][1] - w[0][1];
                (dx * dx + dy * dy).sqrt()
            })
            .collect();

        // Compute statistics
        let mean: f64 = acc_changes.iter().sum::<f64>() / acc_changes.len() as f64;
        let variance: f64 =
            acc_changes.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / acc_changes.len() as f64;
        let std_dev = variance.sqrt().max(0.001);

        // Count points exceeding threshold
        acc_changes
            .iter()
            .filter(|&&x| (x - mean) / std_dev > self.decision_threshold)
            .count()
    }

    /// Compute goal-directedness: does trajectory converge toward a point?
    fn compute_goal_directedness(&self, trajectory: &Trajectory) -> f64 {
        if trajectory.len() < 5 {
            return 0.0;
        }

        // Simple heuristic: check if velocity vectors are converging
        // toward a common point (potential goal)

        let points: Vec<&PhasePoint> = trajectory.points.iter().collect();
        let n = points.len();

        // Take last point as potential goal
        let goal = points[n - 1].position;

        // Check how well velocities point toward goal
        let mut alignment_sum = 0.0;
        let mut count = 0;

        for (i, p) in points.iter().enumerate().take(n - 1) {
            let to_goal = [goal[0] - p.position[0], goal[1] - p.position[1]];
            let to_goal_len = (to_goal[0].powi(2) + to_goal[1].powi(2)).sqrt();

            if to_goal_len < 0.01 || p.speed() < 0.01 {
                continue;
            }

            // Cosine of angle between velocity and direction to goal
            let dot = p.velocity[0] * to_goal[0] + p.velocity[1] * to_goal[1];
            let cos_angle = dot / (p.speed() * to_goal_len);

            // Weight by distance traveled (later points matter more for goal detection)
            let weight = (i as f64 + 1.0) / n as f64;
            alignment_sum += cos_angle * weight;
            count += 1;
        }

        if count == 0 {
            return 0.0;
        }

        // Normalize and map to [0, 1]
        let mean_alignment = alignment_sum / count as f64;
        ((mean_alignment + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Compute biological motion score based on trajectory characteristics
    fn compute_biological_motion(&self, trajectory: &Trajectory) -> f64 {
        if trajectory.len() < 10 {
            return 0.0;
        }

        // Biological motion characteristics:
        // 1. Oscillatory patterns (walking has characteristic frequency)
        // 2. Smooth but variable speed
        // 3. Curvature varies smoothly

        let speeds: Vec<f64> = trajectory.points.iter().map(|p| p.speed()).collect();
        let curvatures = trajectory.curvatures();

        // Speed variability (biological motion has rhythmic speed changes)
        let speed_mean: f64 = speeds.iter().sum::<f64>() / speeds.len() as f64;
        let speed_var: f64 =
            speeds.iter().map(|s| (s - speed_mean).powi(2)).sum::<f64>() / speeds.len() as f64;
        let speed_cv = if speed_mean > 0.01 {
            speed_var.sqrt() / speed_mean
        } else {
            0.0
        };

        // Biological motion has moderate speed variability (CV ~ 0.2-0.5)
        let speed_score = if speed_cv > 0.1 && speed_cv < 1.0 {
            1.0 - ((speed_cv - 0.3).abs() / 0.3).min(1.0)
        } else {
            0.0
        };

        // Curvature smoothness (biological motion is smooth, not jerky)
        let curvature_score = if curvatures.len() >= 2 {
            let curv_changes: Vec<f64> =
                curvatures.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
            let max_change = curv_changes.iter().cloned().fold(0.0, f64::max);
            (1.0 - max_change.min(1.0)).max(0.0)
        } else {
            0.0
        };

        // Combined score
        (speed_score + curvature_score) / 2.0
    }
}

impl Default for AnimacyDetector {
    fn default() -> Self {
        Self::new()
    }
}
