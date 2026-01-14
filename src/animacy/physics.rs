//! Physics model for expected Newtonian dynamics.

use serde::{Deserialize, Serialize};

use super::trajectory::PhasePoint;

/// Expected physics: what acceleration SHOULD be given physical context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsModel {
    /// Gravity (if applicable)
    pub gravity: [f64; 2],
    /// Drag coefficient
    pub drag: f64,
    /// Known force fields (position → force)
    pub external_forces: Vec<([f64; 2], [f64; 2])>, // (position, force) pairs
}

impl Default for PhysicsModel {
    fn default() -> Self {
        Self {
            gravity: [0.0, 0.0], // No gravity by default (top-down view)
            drag: 0.0,
            external_forces: vec![],
        }
    }
}

impl PhysicsModel {
    /// Predict expected acceleration given current state
    pub fn expected_acceleration(&self, point: &PhasePoint) -> [f64; 2] {
        let mut acc = self.gravity;

        // Add drag (opposes velocity)
        let speed = point.speed();
        if speed > 1e-10 {
            acc[0] -= self.drag * point.velocity[0];
            acc[1] -= self.drag * point.velocity[1];
        }

        // Add external forces (simple nearest-neighbor interpolation)
        for (pos, force) in &self.external_forces {
            let dist_sq =
                (point.position[0] - pos[0]).powi(2) + (point.position[1] - pos[1]).powi(2);
            let weight = 1.0 / (1.0 + dist_sq);
            acc[0] += force[0] * weight;
            acc[1] += force[1] * weight;
        }

        acc
    }

    /// With gravity pointing down
    pub fn with_gravity(mut self, g: f64) -> Self {
        self.gravity = [0.0, -g];
        self
    }

    /// With air resistance
    pub fn with_drag(mut self, d: f64) -> Self {
        self.drag = d;
        self
    }
}
