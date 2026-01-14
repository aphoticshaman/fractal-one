//! Trajectory representation in phase space.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A point in phase space: position + velocity
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhasePoint {
    /// Position (can be abstract, not just spatial)
    pub position: [f64; 2],
    /// Velocity (first derivative)
    pub velocity: [f64; 2],
    /// Timestamp
    pub t: f64,
}

impl PhasePoint {
    pub fn new(x: f64, y: f64, vx: f64, vy: f64, t: f64) -> Self {
        Self {
            position: [x, y],
            velocity: [vx, vy],
            t,
        }
    }

    /// Compute acceleration from this point to the next
    pub fn acceleration_to(&self, next: &PhasePoint) -> [f64; 2] {
        let dt = next.t - self.t;
        if dt <= 0.0 {
            return [0.0, 0.0];
        }
        [
            (next.velocity[0] - self.velocity[0]) / dt,
            (next.velocity[1] - self.velocity[1]) / dt,
        ]
    }

    /// Speed magnitude
    pub fn speed(&self) -> f64 {
        (self.velocity[0].powi(2) + self.velocity[1].powi(2)).sqrt()
    }
}

/// A trajectory: sequence of phase points over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub points: VecDeque<PhasePoint>,
    pub max_points: usize,
}

impl Trajectory {
    pub fn new(max_points: usize) -> Self {
        Self {
            points: VecDeque::with_capacity(max_points),
            max_points,
        }
    }

    pub fn push(&mut self, point: PhasePoint) {
        if self.points.len() >= self.max_points {
            self.points.pop_front();
        }
        self.points.push_back(point);
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Extract acceleration sequence
    pub fn accelerations(&self) -> Vec<[f64; 2]> {
        if self.points.len() < 2 {
            return vec![];
        }
        self.points
            .iter()
            .zip(self.points.iter().skip(1))
            .map(|(p1, p2)| p1.acceleration_to(p2))
            .collect()
    }

    /// Compute trajectory curvature (rate of direction change)
    pub fn curvatures(&self) -> Vec<f64> {
        if self.points.len() < 3 {
            return vec![];
        }

        let mut curvatures = Vec::new();
        for i in 1..self.points.len() - 1 {
            let p0 = &self.points[i - 1];
            let p1 = &self.points[i];
            let p2 = &self.points[i + 1];

            // Direction vectors
            let d1 = [
                p1.position[0] - p0.position[0],
                p1.position[1] - p0.position[1],
            ];
            let d2 = [
                p2.position[0] - p1.position[0],
                p2.position[1] - p1.position[1],
            ];

            let len1 = (d1[0].powi(2) + d1[1].powi(2)).sqrt();
            let len2 = (d2[0].powi(2) + d2[1].powi(2)).sqrt();

            if len1 < 1e-10 || len2 < 1e-10 {
                curvatures.push(0.0);
                continue;
            }

            // Angle between directions
            let dot = d1[0] * d2[0] + d1[1] * d2[1];
            let cos_theta = (dot / (len1 * len2)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();

            // Curvature = angle change per arc length
            let arc_len = (len1 + len2) / 2.0;
            curvatures.push(theta / arc_len.max(1e-10));
        }
        curvatures
    }
}
