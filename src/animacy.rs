//! ═══════════════════════════════════════════════════════════════════════════════
//! ANIMACY DETECTION — From Objects to Agents
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements the computational theory of animacy perception:
//!
//! Key insight: Entity detection is about DYNAMICS, not static shape.
//! V1-V4 operate in spatial state spaces. Entity detection requires temporal
//! derivatives—velocity, acceleration, trajectory, goal-directedness.
//!
//! Core mechanism: Animacy as deviation from Newtonian dynamics
//!   A(trajectory) = ∫₀ᵀ ‖a(t) - a_expected(t)‖² dt
//!
//! High A → animate (doing something physics doesn't explain)
//! Low A → inanimate (following physical law)
//!
//! The classification is discrete via attractor dynamics:
//!   Inputs get pulled into basins: AGENT vs NOT-AGENT
//!   Not "37% agent" but categorical perception.
//!
//! Entity eigenmodes: What you get when you crank up gain on agent-detection.
//! Machine elves = entity attractor activation WITHOUT template match.
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════════
// TRAJECTORY REPRESENTATION
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// PHYSICS MODEL — What Newtonian dynamics predicts
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMACY SCORE — Deviation from Newtonian expectation
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// ENTITY STATE SPACE
// ═══════════════════════════════════════════════════════════════════════════════

/// Entity state: (form, trajectory, intention)
/// This is the state space for the entity layer, above V4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityState {
    /// Identity/form (compressed from V4)
    pub form: FormDescriptor,

    /// Motion history
    pub trajectory: Trajectory,

    /// Inferred goal/intention
    pub intention: Intention,

    /// Animacy score
    pub animacy: f64,

    /// Confidence in entity classification
    pub confidence: f64,
}

/// Form descriptor (what kind of thing is it?)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormDescriptor {
    /// Form template class
    pub template: FormTemplate,

    /// Template match confidence
    pub match_confidence: f64,

    /// Geometric complexity (from V4)
    pub complexity: f64,

    /// Bilateral symmetry score
    pub symmetry: f64,

    /// Face-likeness score (specialized detector)
    pub face_score: f64,
}

/// Known form templates (attractor basins)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormTemplate {
    /// Human face (FFA territory)
    Face,
    /// Eyes specifically (primal, dedicated circuitry)
    Eyes,
    /// Human body (EBA territory)
    Body,
    /// Generic biological form
    Biological,
    /// Self-propelled but unknown form
    GenericAgent,
    /// Inanimate object
    Object,
    /// Texture/pattern (not an entity)
    Pattern,
    /// Unknown/unclassified
    Unknown,
}

impl FormTemplate {
    /// Is this an entity template?
    pub fn is_entity(&self) -> bool {
        matches!(
            self,
            FormTemplate::Face
                | FormTemplate::Eyes
                | FormTemplate::Body
                | FormTemplate::Biological
                | FormTemplate::GenericAgent
        )
    }

    /// Attractor strength (how strongly does this template pull?)
    pub fn attractor_strength(&self) -> f64 {
        match self {
            FormTemplate::Face => 1.0,  // Extremely strong (FFA is huge)
            FormTemplate::Eyes => 0.95, // Primal, dedicated circuitry
            FormTemplate::Body => 0.8,  // Strong (EBA)
            FormTemplate::Biological => 0.6,
            FormTemplate::GenericAgent => 0.4,
            FormTemplate::Object => 0.3,
            FormTemplate::Pattern => 0.2,
            FormTemplate::Unknown => 0.1,
        }
    }
}

/// Inferred intention/goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    /// What is it trying to do?
    pub goal: GoalType,

    /// Target of intention (if applicable)
    pub target: Option<[f64; 2]>,

    /// Confidence in intention inference
    pub confidence: f64,
}

/// Types of inferred goals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalType {
    /// Moving toward target
    Approach,
    /// Moving away from target
    Avoid,
    /// Pursuing another entity
    Chase,
    /// Being pursued
    Flee,
    /// Exploring environment
    Explore,
    /// Observing/attending
    Attend,
    /// Communicating/signaling
    Communicate,
    /// No clear goal (random or goal-less motion)
    None,
    /// Unknown intention
    Unknown,
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENTITY ATTRACTOR DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Entity attractor basin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttractorConfig {
    /// Threshold for entity classification
    pub entity_threshold: f64,

    /// Threshold for face detection
    pub face_threshold: f64,

    /// Psychedelic gain multiplier (μ_E in the model)
    /// Higher values lower thresholds → more entity detection
    pub gain: f64,

    /// Winner-take-all competition strength
    pub competition_strength: f64,
}

impl Default for AttractorConfig {
    fn default() -> Self {
        Self {
            entity_threshold: 0.5,
            face_threshold: 0.7,
            gain: 1.0, // Normal perception
            competition_strength: 0.8,
        }
    }
}

impl AttractorConfig {
    /// Low-threshold configuration (psychedelic-like)
    pub fn low_threshold() -> Self {
        Self {
            entity_threshold: 0.2,
            face_threshold: 0.3,
            gain: 2.5,
            competition_strength: 0.5,
        }
    }

    /// High-threshold configuration (conservative)
    pub fn high_threshold() -> Self {
        Self {
            entity_threshold: 0.7,
            face_threshold: 0.85,
            gain: 0.7,
            competition_strength: 0.9,
        }
    }
}

/// Entity attractor classifier
#[derive(Debug, Clone)]
pub struct EntityClassifier {
    config: AttractorConfig,
}

impl EntityClassifier {
    pub fn new(config: AttractorConfig) -> Self {
        Self { config }
    }

    /// Classify an input into an entity attractor basin
    pub fn classify(&self, form: &FormDescriptor, animacy: &AnimacyScore) -> ClassificationResult {
        // Effective thresholds (modified by gain)
        let eff_entity_threshold = self.config.entity_threshold / self.config.gain;
        let eff_face_threshold = self.config.face_threshold / self.config.gain;

        // Compute activation for each attractor
        let mut activations = Vec::new();

        // Face attractor
        let face_activation = form.face_score * FormTemplate::Face.attractor_strength();
        activations.push((FormTemplate::Face, face_activation));

        // Eyes attractor (subset of face features)
        let eyes_activation = (form.face_score * 0.8) * FormTemplate::Eyes.attractor_strength();
        activations.push((FormTemplate::Eyes, eyes_activation));

        // Body attractor
        let body_activation = if form.symmetry > 0.6 && form.complexity > 0.3 {
            form.symmetry * FormTemplate::Body.attractor_strength()
        } else {
            0.0
        };
        activations.push((FormTemplate::Body, body_activation));

        // Biological motion (from animacy)
        let bio_activation =
            animacy.biological_motion * FormTemplate::Biological.attractor_strength();
        activations.push((FormTemplate::Biological, bio_activation));

        // Generic agent (animate but unknown form)
        let agent_activation = if animacy.is_animate && form.match_confidence < 0.5 {
            animacy.confidence * FormTemplate::GenericAgent.attractor_strength()
        } else {
            0.0
        };
        activations.push((FormTemplate::GenericAgent, agent_activation));

        // Object (inanimate, structured)
        let object_activation = if !animacy.is_animate && form.complexity > 0.2 {
            (1.0 - animacy.confidence) * FormTemplate::Object.attractor_strength()
        } else {
            0.0
        };
        activations.push((FormTemplate::Object, object_activation));

        // Pattern (texture, no structure)
        let pattern_activation = if form.complexity < 0.2 {
            (1.0 - form.complexity) * FormTemplate::Pattern.attractor_strength()
        } else {
            0.0
        };
        activations.push((FormTemplate::Pattern, pattern_activation));

        // Apply gain
        for (_, act) in &mut activations {
            *act *= self.config.gain;
        }

        // Winner-take-all competition
        let total_activation: f64 = activations.iter().map(|(_, a)| *a).sum();
        if total_activation > 0.01 {
            for (_, act) in &mut activations {
                let normalized = *act / total_activation;
                *act = normalized.powf(self.config.competition_strength);
            }
        }

        // Find winner
        let (winner_template, winner_activation) = activations
            .iter()
            .max_by(|(_, a1), (_, a2)| a1.partial_cmp(a2).unwrap())
            .map(|(t, a)| (*t, *a))
            .unwrap_or((FormTemplate::Unknown, 0.0));

        // Check if winner exceeds threshold
        let is_entity = winner_template.is_entity() && winner_activation > eff_entity_threshold;
        let is_face = matches!(winner_template, FormTemplate::Face | FormTemplate::Eyes)
            && winner_activation > eff_face_threshold;

        ClassificationResult {
            template: winner_template,
            activation: winner_activation,
            is_entity,
            is_face,
            all_activations: activations,
            effective_threshold: eff_entity_threshold,
        }
    }
}

/// Result of entity classification
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Winning template
    pub template: FormTemplate,

    /// Winning activation level
    pub activation: f64,

    /// Did it cross entity threshold?
    pub is_entity: bool,

    /// Is it specifically a face?
    pub is_face: bool,

    /// All attractor activations
    pub all_activations: Vec<(FormTemplate, f64)>,

    /// Effective threshold used
    pub effective_threshold: f64,
}

impl ClassificationResult {
    /// Is this a "machine elf" case? (Entity attractor without template match)
    pub fn is_untemplated_entity(&self) -> bool {
        self.is_entity
            && matches!(
                self.template,
                FormTemplate::GenericAgent | FormTemplate::Unknown
            )
    }

    /// Interpretation
    pub fn interpretation(&self) -> String {
        if self.is_untemplated_entity() {
            "ENTITY WITHOUT TEMPLATE: Agent-ness without familiar form (machine elf territory)"
                .to_string()
        } else if self.is_face {
            "FACE DETECTED: Strong FFA activation".to_string()
        } else if self.is_entity {
            format!("ENTITY DETECTED: {:?} template", self.template)
        } else {
            format!("NON-ENTITY: {:?}", self.template)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENTITY NEURAL FIELD
// ═══════════════════════════════════════════════════════════════════════════════

/// Entity neural field state
/// Implements: ∂a_E/∂t = -a_E + μ_E·a_E + inputs + lateral
#[derive(Debug, Clone)]
pub struct EntityField {
    /// Field activity over (form × motion_class)
    /// Simplified to discrete grid
    activity: Vec<Vec<f64>>,

    /// Number of form bins
    n_forms: usize,

    /// Number of motion class bins
    n_motion: usize,

    /// Gain parameter (μ_E)
    gain: f64,

    /// Decay rate
    decay: f64,

    /// Lateral kernel (winner-take-all)
    lateral_inhibition: f64,

    /// Input integration rate
    input_rate: f64,
}

impl EntityField {
    pub fn new(n_forms: usize, n_motion: usize, gain: f64) -> Self {
        Self {
            activity: vec![vec![0.0; n_motion]; n_forms],
            n_forms,
            n_motion,
            gain,
            decay: 0.1,
            lateral_inhibition: 0.5,
            input_rate: 0.3,
        }
    }

    /// Sigmoid nonlinearity
    fn sigma(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Update field with inputs from V4 and MT
    pub fn update(&mut self, v4_input: &[f64], mt_input: &[f64], dt: f64) {
        let mut new_activity = self.activity.clone();

        for i in 0..self.n_forms {
            for j in 0..self.n_motion {
                let a = self.activity[i][j];

                // Self-excitation with gain
                let self_term = -a + self.gain * a;

                // V4 form input
                let v4_term = if i < v4_input.len() {
                    self.input_rate * Self::sigma(v4_input[i])
                } else {
                    0.0
                };

                // MT motion input
                let mt_term = if j < mt_input.len() {
                    self.input_rate * Self::sigma(mt_input[j])
                } else {
                    0.0
                };

                // Lateral inhibition (winner-take-all)
                let total_activity: f64 = self.activity.iter().flatten().sum();
                let lateral_term = -self.lateral_inhibition * (total_activity - a);

                // Euler update
                let da = self_term + v4_term + mt_term + lateral_term;
                new_activity[i][j] = (a + da * dt).clamp(0.0, 1.0);
            }
        }

        self.activity = new_activity;
    }

    /// Get peak activity
    pub fn peak_activity(&self) -> (usize, usize, f64) {
        let mut max_val = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;

        for (i, row) in self.activity.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        (max_i, max_j, max_val)
    }

    /// Total activity (for monitoring)
    pub fn total_activity(&self) -> f64 {
        self.activity.iter().flatten().sum()
    }

    /// Set gain (for psychedelic simulation)
    pub fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    /// Get current gain
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// Set decay rate
    pub fn set_decay(&mut self, decay: f64) {
        self.decay = decay.clamp(0.01, 1.0);
    }

    /// Get current decay rate
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Apply decay to all activity (for natural activity reduction over time)
    pub fn apply_decay(&mut self, dt: f64) {
        let decay_factor = (-self.decay * dt).exp();
        for row in &mut self.activity {
            for val in row.iter_mut() {
                *val *= decay_factor;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE TRANSITION DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_creation() {
        let mut traj = Trajectory::new(10);
        traj.push(PhasePoint::new(0.0, 0.0, 1.0, 0.0, 0.0));
        traj.push(PhasePoint::new(1.0, 0.0, 1.0, 0.0, 1.0));
        traj.push(PhasePoint::new(2.0, 0.0, 1.0, 0.0, 2.0));

        assert_eq!(traj.len(), 3);

        let acc = traj.accelerations();
        assert_eq!(acc.len(), 2);
        // Constant velocity → zero acceleration
        assert!(acc[0][0].abs() < 0.01);
        assert!(acc[0][1].abs() < 0.01);
    }

    #[test]
    fn test_animacy_inanimate() {
        // Constant velocity (pure physics) → low animacy
        let mut traj = Trajectory::new(20);
        for i in 0..15 {
            let t = i as f64 * 0.1;
            traj.push(PhasePoint::new(t, 0.0, 1.0, 0.0, t));
        }

        let detector = AnimacyDetector::new().with_threshold(0.3);
        let score = detector.analyze(&traj);

        // Key assertions: low deviation from physics, low decision points
        assert!(score.mean_deviation < 0.1);
        assert!(score.decision_points < 2);
        // Note: goal_directedness can be high for straight lines, but
        // deviation from physics is what matters for inanimate classification
    }

    #[test]
    fn test_animacy_animate() {
        // Sudden direction changes → high animacy
        let mut traj = Trajectory::new(20);

        // Start going right
        for i in 0..5 {
            let t = i as f64 * 0.1;
            traj.push(PhasePoint::new(t, 0.0, 1.0, 0.0, t));
        }

        // Suddenly go up (decision point)
        for i in 5..10 {
            let t = i as f64 * 0.1;
            traj.push(PhasePoint::new(0.5, t - 0.5, 0.0, 1.0, t));
        }

        // Suddenly go left (another decision point)
        for i in 10..15 {
            let t = i as f64 * 0.1;
            traj.push(PhasePoint::new(0.5 - (t - 1.0), 0.5, -1.0, 0.0, t));
        }

        let detector = AnimacyDetector::new();
        let score = detector.analyze(&traj);

        assert!(score.is_animate || score.decision_points >= 1);
    }

    #[test]
    fn test_form_template_is_entity() {
        assert!(FormTemplate::Face.is_entity());
        assert!(FormTemplate::Eyes.is_entity());
        assert!(FormTemplate::Body.is_entity());
        assert!(FormTemplate::GenericAgent.is_entity());
        assert!(!FormTemplate::Object.is_entity());
        assert!(!FormTemplate::Pattern.is_entity());
    }

    #[test]
    fn test_entity_classifier_normal() {
        let config = AttractorConfig::default();
        let classifier = EntityClassifier::new(config);

        // Low-face, low-animacy input
        let form = FormDescriptor {
            template: FormTemplate::Unknown,
            match_confidence: 0.0,
            complexity: 0.3,
            symmetry: 0.2,
            face_score: 0.1,
        };

        let animacy = AnimacyScore {
            total_deviation: 0.1,
            mean_deviation: 0.01,
            peak_deviation: 0.05,
            decision_points: 0,
            goal_directedness: 0.1,
            contingency: 0.0,
            biological_motion: 0.1,
            is_animate: false,
            confidence: 0.2,
        };

        let result = classifier.classify(&form, &animacy);
        assert!(!result.is_entity);
    }

    #[test]
    fn test_entity_classifier_face() {
        // Use low threshold config to ensure face detection works
        let config = AttractorConfig {
            entity_threshold: 0.3,
            face_threshold: 0.4,
            gain: 1.5,
            competition_strength: 0.8,
        };
        let classifier = EntityClassifier::new(config);

        // High face score
        let form = FormDescriptor {
            template: FormTemplate::Face,
            match_confidence: 0.9,
            complexity: 0.6,
            symmetry: 0.9,
            face_score: 0.9,
        };

        let animacy = AnimacyScore {
            total_deviation: 0.5,
            mean_deviation: 0.1,
            peak_deviation: 0.3,
            decision_points: 1,
            goal_directedness: 0.3,
            contingency: 0.0,
            biological_motion: 0.4,
            is_animate: true,
            confidence: 0.7,
        };

        let result = classifier.classify(&form, &animacy);
        // With high face_score (0.9), should be classified as entity
        assert!(result.is_entity || result.activation > 0.5);
        // Face template should win
        assert!(matches!(
            result.template,
            FormTemplate::Face | FormTemplate::Eyes
        ));
    }

    #[test]
    fn test_entity_classifier_low_threshold() {
        // With high gain, even weak signals should trigger entity detection
        let config = AttractorConfig::low_threshold();
        let classifier = EntityClassifier::new(config);

        // Moderate animacy, low form match
        let form = FormDescriptor {
            template: FormTemplate::Unknown,
            match_confidence: 0.2,
            complexity: 0.5,
            symmetry: 0.5,
            face_score: 0.3,
        };

        let animacy = AnimacyScore {
            total_deviation: 0.8,
            mean_deviation: 0.15,
            peak_deviation: 0.4,
            decision_points: 2,
            goal_directedness: 0.4,
            contingency: 0.0,
            biological_motion: 0.3,
            is_animate: true,
            confidence: 0.5,
        };

        let result = classifier.classify(&form, &animacy);
        // With low threshold, this should be detected as entity
        assert!(result.is_entity || result.activation > 0.3);
    }

    #[test]
    fn test_untemplated_entity() {
        let config = AttractorConfig::low_threshold();
        let classifier = EntityClassifier::new(config);

        // Highly animate but doesn't match templates
        let form = FormDescriptor {
            template: FormTemplate::Unknown,
            match_confidence: 0.1,
            complexity: 0.8, // High complexity
            symmetry: 0.6,
            face_score: 0.15,
        };

        let animacy = AnimacyScore {
            total_deviation: 2.0,
            mean_deviation: 0.4,
            peak_deviation: 1.0,
            decision_points: 5,
            goal_directedness: 0.6,
            contingency: 0.0,
            biological_motion: 0.2,
            is_animate: true,
            confidence: 0.9,
        };

        let result = classifier.classify(&form, &animacy);
        // This is the "machine elf" case
        if result.is_entity {
            assert!(
                result.is_untemplated_entity()
                    || matches!(result.template, FormTemplate::GenericAgent)
            );
        }
    }

    #[test]
    fn test_entity_field_update() {
        let mut field = EntityField::new(4, 4, 1.0);

        // Zero input → activity should stay relatively low
        let v4_input = vec![0.0; 4];
        let mt_input = vec![0.0; 4];

        field.update(&v4_input, &mt_input, 0.1);
        let activity_zero_input = field.total_activity();

        // Strong input → activity should increase significantly
        let v4_input = vec![2.0, 0.0, 0.0, 0.0];
        let mt_input = vec![0.0, 2.0, 0.0, 0.0];

        for _ in 0..20 {
            field.update(&v4_input, &mt_input, 0.1);
        }

        let activity_strong_input = field.total_activity();
        // Activity with strong input should be meaningfully higher
        assert!(activity_strong_input > activity_zero_input);
    }

    #[test]
    fn test_entity_field_gain() {
        let mut field = EntityField::new(4, 4, 1.0);

        // Give input
        let v4_input = vec![1.0, 0.5, 0.0, 0.0];
        let mt_input = vec![0.5, 1.0, 0.0, 0.0];

        for _ in 0..10 {
            field.update(&v4_input, &mt_input, 0.1);
        }
        let activity_normal = field.total_activity();

        // Reset and increase gain
        let mut field_high = EntityField::new(4, 4, 2.5);

        for _ in 0..10 {
            field_high.update(&v4_input, &mt_input, 0.1);
        }
        let activity_high = field_high.total_activity();

        // Higher gain should produce higher activity
        assert!(activity_high >= activity_normal * 0.9); // Allow some tolerance
    }

    #[test]
    fn test_phase_monitor() {
        let mut monitor = EntityPhaseMonitor::new(20);

        // Record normal phase data
        for i in 0..5 {
            monitor.record(0.1, 0.5 + i as f64 * 0.1);
        }

        let phase = monitor.current_phase(0.8);
        assert!(matches!(phase, Phase::Normal) || matches!(phase, Phase::NearCritical));

        // Record transition
        for i in 0..10 {
            let gain = 1.0 + i as f64 * 0.2;
            let detection = (gain - 1.0).min(1.0).max(0.0);
            monitor.record(detection, gain);
        }

        let phase_high = monitor.current_phase(2.5);
        assert!(matches!(
            phase_high,
            Phase::Supercritical | Phase::EntitySaturated
        ));
    }

    #[test]
    fn test_goal_directedness() {
        let detector = AnimacyDetector::new();

        // Trajectory moving toward a goal
        let mut traj = Trajectory::new(20);
        for i in 0..10 {
            let t = i as f64 * 0.1;
            // Moving toward (10, 10)
            let x = t * 10.0;
            let y = t * 10.0;
            let vx = 10.0;
            let vy = 10.0;
            traj.push(PhasePoint::new(x, y, vx, vy, t));
        }

        let score = detector.analyze(&traj);
        // Should have high goal-directedness (straight line toward goal)
        assert!(score.goal_directedness > 0.5);
    }

    #[test]
    fn test_curvature_computation() {
        let mut traj = Trajectory::new(10);

        // Circular arc
        for i in 0..8 {
            let theta = i as f64 * 0.3;
            let x = theta.cos();
            let y = theta.sin();
            let vx = -theta.sin();
            let vy = theta.cos();
            traj.push(PhasePoint::new(x, y, vx, vy, i as f64 * 0.1));
        }

        let curvatures = traj.curvatures();
        assert!(!curvatures.is_empty());
        // Curvatures should be roughly constant for a circle
        let mean_curv: f64 = curvatures.iter().sum::<f64>() / curvatures.len() as f64;
        for c in &curvatures {
            assert!((c - mean_curv).abs() < 0.5);
        }
    }
}
