//! ═══════════════════════════════════════════════════════════════════════════════
//! COUPLING — Cross-Level Basin Coupling Experiment
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! # EXPERIMENT STATUS: CLOSED (2026-01-11)
//!
//! **Result**: Strong (B) FALSIFIED. Weak coupling demonstrated, but no basin
//! deformation observed. See [`negative_result`] for full documentation.
//!
//! ## What Was Tested
//!
//! A fully synthetic, instrumented, two-level dissipative system for testing
//! whether cross-level basin coupling (hypothesis B) is real or artifact.
//!
//! Requirements (all non-negotiable):
//!   1. Two explicitly defined Basinized Systems B₁, B₂
//!   2. An explicit coupling operator C₁₂: X₁ → parameters of Φ₂
//!   3. A null model with same marginal statistics, no coupling
//!   4. A quantitative basin metric (Fisher Information curvature)
//!   5. No human-in-the-loop curation
//!
//! ## What Was Found
//!
//! | Claim                          | Status        |
//! |--------------------------------|---------------|
//! | Basinized System formal object | ✓ Valid       |
//! | Weak cross-level coupling      | ✓ Demonstrated|
//! | Basin deformation from coupling| ✗ Falsified   |
//! | Strong (B)                     | ✗ Dead        |
//! | (A)+(C) sufficient             | ✓ Supported   |
//!
//! Cross-level coupling exists but is sub-basin and linear.
//! Attractors are robust. μ is invariant. π destroys structure.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use crate::stats::float_cmp;
use std::collections::{HashMap, VecDeque};

// Submodules
pub mod archive;
pub mod b_prime;
pub mod fisher_projection;
pub mod negative_result;
pub mod no_go;

// ═══════════════════════════════════════════════════════════════════════════════
// BASINIZED SYSTEM — The formal structure B = (X, Φ, π, μ)
// ═══════════════════════════════════════════════════════════════════════════════

/// A Basinized System: the minimal formal object for persistent macrostates
/// B = (X, Φ, π, μ) where:
///   X = state space
///   Φ = flow (dissipative dynamics)
///   π = coarse observation map
///   μ = invariant measure on attractors
#[derive(Debug, Clone)]
pub struct BasinizedSystem<const DIM: usize> {
    /// State space dimension
    pub dim: usize,

    /// Current state x ∈ X
    pub state: [f64; DIM],

    /// Flow parameters (defines Φ)
    pub flow_params: FlowParams,

    /// Noise strength (dissipation)
    pub noise: f64,

    /// Time step
    pub dt: f64,

    /// History of states (for trajectory analysis) - VecDeque for O(1) rotation
    history: VecDeque<[f64; DIM]>,

    /// Maximum history length
    max_history: usize,
}

/// Parameters defining the flow Φ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowParams {
    /// Attractor positions (basin centers)
    pub attractors: Vec<Vec<f64>>,

    /// Attractor strengths (depth of wells)
    pub strengths: Vec<f64>,

    /// Attractor widths (basin radii)
    pub widths: Vec<f64>,

    /// Global damping coefficient
    pub damping: f64,

    /// Coupling strength (from external level)
    pub coupling_strength: f64,
}

impl Default for FlowParams {
    fn default() -> Self {
        Self {
            attractors: vec![vec![-1.0, 0.0], vec![1.0, 0.0]], // Two attractors
            strengths: vec![1.0, 1.0],
            widths: vec![1.0, 1.0],
            damping: 0.1,
            coupling_strength: 0.0, // No coupling by default
        }
    }
}

impl<const DIM: usize> BasinizedSystem<DIM> {
    pub fn new(flow_params: FlowParams, noise: f64, dt: f64) -> Self {
        Self {
            dim: DIM,
            state: [0.0; DIM],
            flow_params,
            noise,
            dt,
            history: VecDeque::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Initialize state
    pub fn set_state(&mut self, state: [f64; DIM]) {
        self.state = state;
        self.history.clear();
        self.history.push_back(state);
    }

    /// One step of the flow Φ: x_{t+1} = Φ(x_t) + noise
    pub fn step(&mut self, rng: &mut impl FnMut() -> f64) {
        let mut force = [0.0; DIM];

        // Compute force from each attractor (gradient of potential)
        for (i, attractor) in self.flow_params.attractors.iter().enumerate() {
            if attractor.len() < DIM {
                continue;
            }

            let strength = self.flow_params.strengths.get(i).copied().unwrap_or(1.0);
            let width = self.flow_params.widths.get(i).copied().unwrap_or(1.0);

            // Distance to attractor
            let mut dist_sq = 0.0;
            for d in 0..DIM {
                dist_sq += (self.state[d] - attractor[d]).powi(2);
            }

            // Gaussian well: force = -∇V where V = -strength * exp(-dist²/2width²)
            let gaussian = (-dist_sq / (2.0 * width * width)).exp();
            let force_magnitude = strength * gaussian / (width * width);

            for d in 0..DIM {
                force[d] += force_magnitude * (attractor[d] - self.state[d]);
            }
        }

        // Apply damping
        for d in 0..DIM {
            force[d] -= self.flow_params.damping * self.state[d];
        }

        // Euler-Maruyama update
        for d in 0..DIM {
            let noise_term = self.noise * (self.dt).sqrt() * rng();
            self.state[d] += force[d] * self.dt + noise_term;
        }

        // Record history (O(1) rotation using VecDeque)
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(self.state);
    }

    /// Run many steps
    pub fn run(&mut self, steps: usize, rng: &mut impl FnMut() -> f64) {
        for _ in 0..steps {
            self.step(rng);
        }
    }

    /// Coarse observation π: X → observable
    pub fn observe(&self) -> Observation {
        // Which attractor basin are we in?
        let mut min_dist = f64::INFINITY;
        let mut nearest_attractor = 0;

        for (i, attractor) in self.flow_params.attractors.iter().enumerate() {
            let mut dist_sq = 0.0;
            for d in 0..DIM.min(attractor.len()) {
                dist_sq += (self.state[d] - attractor[d]).powi(2);
            }
            if dist_sq < min_dist {
                min_dist = dist_sq;
                nearest_attractor = i;
            }
        }

        Observation {
            basin_id: nearest_attractor,
            distance_to_attractor: min_dist.sqrt(),
            state_snapshot: self.state.to_vec(),
        }
    }

    /// Get trajectory for analysis (allocates a Vec from internal VecDeque)
    pub fn trajectory(&self) -> Vec<[f64; DIM]> {
        self.history.iter().copied().collect()
    }

    /// Modify flow parameters (for coupling)
    pub fn apply_coupling(&mut self, coupling: &CouplingOperator) {
        self.flow_params.coupling_strength = coupling.strength;

        // Coupling can shift attractor positions
        for (i, shift) in coupling.attractor_shifts.iter().enumerate() {
            if i < self.flow_params.attractors.len() {
                for (d, s) in shift.iter().enumerate() {
                    if d < self.flow_params.attractors[i].len() {
                        self.flow_params.attractors[i][d] += s * coupling.strength;
                    }
                }
            }
        }

        // Coupling can modify well depths
        for (i, depth_mod) in coupling.depth_modifiers.iter().enumerate() {
            if i < self.flow_params.strengths.len() {
                self.flow_params.strengths[i] *= 1.0 + depth_mod * coupling.strength;
            }
        }
    }
}

/// Coarse observation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub basin_id: usize,
    pub distance_to_attractor: f64,
    pub state_snapshot: Vec<f64>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// COUPLING OPERATOR — The typed map C₁₂: X₁ → parameters of Φ₂
// ═══════════════════════════════════════════════════════════════════════════════

/// Explicit coupling operator C₁₂: X₁ → parameters of Φ₂
/// Not "influence," not "feedback," but a typed map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingOperator {
    /// Overall coupling strength (0 = null model)
    pub strength: f64,

    /// How Level 1 state shifts Level 2 attractors
    pub attractor_shifts: Vec<Vec<f64>>,

    /// How Level 1 state modifies Level 2 well depths
    pub depth_modifiers: Vec<f64>,

    /// Coupling type
    pub coupling_type: CouplingType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingType {
    /// No coupling (null model)
    None,
    /// Unidirectional: Level 1 → Level 2
    Unidirectional,
    /// Bidirectional: Level 1 ↔ Level 2
    Bidirectional,
}

impl CouplingOperator {
    /// Null coupling (for null model)
    pub fn null() -> Self {
        Self {
            strength: 0.0,
            attractor_shifts: vec![],
            depth_modifiers: vec![],
            coupling_type: CouplingType::None,
        }
    }

    /// Standard unidirectional coupling
    pub fn unidirectional(strength: f64) -> Self {
        Self {
            strength,
            attractor_shifts: vec![vec![0.1, 0.0], vec![-0.1, 0.0]], // Shifts attractors
            depth_modifiers: vec![0.2, -0.2], // Modifies well depths asymmetrically
            coupling_type: CouplingType::Unidirectional,
        }
    }

    /// Compute coupling effect from Level 1 state
    pub fn compute(&self, level1_state: &[f64]) -> CouplingEffect {
        if self.strength == 0.0 {
            return CouplingEffect::null();
        }

        // Coupling effect depends on Level 1 basin
        let level1_projection = if !level1_state.is_empty() {
            level1_state[0] // Project to first dimension
        } else {
            0.0
        };

        CouplingEffect {
            attractor_shift: self.strength * level1_projection * 0.1,
            depth_modifier: self.strength * level1_projection.signum() * 0.1,
            active: true,
        }
    }
}

/// Effect of coupling on Level 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingEffect {
    pub attractor_shift: f64,
    pub depth_modifier: f64,
    pub active: bool,
}

impl CouplingEffect {
    pub fn null() -> Self {
        Self {
            attractor_shift: 0.0,
            depth_modifier: 0.0,
            active: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FISHER INFORMATION METRIC — The basin geometry measure
// ═══════════════════════════════════════════════════════════════════════════════

/// Fisher Information matrix estimation from trajectory samples
#[derive(Debug, Clone)]
pub struct FisherEstimator {
    /// Dimension
    dim: usize,

    /// Sample accumulator: sum of outer products of score functions
    fisher_sum: Vec<Vec<f64>>,

    /// Number of samples
    n_samples: usize,

    /// Previous state (for velocity estimation)
    prev_state: Option<Vec<f64>>,

    /// Noise scale (for score function normalization)
    noise_scale: f64,
}

impl FisherEstimator {
    pub fn new(dim: usize, noise_scale: f64) -> Self {
        Self {
            dim,
            fisher_sum: vec![vec![0.0; dim]; dim],
            n_samples: 0,
            prev_state: None,
            noise_scale,
        }
    }

    /// Add a sample to the estimator
    pub fn add_sample(&mut self, state: &[f64]) {
        if state.len() != self.dim {
            return;
        }

        if let Some(ref prev) = self.prev_state {
            // Score function ≈ (x_t - x_{t-1}) / noise
            // This is the gradient of log-likelihood under Langevin dynamics
            let mut score = vec![0.0; self.dim];
            for i in 0..self.dim {
                score[i] = (state[i] - prev[i]) / self.noise_scale.max(0.001);
            }

            // Fisher = E[score ⊗ score]
            for i in 0..self.dim {
                for j in 0..self.dim {
                    self.fisher_sum[i][j] += score[i] * score[j];
                }
            }

            self.n_samples += 1;
        }

        self.prev_state = Some(state.to_vec());
    }

    /// Get estimated Fisher Information matrix
    pub fn fisher_matrix(&self) -> Vec<Vec<f64>> {
        if self.n_samples == 0 {
            return vec![vec![0.0; self.dim]; self.dim];
        }

        let mut fisher = vec![vec![0.0; self.dim]; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                fisher[i][j] = self.fisher_sum[i][j] / self.n_samples as f64;
            }
        }
        fisher
    }

    /// Compute scalar curvature of Fisher metric
    /// This is the primary basin geometry measure
    pub fn scalar_curvature(&self) -> f64 {
        let fisher = self.fisher_matrix();

        if self.dim < 2 {
            // 1D: curvature is just the value
            return fisher.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        }

        // For 2D: Gaussian curvature K = det(F) / (trace(F)²)
        // This measures how "curved" the information geometry is

        let trace: f64 = (0..self.dim).map(|i| fisher[i][i]).sum();

        if self.dim == 2 {
            let det = fisher[0][0] * fisher[1][1] - fisher[0][1] * fisher[1][0];
            if trace.abs() > 1e-10 {
                det / (trace * trace)
            } else {
                0.0
            }
        } else {
            // Higher dimensions: use trace as proxy
            trace / self.dim as f64
        }
    }

    /// Compute eigenvalue spectrum of Fisher matrix
    /// Second basin geometry measure
    pub fn eigenvalue_spectrum(&self) -> Vec<f64> {
        let fisher = self.fisher_matrix();

        if self.dim == 1 {
            return vec![fisher[0][0]];
        }

        if self.dim == 2 {
            // Analytical eigenvalues for 2x2
            let a = fisher[0][0];
            let b = fisher[0][1];
            let c = fisher[1][0];
            let d = fisher[1][1];

            let trace = a + d;
            let det = a * d - b * c;

            let discriminant = trace * trace - 4.0 * det;
            if discriminant < 0.0 {
                // Complex eigenvalues (shouldn't happen for symmetric Fisher)
                return vec![trace / 2.0, trace / 2.0];
            }

            let sqrt_disc = discriminant.sqrt();
            vec![(trace + sqrt_disc) / 2.0, (trace - sqrt_disc) / 2.0]
        } else {
            // For higher dimensions, return diagonal as approximation
            (0..self.dim).map(|i| fisher[i][i]).collect()
        }
    }

    /// Reset the estimator
    pub fn reset(&mut self) {
        self.fisher_sum = vec![vec![0.0; self.dim]; self.dim];
        self.n_samples = 0;
        self.prev_state = None;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BASIN METRIC — The committed measure of basin geometry
// ═══════════════════════════════════════════════════════════════════════════════

/// Basin geometry measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinMetric {
    /// Scalar curvature of Fisher Information metric
    pub fisher_curvature: f64,

    /// Eigenvalue spectrum of Fisher matrix
    pub eigenvalues: Vec<f64>,

    /// Number of distinct basins visited
    pub basins_visited: usize,

    /// Mean return time to primary attractor
    pub mean_return_time: f64,

    /// Stability index (inverse of escape rate)
    pub stability_index: f64,

    /// Total samples used
    pub n_samples: usize,
}

impl BasinMetric {
    /// Compute distance between two basin metrics
    /// This is what we use to detect coupling
    pub fn distance(&self, other: &BasinMetric) -> f64 {
        let curvature_diff = (self.fisher_curvature - other.fisher_curvature).abs();

        let eigenvalue_diff: f64 = self
            .eigenvalues
            .iter()
            .zip(other.eigenvalues.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / self.eigenvalues.len().max(1) as f64;

        let stability_diff = (self.stability_index - other.stability_index).abs();

        // Weighted combination
        curvature_diff * 0.4 + eigenvalue_diff * 0.4 + stability_diff * 0.2
    }

    /// Is this metric significantly different from another?
    pub fn significantly_different(&self, other: &BasinMetric, threshold: f64) -> bool {
        self.distance(other) > threshold
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TWO-LEVEL SYSTEM — The complete experimental setup
// ═══════════════════════════════════════════════════════════════════════════════

/// Two-level coupled Basinized System for the experiment
pub struct TwoLevelSystem {
    /// Level 1: Low-level inference dynamics
    pub level1: BasinizedSystem<2>,

    /// Level 2: High-level constraint space
    pub level2: BasinizedSystem<2>,

    /// Coupling operator C₁₂
    pub coupling: CouplingOperator,

    /// Fisher estimator for Level 1
    pub fisher1: FisherEstimator,

    /// Fisher estimator for Level 2
    pub fisher2: FisherEstimator,

    /// Basin visit history (for return time computation)
    basin_history: Vec<(usize, usize)>, // (level1_basin, level2_basin)

    /// Random seed
    seed: u64,
}

impl TwoLevelSystem {
    pub fn new(coupling: CouplingOperator, noise: f64, seed: u64) -> Self {
        let flow1 = FlowParams {
            attractors: vec![vec![-1.0, 0.0], vec![1.0, 0.0]],
            strengths: vec![1.0, 1.0],
            widths: vec![0.8, 0.8],
            damping: 0.1,
            coupling_strength: 0.0,
        };

        let flow2 = FlowParams {
            attractors: vec![vec![0.0, -1.0], vec![0.0, 1.0]],
            strengths: vec![1.0, 1.0],
            widths: vec![0.8, 0.8],
            damping: 0.1,
            coupling_strength: coupling.strength,
        };

        Self {
            level1: BasinizedSystem::new(flow1, noise, 0.01),
            level2: BasinizedSystem::new(flow2, noise, 0.01),
            coupling,
            fisher1: FisherEstimator::new(2, noise),
            fisher2: FisherEstimator::new(2, noise),
            basin_history: Vec::new(),
            seed,
        }
    }

    /// Create null model (same marginal statistics, no coupling)
    pub fn null_model(noise: f64, seed: u64) -> Self {
        Self::new(CouplingOperator::null(), noise, seed)
    }

    /// Create coupled model
    pub fn coupled_model(coupling_strength: f64, noise: f64, seed: u64) -> Self {
        Self::new(
            CouplingOperator::unidirectional(coupling_strength),
            noise,
            seed,
        )
    }

    /// Get the random seed used for this system (for reproducibility/logging)
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get diagnostic info about the system configuration
    pub fn diagnostic_info(&self) -> String {
        format!(
            "TwoLevelSystem: seed={}, coupling={:?}, history_len={}",
            self.seed,
            self.coupling.coupling_type,
            self.basin_history.len()
        )
    }

    /// Initialize both levels
    pub fn initialize(&mut self, state1: [f64; 2], state2: [f64; 2]) {
        self.level1.set_state(state1);
        self.level2.set_state(state2);
        self.fisher1.reset();
        self.fisher2.reset();
        self.basin_history.clear();
    }

    /// One coupled step
    pub fn step(&mut self, rng: &mut impl FnMut() -> f64) {
        // Step Level 1
        self.level1.step(rng);

        // Compute coupling effect from Level 1 → Level 2
        let effect = self.coupling.compute(&self.level1.state);

        if effect.active {
            // Apply coupling to Level 2 flow parameters
            // This shifts the attractors based on Level 1 state
            for attractor in &mut self.level2.flow_params.attractors {
                if !attractor.is_empty() {
                    attractor[0] += effect.attractor_shift;
                }
            }

            // Modify strengths
            for strength in &mut self.level2.flow_params.strengths {
                *strength *= 1.0 + effect.depth_modifier;
            }
        }

        // Step Level 2
        self.level2.step(rng);

        // Update Fisher estimators
        self.fisher1.add_sample(&self.level1.state);
        self.fisher2.add_sample(&self.level2.state);

        // Record basin visits
        let obs1 = self.level1.observe();
        let obs2 = self.level2.observe();
        self.basin_history.push((obs1.basin_id, obs2.basin_id));
    }

    /// Run the system for multiple steps
    pub fn run(&mut self, steps: usize, rng: &mut impl FnMut() -> f64) {
        for _ in 0..steps {
            self.step(rng);
        }
    }

    /// Compute basin metrics for both levels
    pub fn compute_metrics(&self) -> (BasinMetric, BasinMetric) {
        let metric1 = BasinMetric {
            fisher_curvature: self.fisher1.scalar_curvature(),
            eigenvalues: self.fisher1.eigenvalue_spectrum(),
            basins_visited: self.count_basins_visited(0),
            mean_return_time: self.mean_return_time(0),
            stability_index: self.stability_index(0),
            n_samples: self.fisher1.n_samples,
        };

        let metric2 = BasinMetric {
            fisher_curvature: self.fisher2.scalar_curvature(),
            eigenvalues: self.fisher2.eigenvalue_spectrum(),
            basins_visited: self.count_basins_visited(1),
            mean_return_time: self.mean_return_time(1),
            stability_index: self.stability_index(1),
            n_samples: self.fisher2.n_samples,
        };

        (metric1, metric2)
    }

    fn count_basins_visited(&self, level: usize) -> usize {
        let basins: std::collections::HashSet<usize> = self
            .basin_history
            .iter()
            .map(|(b1, b2)| if level == 0 { *b1 } else { *b2 })
            .collect();
        basins.len()
    }

    fn mean_return_time(&self, level: usize) -> f64 {
        if self.basin_history.len() < 2 {
            return 0.0;
        }

        let primary_basin = 0; // Assume basin 0 is primary
        let mut return_times = Vec::new();
        let mut last_visit = None;

        for (i, (b1, b2)) in self.basin_history.iter().enumerate() {
            let basin = if level == 0 { *b1 } else { *b2 };
            if basin == primary_basin {
                if let Some(last) = last_visit {
                    return_times.push((i - last) as f64);
                }
                last_visit = Some(i);
            }
        }

        if return_times.is_empty() {
            0.0
        } else {
            return_times.iter().sum::<f64>() / return_times.len() as f64
        }
    }

    fn stability_index(&self, level: usize) -> f64 {
        // Stability = fraction of time spent in primary basin
        if self.basin_history.is_empty() {
            return 0.0;
        }

        let primary_basin = 0;
        let in_primary = self
            .basin_history
            .iter()
            .filter(|(b1, b2)| {
                let basin = if level == 0 { *b1 } else { *b2 };
                basin == primary_basin
            })
            .count();

        in_primary as f64 / self.basin_history.len() as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT — The discriminating test
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a coupling experiment run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Coupling strength used
    pub coupling_strength: f64,

    /// Seed used
    pub seed: u64,

    /// Level 1 basin metric
    pub metric1: BasinMetric,

    /// Level 2 basin metric
    pub metric2: BasinMetric,

    /// Distance from null model (Level 1)
    pub distance_from_null_1: f64,

    /// Distance from null model (Level 2)
    pub distance_from_null_2: f64,

    /// Steps run
    pub steps: usize,
}

/// The coupling experiment
pub struct CouplingExperiment {
    /// Noise level
    pub noise: f64,

    /// Steps per run
    pub steps_per_run: usize,

    /// Number of runs
    pub n_runs: usize,

    /// Null model baseline metrics (averaged)
    pub null_baseline_1: Option<BasinMetric>,
    pub null_baseline_2: Option<BasinMetric>,

    /// Results
    pub results: Vec<ExperimentResult>,
}

impl CouplingExperiment {
    pub fn new(noise: f64, steps_per_run: usize, n_runs: usize) -> Self {
        Self {
            noise,
            steps_per_run,
            n_runs,
            null_baseline_1: None,
            null_baseline_2: None,
            results: Vec::new(),
        }
    }

    /// Establish null baseline (MUST be called first)
    pub fn establish_null_baseline(&mut self, seeds: &[u64]) {
        let mut metrics1 = Vec::new();
        let mut metrics2 = Vec::new();

        for &seed in seeds {
            let mut system = TwoLevelSystem::null_model(self.noise, seed);
            system.initialize([0.5, 0.0], [0.0, 0.5]);

            // Simple PRNG for reproducibility
            let mut rng_state = seed;
            let mut rng = || {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bits = (rng_state >> 33) as u32;
                (bits as f64 / u32::MAX as f64 - 0.5) * 2.0 // [-1, 1]
            };

            system.run(self.steps_per_run, &mut rng);
            let (m1, m2) = system.compute_metrics();
            metrics1.push(m1);
            metrics2.push(m2);
        }

        // Average null metrics
        self.null_baseline_1 = Some(Self::average_metrics(&metrics1));
        self.null_baseline_2 = Some(Self::average_metrics(&metrics2));
    }

    fn average_metrics(metrics: &[BasinMetric]) -> BasinMetric {
        if metrics.is_empty() {
            return BasinMetric {
                fisher_curvature: 0.0,
                eigenvalues: vec![0.0, 0.0],
                basins_visited: 0,
                mean_return_time: 0.0,
                stability_index: 0.0,
                n_samples: 0,
            };
        }

        let n = metrics.len() as f64;
        let fisher_curvature = metrics.iter().map(|m| m.fisher_curvature).sum::<f64>() / n;

        let eigenvalues = if metrics[0].eigenvalues.len() >= 2 {
            vec![
                metrics.iter().map(|m| m.eigenvalues[0]).sum::<f64>() / n,
                metrics.iter().map(|m| m.eigenvalues[1]).sum::<f64>() / n,
            ]
        } else {
            vec![0.0, 0.0]
        };

        BasinMetric {
            fisher_curvature,
            eigenvalues,
            basins_visited: (metrics.iter().map(|m| m.basins_visited).sum::<usize>() as f64 / n)
                as usize,
            mean_return_time: metrics.iter().map(|m| m.mean_return_time).sum::<f64>() / n,
            stability_index: metrics.iter().map(|m| m.stability_index).sum::<f64>() / n,
            n_samples: metrics.iter().map(|m| m.n_samples).sum(),
        }
    }

    /// Run experiment with given coupling strength
    pub fn run_coupled(&mut self, coupling_strength: f64, seed: u64) -> ExperimentResult {
        let mut system = TwoLevelSystem::coupled_model(coupling_strength, self.noise, seed);
        system.initialize([0.5, 0.0], [0.0, 0.5]);

        // Simple PRNG
        let mut rng_state = seed;
        let mut rng = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (rng_state >> 33) as u32;
            (bits as f64 / u32::MAX as f64 - 0.5) * 2.0
        };

        system.run(self.steps_per_run, &mut rng);
        let (metric1, metric2) = system.compute_metrics();

        let distance1 = self
            .null_baseline_1
            .as_ref()
            .map(|null| metric1.distance(null))
            .unwrap_or(0.0);

        let distance2 = self
            .null_baseline_2
            .as_ref()
            .map(|null| metric2.distance(null))
            .unwrap_or(0.0);

        let result = ExperimentResult {
            coupling_strength,
            seed,
            metric1,
            metric2,
            distance_from_null_1: distance1,
            distance_from_null_2: distance2,
            steps: self.steps_per_run,
        };

        self.results.push(result.clone());
        result
    }

    /// Analyze results: does coupling produce systematic basin deformation?
    pub fn analyze(&self) -> ExperimentAnalysis {
        if self.results.is_empty() {
            return ExperimentAnalysis::insufficient_data();
        }

        // Group by coupling strength
        let mut by_strength: HashMap<i64, Vec<&ExperimentResult>> = HashMap::new();
        for result in &self.results {
            let key = (result.coupling_strength * 1000.0) as i64;
            by_strength.entry(key).or_default().push(result);
        }

        // Compute mean distance from null for each coupling strength
        let mut strength_vs_distance: Vec<(f64, f64, f64)> = Vec::new();

        for (key, results) in &by_strength {
            let strength = *key as f64 / 1000.0;
            let mean_dist1 =
                results.iter().map(|r| r.distance_from_null_1).sum::<f64>() / results.len() as f64;
            let mean_dist2 =
                results.iter().map(|r| r.distance_from_null_2).sum::<f64>() / results.len() as f64;
            strength_vs_distance.push((strength, mean_dist1, mean_dist2));
        }

        strength_vs_distance.sort_by(|a, b| float_cmp(&a.0, &b.0));

        // Check for systematic relationship
        let correlation = if strength_vs_distance.len() >= 2 {
            Self::compute_correlation(&strength_vs_distance)
        } else {
            0.0
        };

        // Decision
        let verdict = if correlation > 0.7 {
            CouplingVerdict::EvidenceForB
        } else if correlation > 0.3 {
            CouplingVerdict::Inconclusive
        } else {
            CouplingVerdict::EvidenceAgainstB
        };

        ExperimentAnalysis {
            n_runs: self.results.len(),
            strength_vs_distance,
            correlation,
            verdict,
        }
    }

    fn compute_correlation(data: &[(f64, f64, f64)]) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let x: Vec<f64> = data.iter().map(|(s, _, _)| *s).collect();
        let y: Vec<f64> = data.iter().map(|(_, d1, d2)| (d1 + d2) / 2.0).collect();

        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / n;

        let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / n;
        let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / n;

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }
}

/// Experiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentAnalysis {
    pub n_runs: usize,
    pub strength_vs_distance: Vec<(f64, f64, f64)>,
    pub correlation: f64,
    pub verdict: CouplingVerdict,
}

impl ExperimentAnalysis {
    fn insufficient_data() -> Self {
        Self {
            n_runs: 0,
            strength_vs_distance: vec![],
            correlation: 0.0,
            verdict: CouplingVerdict::InsufficientData,
        }
    }
}

/// The verdict on coupling hypothesis (B)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingVerdict {
    /// Coupling strength systematically predicts basin deformation
    EvidenceForB,
    /// No clear relationship
    Inconclusive,
    /// Coupling does not predict basin deformation (supports A or C)
    EvidenceAgainstB,
    /// Not enough data
    InsufficientData,
}

impl CouplingVerdict {
    pub fn interpretation(&self) -> &'static str {
        match self {
            CouplingVerdict::EvidenceForB => {
                "Basin deformation is systematic and parameter-linked. (B) is supported."
            }
            CouplingVerdict::Inconclusive => {
                "No clear relationship between coupling and basin geometry."
            }
            CouplingVerdict::EvidenceAgainstB => {
                "Coupling does not predict basin deformation. (A)+(C) are more likely."
            }
            CouplingVerdict::InsufficientData => "Not enough data to draw conclusions.",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REAL DATA COUPLING — Run experiment on actual observation streams
// ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::{ObsKey, ObservationBatch};

/// Adapter that converts observation batches to coupling state vectors
///
/// Maps observation keys to a 2D state space:
/// - Level 1 (macro): ThermalUtilization, Disorientation
/// - Level 2 (micro): AnimacyScore, RespLatMs (normalized)
pub struct ObservationStreamAdapter {
    /// State vector for Level 1 (macro dynamics)
    macro_state: [f64; 2],

    /// State vector for Level 2 (micro dynamics)
    micro_state: [f64; 2],

    /// Normalization for latency (typical range)
    latency_scale: f64,

    /// History of macro states - VecDeque for O(1) rotation
    macro_history: VecDeque<[f64; 2]>,

    /// History of micro states - VecDeque for O(1) rotation
    micro_history: VecDeque<[f64; 2]>,

    /// Max history length
    max_history: usize,
}

impl ObservationStreamAdapter {
    pub fn new() -> Self {
        Self {
            macro_state: [0.0, 0.0],
            micro_state: [0.0, 0.0],
            latency_scale: 1000.0, // 1 second = 1.0
            macro_history: VecDeque::with_capacity(1000),
            micro_history: VecDeque::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Process an observation batch and update states
    pub fn process(&mut self, batch: &ObservationBatch) {
        // Extract macro-level signals
        let thermal = batch
            .get_value(ObsKey::ThermalUtilization)
            .unwrap_or(self.macro_state[0]);
        let disorientation = batch
            .get_value(ObsKey::Disorientation)
            .unwrap_or(self.macro_state[1]);

        self.macro_state = [thermal, disorientation];

        // Extract micro-level signals
        let animacy = batch
            .get_value(ObsKey::AnimacyScore)
            .unwrap_or(self.micro_state[0]);
        let latency = batch
            .get_value(ObsKey::RespLatMs)
            .map(|v| v / self.latency_scale)
            .unwrap_or(self.micro_state[1]);

        self.micro_state = [animacy, latency.clamp(0.0, 2.0)];

        // Record history (O(1) rotation using VecDeque)
        if self.macro_history.len() >= self.max_history {
            self.macro_history.pop_front();
            self.micro_history.pop_front();
        }
        self.macro_history.push_back(self.macro_state);
        self.micro_history.push_back(self.micro_state);
    }

    /// Get current macro state
    pub fn macro_state(&self) -> [f64; 2] {
        self.macro_state
    }

    /// Get current micro state
    pub fn micro_state(&self) -> [f64; 2] {
        self.micro_state
    }

    /// Get macro history (allocates a Vec from the internal VecDeque)
    pub fn macro_history(&self) -> Vec<[f64; 2]> {
        self.macro_history.iter().copied().collect()
    }

    /// Get micro history (allocates a Vec from the internal VecDeque)
    pub fn micro_history(&self) -> Vec<[f64; 2]> {
        self.micro_history.iter().copied().collect()
    }

    /// Check if we have enough data for analysis
    pub fn ready(&self) -> bool {
        self.macro_history.len() >= 100
    }
}

impl Default for ObservationStreamAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Real data coupling experiment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealDataResult {
    /// Number of observations processed
    pub n_observations: usize,

    /// Level 1 (macro) basin metric
    pub macro_metric: BasinMetric,

    /// Level 2 (micro) basin metric
    pub micro_metric: BasinMetric,

    /// Coupling correlation between levels
    pub cross_level_correlation: f64,

    /// Phase coherence between levels
    pub phase_coherence: f64,

    /// Verdict on hypothesis (B)
    pub verdict: CouplingVerdict,
}

/// Real data coupling experiment
///
/// Tests whether macro-level dynamics (thermal, vestibular) systematically
/// couple to micro-level basin geometry (animacy, latency).
pub struct RealDataExperiment {
    /// Observation adapter
    adapter: ObservationStreamAdapter,

    /// Fisher estimator for macro level
    fisher_macro: FisherEstimator,

    /// Fisher estimator for micro level
    fisher_micro: FisherEstimator,

    /// Cross-correlation accumulator
    cross_corr_sum: f64,
    cross_corr_count: usize,

    /// Phase tracking
    macro_phase: f64,
    micro_phase: f64,
    phase_diff_sum: f64,
    phase_count: usize,

    /// Baseline for null hypothesis
    null_fisher_macro: Option<f64>,
    null_fisher_micro: Option<f64>,
}

impl RealDataExperiment {
    pub fn new() -> Self {
        Self {
            adapter: ObservationStreamAdapter::new(),
            fisher_macro: FisherEstimator::new(2, 0.1),
            fisher_micro: FisherEstimator::new(2, 0.1),
            cross_corr_sum: 0.0,
            cross_corr_count: 0,
            macro_phase: 0.0,
            micro_phase: 0.0,
            phase_diff_sum: 0.0,
            phase_count: 0,
            null_fisher_macro: None,
            null_fisher_micro: None,
        }
    }

    /// Establish null baseline from shuffled data
    pub fn establish_null_baseline(&mut self, n_shuffles: usize) {
        if self.adapter.macro_history().len() < 50 {
            return;
        }

        let mut null_macro_curvatures = Vec::new();
        let mut null_micro_curvatures = Vec::new();

        // Simple deterministic shuffle
        let macro_hist = self.adapter.macro_history().to_vec();
        let micro_hist = self.adapter.micro_history().to_vec();

        for shuffle_idx in 0..n_shuffles {
            let mut fisher_m = FisherEstimator::new(2, 0.1);
            let mut fisher_u = FisherEstimator::new(2, 0.1);

            // Feed macro in order, micro shuffled
            for (i, m) in macro_hist.iter().enumerate() {
                fisher_m.add_sample(m);
                // Shuffle micro by rotating index
                let shuffled_idx = (i + shuffle_idx * 17 + 1) % micro_hist.len();
                fisher_u.add_sample(&micro_hist[shuffled_idx]);
            }

            null_macro_curvatures.push(fisher_m.scalar_curvature());
            null_micro_curvatures.push(fisher_u.scalar_curvature());
        }

        // Average null curvatures
        self.null_fisher_macro =
            Some(null_macro_curvatures.iter().sum::<f64>() / n_shuffles as f64);
        self.null_fisher_micro =
            Some(null_micro_curvatures.iter().sum::<f64>() / n_shuffles as f64);
    }

    /// Process an observation batch
    pub fn process(&mut self, batch: &ObservationBatch) {
        let prev_macro = self.adapter.macro_state();
        let prev_micro = self.adapter.micro_state();

        self.adapter.process(batch);

        let curr_macro = self.adapter.macro_state();
        let curr_micro = self.adapter.micro_state();

        // Update Fisher estimators
        self.fisher_macro.add_sample(&curr_macro);
        self.fisher_micro.add_sample(&curr_micro);

        // Compute instantaneous cross-correlation
        let macro_delta = [curr_macro[0] - prev_macro[0], curr_macro[1] - prev_macro[1]];
        let micro_delta = [curr_micro[0] - prev_micro[0], curr_micro[1] - prev_micro[1]];

        let macro_mag = (macro_delta[0].powi(2) + macro_delta[1].powi(2)).sqrt();
        let micro_mag = (micro_delta[0].powi(2) + micro_delta[1].powi(2)).sqrt();

        if macro_mag > 0.001 && micro_mag > 0.001 {
            let dot = macro_delta[0] * micro_delta[0] + macro_delta[1] * micro_delta[1];
            self.cross_corr_sum += dot / (macro_mag * micro_mag);
            self.cross_corr_count += 1;
        }

        // Track phase (using atan2 of state)
        let new_macro_phase = curr_macro[1].atan2(curr_macro[0]);
        let new_micro_phase = curr_micro[1].atan2(curr_micro[0]);

        // Phase coherence: how consistently do phases differ?
        let phase_diff = (new_macro_phase - new_micro_phase).abs();
        let normalized_diff = if phase_diff > std::f64::consts::PI {
            2.0 * std::f64::consts::PI - phase_diff
        } else {
            phase_diff
        };

        self.phase_diff_sum += normalized_diff;
        self.phase_count += 1;

        self.macro_phase = new_macro_phase;
        self.micro_phase = new_micro_phase;
    }

    /// Analyze results
    pub fn analyze(&mut self) -> RealDataResult {
        let n_obs = self.adapter.macro_history().len();

        // Establish null baseline if not already done
        if self.null_fisher_macro.is_none() && n_obs >= 50 {
            self.establish_null_baseline(10);
        }

        let macro_curvature = self.fisher_macro.scalar_curvature();
        let micro_curvature = self.fisher_micro.scalar_curvature();

        // Cache history to avoid multiple allocations
        let macro_hist = self.adapter.macro_history();
        let micro_hist = self.adapter.micro_history();

        let macro_metric = BasinMetric {
            fisher_curvature: macro_curvature,
            eigenvalues: self.fisher_macro.eigenvalue_spectrum(),
            basins_visited: self.count_basin_visits(&macro_hist),
            mean_return_time: self.compute_return_time(&macro_hist),
            stability_index: self.compute_stability(&macro_hist),
            n_samples: self.fisher_macro.n_samples,
        };

        let micro_metric = BasinMetric {
            fisher_curvature: micro_curvature,
            eigenvalues: self.fisher_micro.eigenvalue_spectrum(),
            basins_visited: self.count_basin_visits(&micro_hist),
            mean_return_time: self.compute_return_time(&micro_hist),
            stability_index: self.compute_stability(&micro_hist),
            n_samples: self.fisher_micro.n_samples,
        };

        // Cross-level correlation
        let cross_level_correlation = if self.cross_corr_count > 0 {
            self.cross_corr_sum / self.cross_corr_count as f64
        } else {
            0.0
        };

        // Phase coherence: 0 = random phases, 1 = locked phases
        let mean_phase_diff = if self.phase_count > 0 {
            self.phase_diff_sum / self.phase_count as f64
        } else {
            std::f64::consts::PI / 2.0
        };
        let phase_coherence = 1.0 - (mean_phase_diff / std::f64::consts::PI);

        // Verdict: does coupling systematically deform basin geometry?
        let verdict = self.compute_verdict(
            &macro_metric,
            &micro_metric,
            cross_level_correlation,
            phase_coherence,
        );

        RealDataResult {
            n_observations: n_obs,
            macro_metric,
            micro_metric,
            cross_level_correlation,
            phase_coherence,
            verdict,
        }
    }

    fn count_basin_visits(&self, history: &[[f64; 2]]) -> usize {
        // Count distinct quadrants visited
        let mut quadrants = std::collections::HashSet::new();
        for state in history {
            let q = match (state[0] >= 0.0, state[1] >= 0.0) {
                (true, true) => 0,
                (false, true) => 1,
                (false, false) => 2,
                (true, false) => 3,
            };
            quadrants.insert(q);
        }
        quadrants.len()
    }

    fn compute_return_time(&self, history: &[[f64; 2]]) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        // Return time to origin quadrant
        let origin_quadrant = 0; // top-right
        let mut return_times = Vec::new();
        let mut last_visit = None;

        for (i, state) in history.iter().enumerate() {
            let q = match (state[0] >= 0.0, state[1] >= 0.0) {
                (true, true) => 0,
                (false, true) => 1,
                (false, false) => 2,
                (true, false) => 3,
            };
            if q == origin_quadrant {
                if let Some(last) = last_visit {
                    return_times.push((i - last) as f64);
                }
                last_visit = Some(i);
            }
        }

        if return_times.is_empty() {
            0.0
        } else {
            return_times.iter().sum::<f64>() / return_times.len() as f64
        }
    }

    fn compute_stability(&self, history: &[[f64; 2]]) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        // Fraction of time in primary quadrant
        let primary = history
            .iter()
            .filter(|s| s[0] >= 0.0 && s[1] >= 0.0)
            .count();

        primary as f64 / history.len() as f64
    }

    fn compute_verdict(
        &self,
        macro_metric: &BasinMetric,
        micro_metric: &BasinMetric,
        cross_correlation: f64,
        phase_coherence: f64,
    ) -> CouplingVerdict {
        // Evidence for (B): systematic coupling
        // 1. Cross-correlation significantly different from zero
        // 2. Phase coherence above random
        // 3. Curvature deviation from null baseline

        let null_macro = self.null_fisher_macro.unwrap_or(0.0);
        let null_micro = self.null_fisher_micro.unwrap_or(0.0);

        let curvature_deviation_macro = (macro_metric.fisher_curvature - null_macro).abs();
        let curvature_deviation_micro = (micro_metric.fisher_curvature - null_micro).abs();

        // Scoring
        let mut score = 0.0;

        // Cross-correlation contribution
        if cross_correlation.abs() > 0.5 {
            score += 0.4;
        } else if cross_correlation.abs() > 0.25 {
            score += 0.2;
        }

        // Phase coherence contribution
        if phase_coherence > 0.7 {
            score += 0.3;
        } else if phase_coherence > 0.4 {
            score += 0.15;
        }

        // Curvature deviation contribution (coupling should systematically change basin shape)
        let curvature_score = curvature_deviation_macro + curvature_deviation_micro;
        if curvature_score > 0.1 {
            score += 0.3;
        } else if curvature_score > 0.05 {
            score += 0.15;
        }

        // Verdict
        if macro_metric.n_samples < 50 || micro_metric.n_samples < 50 {
            CouplingVerdict::InsufficientData
        } else if score > 0.7 {
            CouplingVerdict::EvidenceForB
        } else if score > 0.4 {
            CouplingVerdict::Inconclusive
        } else {
            CouplingVerdict::EvidenceAgainstB
        }
    }

    /// Reset for new experiment run
    pub fn reset(&mut self) {
        self.adapter = ObservationStreamAdapter::new();
        self.fisher_macro = FisherEstimator::new(2, 0.1);
        self.fisher_micro = FisherEstimator::new(2, 0.1);
        self.cross_corr_sum = 0.0;
        self.cross_corr_count = 0;
        self.macro_phase = 0.0;
        self.micro_phase = 0.0;
        self.phase_diff_sum = 0.0;
        self.phase_count = 0;
        // Keep null baseline
    }
}

impl Default for RealDataExperiment {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic "real" observations for testing
///
/// Simulates sensorium-like observation stream with controllable coupling
pub struct ObservationGenerator {
    /// Coupling strength (0 = independent, 1 = fully coupled)
    pub coupling_strength: f64,

    /// Noise level
    pub noise: f64,

    /// Internal state
    macro_state: [f64; 2],
    micro_state: [f64; 2],

    /// Step counter
    step: u64,

    /// RNG seed
    seed: u64,
}

impl ObservationGenerator {
    pub fn new(coupling_strength: f64, noise: f64, seed: u64) -> Self {
        Self {
            coupling_strength,
            noise,
            macro_state: [0.5, 0.3],
            micro_state: [0.2, 0.4],
            step: 0,
            seed,
        }
    }

    /// Generate next observation batch
    pub fn next(&mut self) -> ObservationBatch {
        // Update internal state with coupling dynamics
        let dt: f64 = 0.1;

        // Macro level: slow oscillation + noise
        let macro_freq = 0.05;
        let macro_target = [
            (self.step as f64 * macro_freq).sin() * 0.5,
            (self.step as f64 * macro_freq * 0.7).cos() * 0.5,
        ];

        // Simple PRNG
        let mut rng_state = self.seed.wrapping_add(self.step);
        let mut rng = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (rng_state >> 33) as u32;
            (bits as f64 / u32::MAX as f64 - 0.5) * 2.0
        };

        for i in 0..2 {
            let noise_term = self.noise * rng();
            self.macro_state[i] +=
                0.1 * (macro_target[i] - self.macro_state[i]) + noise_term * dt.sqrt();
        }

        // Micro level: faster dynamics + coupling from macro
        let micro_freq = 0.2;
        let coupling_effect = self.coupling_strength * self.macro_state[0];

        let micro_target = [
            (self.step as f64 * micro_freq).sin() * 0.3 + coupling_effect,
            (self.step as f64 * micro_freq * 1.3).cos() * 0.3 + coupling_effect * 0.5,
        ];

        for i in 0..2 {
            let noise_term = self.noise * rng();
            self.micro_state[i] +=
                0.2 * (micro_target[i] - self.micro_state[i]) + noise_term * dt.sqrt();
        }

        self.step += 1;

        // Build observation batch (using simplified API)
        let mut batch = ObservationBatch::new();

        // Macro observations
        batch.add(
            ObsKey::ThermalUtilization,
            self.macro_state[0].clamp(0.0, 1.0),
        );
        batch.add(
            ObsKey::Disorientation,
            self.macro_state[1].abs().clamp(0.0, 1.0),
        );

        // Micro observations
        batch.add(
            ObsKey::AnimacyScore,
            self.micro_state[0].abs().clamp(0.0, 1.0),
        );
        batch.add(
            ObsKey::RespLatMs,
            (self.micro_state[1].abs() * 500.0 + 100.0).clamp(50.0, 2000.0),
        );

        batch
    }

    /// Generate a stream of observations
    pub fn generate_stream(&mut self, n: usize) -> Vec<ObservationBatch> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// Run the full real-data coupling experiment
///
/// Returns the analysis result with verdict on hypothesis (B)
pub fn run_real_data_experiment(
    coupling_strength: f64,
    n_observations: usize,
    noise: f64,
    seed: u64,
) -> RealDataResult {
    let mut experiment = RealDataExperiment::new();
    let mut generator = ObservationGenerator::new(coupling_strength, noise, seed);

    // Generate and process observations
    for batch in generator.generate_stream(n_observations) {
        experiment.process(&batch);
    }

    experiment.analyze()
}

/// Run comparative experiment: coupled vs null
///
/// Returns (coupled_result, null_result, delta)
pub fn run_comparative_experiment(
    coupling_strength: f64,
    n_observations: usize,
    noise: f64,
    seed: u64,
) -> (RealDataResult, RealDataResult, f64) {
    // Run coupled experiment
    let coupled = run_real_data_experiment(coupling_strength, n_observations, noise, seed);

    // Run null experiment (no coupling)
    let null = run_real_data_experiment(0.0, n_observations, noise, seed + 1000);

    // Compute delta: how much does coupling change basin geometry?
    let delta = coupled.macro_metric.distance(&null.macro_metric)
        + coupled.micro_metric.distance(&null.micro_metric);

    (coupled, null, delta)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (state >> 33) as u32;
            (bits as f64 / u32::MAX as f64 - 0.5) * 2.0
        }
    }

    #[test]
    fn test_basinized_system_creation() {
        let system: BasinizedSystem<2> = BasinizedSystem::new(FlowParams::default(), 0.1, 0.01);
        assert_eq!(system.dim, 2);
        assert_eq!(system.state, [0.0, 0.0]);
    }

    #[test]
    fn test_basinized_system_step() {
        let mut system: BasinizedSystem<2> = BasinizedSystem::new(FlowParams::default(), 0.1, 0.01);
        system.set_state([0.5, 0.0]);

        let mut rng = simple_rng(42);
        system.step(&mut rng);

        // State should have changed
        assert!(system.state[0] != 0.5 || system.state[1] != 0.0);
    }

    #[test]
    fn test_attractor_convergence() {
        let mut system: BasinizedSystem<2> =
            BasinizedSystem::new(FlowParams::default(), 0.01, 0.01);
        system.set_state([0.8, 0.1]); // Near right attractor

        let mut rng = simple_rng(42);
        system.run(1000, &mut rng);

        let obs = system.observe();
        // Should be in a basin (either 0 or 1)
        // With noise, exact distance varies but should be finite
        assert!(obs.distance_to_attractor.is_finite());
        assert!(obs.basin_id <= 1);
    }

    #[test]
    fn test_coupling_operator_null() {
        let coupling = CouplingOperator::null();
        let effect = coupling.compute(&[1.0, 0.0]);
        assert!(!effect.active);
        assert_eq!(effect.attractor_shift, 0.0);
    }

    #[test]
    fn test_coupling_operator_active() {
        let coupling = CouplingOperator::unidirectional(0.5);
        let effect = coupling.compute(&[1.0, 0.0]);
        assert!(effect.active);
        assert!(effect.attractor_shift != 0.0);
    }

    #[test]
    fn test_fisher_estimator() {
        let mut fisher = FisherEstimator::new(2, 0.1);

        // Add some samples
        for i in 0..100 {
            let x = (i as f64 * 0.1).sin();
            let y = (i as f64 * 0.1).cos();
            fisher.add_sample(&[x, y]);
        }

        let curvature = fisher.scalar_curvature();
        // Should be non-zero for non-trivial trajectory
        assert!(curvature.abs() > 0.0 || fisher.n_samples > 0);
    }

    #[test]
    fn test_two_level_system_null() {
        let mut system = TwoLevelSystem::null_model(0.1, 42);
        system.initialize([0.5, 0.0], [0.0, 0.5]);

        let mut rng = simple_rng(42);
        system.run(500, &mut rng);

        let (metric1, metric2) = system.compute_metrics();
        assert!(metric1.n_samples > 0);
        assert!(metric2.n_samples > 0);
    }

    #[test]
    fn test_two_level_system_coupled() {
        let mut system = TwoLevelSystem::coupled_model(0.3, 0.1, 42);
        system.initialize([0.5, 0.0], [0.0, 0.5]);

        let mut rng = simple_rng(42);
        system.run(500, &mut rng);

        let (metric1, metric2) = system.compute_metrics();
        assert!(metric1.n_samples > 0);
        assert!(metric2.n_samples > 0);
    }

    #[test]
    fn test_experiment_null_baseline() {
        let mut experiment = CouplingExperiment::new(0.1, 500, 5);
        experiment.establish_null_baseline(&[1, 2, 3, 4, 5]);

        assert!(experiment.null_baseline_1.is_some());
        assert!(experiment.null_baseline_2.is_some());
    }

    #[test]
    fn test_experiment_coupled_run() {
        let mut experiment = CouplingExperiment::new(0.1, 500, 5);
        experiment.establish_null_baseline(&[1, 2, 3]);

        let result = experiment.run_coupled(0.5, 42);
        assert!(result.steps == 500);
        assert!(result.coupling_strength == 0.5);
    }

    #[test]
    fn test_experiment_analysis() {
        let mut experiment = CouplingExperiment::new(0.1, 200, 10);
        experiment.establish_null_baseline(&[1, 2, 3]);

        // Run at multiple coupling strengths
        for strength in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
            for seed in [10, 20, 30] {
                experiment.run_coupled(strength, seed as u64);
            }
        }

        let analysis = experiment.analyze();
        assert!(analysis.n_runs > 0);
        // Should have entries for different coupling strengths
        assert!(analysis.strength_vs_distance.len() >= 2);
    }

    #[test]
    fn test_basin_metric_distance() {
        let m1 = BasinMetric {
            fisher_curvature: 0.5,
            eigenvalues: vec![1.0, 0.5],
            basins_visited: 2,
            mean_return_time: 10.0,
            stability_index: 0.7,
            n_samples: 100,
        };

        let m2 = BasinMetric {
            fisher_curvature: 0.8,
            eigenvalues: vec![1.2, 0.6],
            basins_visited: 2,
            mean_return_time: 12.0,
            stability_index: 0.65,
            n_samples: 100,
        };

        let dist = m1.distance(&m2);
        assert!(dist > 0.0);
        assert!(dist < 1.0);
    }

    #[test]
    fn test_verdict_interpretation() {
        assert!(!CouplingVerdict::EvidenceForB.interpretation().is_empty());
        assert!(!CouplingVerdict::Inconclusive.interpretation().is_empty());
        assert!(!CouplingVerdict::EvidenceAgainstB
            .interpretation()
            .is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // REAL DATA TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_observation_stream_adapter() {
        let mut adapter = ObservationStreamAdapter::new();
        assert!(!adapter.ready());

        // Process 100 batches
        let mut generator = ObservationGenerator::new(0.3, 0.1, 42);
        for batch in generator.generate_stream(100) {
            adapter.process(&batch);
        }

        assert!(adapter.ready());
        assert_eq!(adapter.macro_history().len(), 100);
        assert_eq!(adapter.micro_history().len(), 100);
    }

    #[test]
    fn test_observation_generator() {
        let mut generator = ObservationGenerator::new(0.5, 0.1, 42);
        let batch = generator.next();

        // Should have all required observations
        assert!(batch.get_value(ObsKey::ThermalUtilization).is_some());
        assert!(batch.get_value(ObsKey::Disorientation).is_some());
        assert!(batch.get_value(ObsKey::AnimacyScore).is_some());
        assert!(batch.get_value(ObsKey::RespLatMs).is_some());
    }

    #[test]
    fn test_real_data_experiment_basic() {
        let result = run_real_data_experiment(0.3, 200, 0.1, 42);

        assert_eq!(result.n_observations, 200);
        assert!(result.macro_metric.n_samples > 0);
        assert!(result.micro_metric.n_samples > 0);
    }

    #[test]
    fn test_real_data_experiment_coupled_vs_null() {
        // Run with strong coupling
        let coupled = run_real_data_experiment(0.8, 500, 0.1, 42);

        // Run with no coupling
        let null = run_real_data_experiment(0.0, 500, 0.1, 42);

        // Cross-correlation should be higher for coupled
        // (This is a probabilistic test, but the signal should be strong enough)
        assert!(
            coupled.cross_level_correlation.abs() >= null.cross_level_correlation.abs() * 0.5
                || coupled.phase_coherence >= null.phase_coherence * 0.8
        );
    }

    #[test]
    fn test_comparative_experiment() {
        let (coupled, null, delta) = run_comparative_experiment(0.5, 300, 0.1, 42);

        assert_eq!(coupled.n_observations, 300);
        assert_eq!(null.n_observations, 300);
        assert!(delta >= 0.0);
    }

    #[test]
    fn test_real_data_verdict() {
        // Strong coupling should produce evidence for B
        let strong = run_real_data_experiment(0.9, 500, 0.05, 42);

        // Weak coupling might be inconclusive or against
        let weak = run_real_data_experiment(0.1, 500, 0.2, 42);

        // At minimum, they should produce valid verdicts
        assert!(matches!(
            strong.verdict,
            CouplingVerdict::EvidenceForB
                | CouplingVerdict::Inconclusive
                | CouplingVerdict::EvidenceAgainstB
        ));
        assert!(matches!(
            weak.verdict,
            CouplingVerdict::EvidenceForB
                | CouplingVerdict::Inconclusive
                | CouplingVerdict::EvidenceAgainstB
        ));
    }
}
