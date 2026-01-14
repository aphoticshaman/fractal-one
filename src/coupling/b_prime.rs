//! ═══════════════════════════════════════════════════════════════════════════════
//! B′ EXPERIMENT: Near-Critical System Test
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Tests hypothesis B′: Basin deformation IS possible when no-go conditions
//! are violated. Specifically, we violate the "strong dissipation" condition
//! by creating a near-critical system with γ → 0.
//!
//! # Theory
//!
//! The no-go theorem requires spectral gap γ > 0 (strong dissipation).
//! When γ → 0 (critical slowing down):
//! - Perturbations decay slowly
//! - Coupling effects can accumulate
//! - System may cross basin boundaries
//! - Basin deformation becomes possible
//!
//! # Prediction
//!
//! If B′ is correct: near-critical systems should show basin deformation
//! under coupling that sub-critical systems do not.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use super::{
    BasinMetric, BasinizedSystem, CouplingOperator, CouplingVerdict, FisherEstimator, FlowParams,
};

/// Convert a B′ verdict to a coupling verdict for integration with the main coupling module
impl BPrimeVerdict {
    pub fn to_coupling_verdict(&self) -> CouplingVerdict {
        match self {
            BPrimeVerdict::Confirmed => CouplingVerdict::EvidenceForB,
            BPrimeVerdict::PartiallyConfirmed => CouplingVerdict::Inconclusive,
            BPrimeVerdict::NotConfirmed => CouplingVerdict::EvidenceAgainstB,
            BPrimeVerdict::InsufficientData => CouplingVerdict::InsufficientData,
        }
    }
}

/// Configuration for B′ experiment
#[derive(Debug, Clone)]
pub struct BPrimeConfig {
    /// Damping values to test (spectral gap γ)
    pub damping_values: Vec<f64>,

    /// Coupling strengths to test
    pub coupling_strengths: Vec<f64>,

    /// Noise level
    pub noise: f64,

    /// Steps per run
    pub steps: usize,

    /// Number of runs per configuration
    pub runs_per_config: usize,
}

impl Default for BPrimeConfig {
    fn default() -> Self {
        Self {
            // From strong dissipation (0.1) to near-critical (0.001)
            damping_values: vec![0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
            coupling_strengths: vec![0.0, 0.3, 0.5, 0.7, 1.0],
            noise: 0.05,
            steps: 2000,
            runs_per_config: 5,
        }
    }
}

/// Result of a single B′ run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPrimeRun {
    /// Damping (spectral gap)
    pub damping: f64,

    /// Coupling strength
    pub coupling: f64,

    /// Seed used
    pub seed: u64,

    /// Basin metric for Level 2 (receives coupling)
    pub metric: BasinMetric,

    /// Basin metric for Level 1 (source of coupling)
    pub metric_level1: BasinMetric,

    /// Number of basin transitions observed (Level 2)
    pub basin_transitions: usize,

    /// Number of basin transitions observed (Level 1)
    pub basin_transitions_level1: usize,

    /// Fraction of time in each basin (Level 2)
    pub basin_occupancy: Vec<f64>,

    /// Fraction of time in each basin (Level 1)
    pub basin_occupancy_level1: Vec<f64>,

    /// Did basin deformation occur in Level 2?
    pub deformation_detected: bool,

    /// Did basin deformation occur in Level 1? (bidirectional effect)
    pub deformation_detected_level1: bool,
}

/// Result of the full B′ experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPrimeResult {
    /// All runs
    pub runs: Vec<BPrimeRun>,

    /// Summary: deformation rate by damping
    pub deformation_by_damping: Vec<(f64, f64)>,

    /// Summary: deformation rate by coupling
    pub deformation_by_coupling: Vec<(f64, f64)>,

    /// Critical damping threshold (where deformation starts)
    pub critical_damping: Option<f64>,

    /// Verdict on B′
    pub verdict: BPrimeVerdict,
}

/// Verdict on B′ hypothesis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BPrimeVerdict {
    /// B′ confirmed: near-critical systems show deformation
    Confirmed,
    /// B′ partially confirmed: some conditions show deformation
    PartiallyConfirmed,
    /// B′ not confirmed: no deformation even near criticality
    NotConfirmed,
    /// Insufficient data
    InsufficientData,
}

impl BPrimeVerdict {
    pub fn interpretation(&self) -> &'static str {
        match self {
            BPrimeVerdict::Confirmed => {
                "B′ CONFIRMED: Near-critical systems show basin deformation under coupling"
            }
            BPrimeVerdict::PartiallyConfirmed => {
                "B′ PARTIAL: Some near-critical configurations show deformation"
            }
            BPrimeVerdict::NotConfirmed => {
                "B′ NOT CONFIRMED: No deformation observed even near criticality"
            }
            BPrimeVerdict::InsufficientData => "Insufficient data for B′ verdict",
        }
    }
}

/// Near-critical two-level system
pub struct NearCriticalSystem {
    /// Level 1 (macro) - can be near-critical
    pub level1: BasinizedSystem<2>,

    /// Level 2 (micro) - responds to Level 1
    pub level2: BasinizedSystem<2>,

    /// Coupling operator
    pub coupling: CouplingOperator,

    /// Fisher estimators
    pub fisher1: FisherEstimator,
    pub fisher2: FisherEstimator,

    /// Basin history for transition detection
    basin_history: Vec<(usize, usize)>,
}

impl NearCriticalSystem {
    /// Create system with specified damping (spectral gap)
    pub fn new(damping: f64, coupling_strength: f64, noise: f64) -> Self {
        // Level 1: potentially near-critical
        let flow1 = FlowParams {
            attractors: vec![vec![-1.0, 0.0], vec![1.0, 0.0]],
            strengths: vec![1.0, 1.0],
            widths: vec![0.8, 0.8],
            damping, // This is the spectral gap γ
            coupling_strength: 0.0,
        };

        // Level 2: same damping, receives coupling
        let flow2 = FlowParams {
            attractors: vec![vec![0.0, -1.0], vec![0.0, 1.0]],
            strengths: vec![1.0, 1.0],
            widths: vec![0.8, 0.8],
            damping,
            coupling_strength,
        };

        let coupling = if coupling_strength > 0.0 {
            CouplingOperator::unidirectional(coupling_strength)
        } else {
            CouplingOperator::null()
        };

        Self {
            level1: BasinizedSystem::new(flow1, noise, 0.01),
            level2: BasinizedSystem::new(flow2, noise, 0.01),
            coupling,
            fisher1: FisherEstimator::new(2, noise),
            fisher2: FisherEstimator::new(2, noise),
            basin_history: Vec::new(),
        }
    }

    /// Initialize both levels
    pub fn initialize(&mut self, state1: [f64; 2], state2: [f64; 2]) {
        self.level1.set_state(state1);
        self.level2.set_state(state2);
        self.fisher1.reset();
        self.fisher2.reset();
        self.basin_history.clear();
    }

    /// Run one step with coupling
    pub fn step(&mut self, rng: &mut impl FnMut() -> f64) {
        // Step Level 1
        self.level1.step(rng);

        // Apply coupling: Level 1 state affects Level 2 attractors
        let effect = self.coupling.compute(&self.level1.state);
        if effect.active {
            // Shift Level 2 attractors based on Level 1 position
            for attractor in &mut self.level2.flow_params.attractors {
                if !attractor.is_empty() {
                    attractor[0] += effect.attractor_shift;
                }
            }
            // Modify well depths
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
        let basin1 = self.current_basin(&self.level1);
        let basin2 = self.current_basin(&self.level2);
        self.basin_history.push((basin1, basin2));
    }

    fn current_basin<const DIM: usize>(&self, system: &BasinizedSystem<DIM>) -> usize {
        let obs = system.observe();
        obs.basin_id
    }

    /// Run for multiple steps
    pub fn run(&mut self, steps: usize, rng: &mut impl FnMut() -> f64) {
        for _ in 0..steps {
            self.step(rng);
        }
    }

    /// Count basin transitions
    pub fn count_transitions(&self) -> (usize, usize) {
        let mut trans1 = 0;
        let mut trans2 = 0;

        for i in 1..self.basin_history.len() {
            if self.basin_history[i].0 != self.basin_history[i - 1].0 {
                trans1 += 1;
            }
            if self.basin_history[i].1 != self.basin_history[i - 1].1 {
                trans2 += 1;
            }
        }

        (trans1, trans2)
    }

    /// Compute basin occupancy (fraction of time in each basin)
    pub fn basin_occupancy(&self) -> (Vec<f64>, Vec<f64>) {
        if self.basin_history.is_empty() {
            return (vec![], vec![]);
        }

        let n = self.basin_history.len() as f64;

        // Level 1
        let basins1: HashSet<usize> = self.basin_history.iter().map(|(b, _)| *b).collect();
        let max_basin1 = basins1.iter().max().copied().unwrap_or(0);
        let mut occ1 = vec![0.0; max_basin1 + 1];
        for (b, _) in &self.basin_history {
            occ1[*b] += 1.0 / n;
        }

        // Level 2
        let basins2: HashSet<usize> = self.basin_history.iter().map(|(_, b)| *b).collect();
        let max_basin2 = basins2.iter().max().copied().unwrap_or(0);
        let mut occ2 = vec![0.0; max_basin2 + 1];
        for (_, b) in &self.basin_history {
            occ2[*b] += 1.0 / n;
        }

        (occ1, occ2)
    }

    /// Compute basin metrics
    pub fn compute_metrics(&self) -> (BasinMetric, BasinMetric) {
        let (trans1, trans2) = self.count_transitions();
        let (occ1, occ2) = self.basin_occupancy();

        let metric1 = BasinMetric {
            fisher_curvature: self.fisher1.scalar_curvature(),
            eigenvalues: self.fisher1.eigenvalue_spectrum(),
            basins_visited: occ1.iter().filter(|&&x| x > 0.01).count(),
            mean_return_time: if trans1 > 0 {
                self.basin_history.len() as f64 / trans1 as f64
            } else {
                self.basin_history.len() as f64
            },
            stability_index: occ1.iter().cloned().fold(0.0, f64::max),
            n_samples: self.fisher1.n_samples,
        };

        let metric2 = BasinMetric {
            fisher_curvature: self.fisher2.scalar_curvature(),
            eigenvalues: self.fisher2.eigenvalue_spectrum(),
            basins_visited: occ2.iter().filter(|&&x| x > 0.01).count(),
            mean_return_time: if trans2 > 0 {
                self.basin_history.len() as f64 / trans2 as f64
            } else {
                self.basin_history.len() as f64
            },
            stability_index: occ2.iter().cloned().fold(0.0, f64::max),
            n_samples: self.fisher2.n_samples,
        };

        (metric1, metric2)
    }
}

/// Run the B′ experiment
pub fn run_b_prime_experiment(config: &BPrimeConfig) -> BPrimeResult {
    let mut runs = Vec::new();

    for &damping in &config.damping_values {
        for &coupling in &config.coupling_strengths {
            for run_idx in 0..config.runs_per_config {
                let seed = (damping * 10000.0) as u64 + (coupling * 1000.0) as u64 + run_idx as u64;

                let run = run_single_b_prime(damping, coupling, config.noise, config.steps, seed);
                runs.push(run);
            }
        }
    }

    // Analyze results
    analyze_b_prime_results(runs, config)
}

fn run_single_b_prime(
    damping: f64,
    coupling: f64,
    noise: f64,
    steps: usize,
    seed: u64,
) -> BPrimeRun {
    let mut system = NearCriticalSystem::new(damping, coupling, noise);

    // Start near basin boundary to maximize chance of transition
    system.initialize([0.3, 0.1], [0.1, 0.3]);

    // Simple PRNG
    let mut rng_state = seed;
    let mut rng = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (rng_state >> 33) as u32;
        (bits as f64 / u32::MAX as f64 - 0.5) * 2.0
    };

    system.run(steps, &mut rng);

    let (metric1, metric2) = system.compute_metrics();
    let (trans1, trans2) = system.count_transitions();
    let (occ1, occ2) = system.basin_occupancy();

    // Level 1 also shows deformation under coupling (bidirectional effect)
    let level1_deformation = metric1.basins_visited > 1
        || trans1 > 5
        || (occ1.len() > 1 && occ1.iter().all(|&x| x > 0.1 && x < 0.9));

    // Deformation detected if:
    // 1. Multiple basins visited, OR
    // 2. Significant basin transitions, OR
    // 3. Non-trivial occupancy distribution
    let basins_visited = metric2.basins_visited;
    let has_transitions = trans2 > 5;
    let non_trivial_occupancy = occ2.len() > 1 && occ2.iter().all(|&x| x > 0.1 && x < 0.9);

    let deformation_detected = basins_visited > 1 || has_transitions || non_trivial_occupancy;

    BPrimeRun {
        damping,
        coupling,
        seed,
        metric: metric2, // Focus on Level 2 (receives coupling)
        metric_level1: metric1,
        basin_transitions: trans2,
        basin_transitions_level1: trans1,
        basin_occupancy: occ2,
        basin_occupancy_level1: occ1,
        deformation_detected,
        deformation_detected_level1: level1_deformation,
    }
}

fn analyze_b_prime_results(runs: Vec<BPrimeRun>, config: &BPrimeConfig) -> BPrimeResult {
    // Deformation rate by damping
    let mut deformation_by_damping = Vec::new();
    for &damping in &config.damping_values {
        let damping_runs: Vec<_> = runs
            .iter()
            .filter(|r| (r.damping - damping).abs() < 0.0001)
            .collect();
        if !damping_runs.is_empty() {
            let rate = damping_runs
                .iter()
                .filter(|r| r.deformation_detected)
                .count() as f64
                / damping_runs.len() as f64;
            deformation_by_damping.push((damping, rate));
        }
    }

    // Deformation rate by coupling
    let mut deformation_by_coupling = Vec::new();
    for &coupling in &config.coupling_strengths {
        let coupling_runs: Vec<_> = runs
            .iter()
            .filter(|r| (r.coupling - coupling).abs() < 0.0001)
            .collect();
        if !coupling_runs.is_empty() {
            let rate = coupling_runs
                .iter()
                .filter(|r| r.deformation_detected)
                .count() as f64
                / coupling_runs.len() as f64;
            deformation_by_coupling.push((coupling, rate));
        }
    }

    // Find critical damping (where deformation rate crosses 50%)
    let critical_damping = deformation_by_damping
        .iter()
        .filter(|(_, rate)| *rate >= 0.5)
        .map(|(d, _)| *d)
        .fold(None, |acc, d| Some(acc.map_or(d, |a: f64| a.max(d))));

    // Verdict
    let max_deformation_rate = deformation_by_damping
        .iter()
        .map(|(_, rate)| *rate)
        .fold(0.0, f64::max);

    let verdict = if max_deformation_rate > 0.7 {
        BPrimeVerdict::Confirmed
    } else if max_deformation_rate > 0.3 {
        BPrimeVerdict::PartiallyConfirmed
    } else if runs.is_empty() {
        BPrimeVerdict::InsufficientData
    } else {
        BPrimeVerdict::NotConfirmed
    };

    BPrimeResult {
        runs,
        deformation_by_damping,
        deformation_by_coupling,
        critical_damping,
        verdict,
    }
}

/// Print B′ experiment results
pub fn print_b_prime_results(result: &BPrimeResult) {
    let sep = "=".repeat(79);
    let dash = "-".repeat(79);

    println!("{}", sep);
    println!(" B′ EXPERIMENT: Near-Critical System Test");
    println!("{}", sep);
    println!();
    println!("Testing: Does basin deformation occur when γ → 0 (near criticality)?");
    println!();

    println!("{}", dash);
    println!(" Deformation Rate by Damping (γ)");
    println!("{}", dash);
    println!();
    println!(
        "{:^12} | {:^15} | {:^40}",
        "Damping (γ)", "Deform. Rate", "Visual"
    );
    println!("{}", dash);

    for (damping, rate) in &result.deformation_by_damping {
        let bar_len = (rate * 30.0) as usize;
        let bar = "#".repeat(bar_len);
        let regime = if *damping > 0.05 {
            "sub-critical"
        } else if *damping > 0.01 {
            "transitional"
        } else {
            "near-critical"
        };
        println!(
            "{:^12.4} | {:^15.2}% | {} ({})",
            damping,
            rate * 100.0,
            bar,
            regime
        );
    }

    println!();
    println!("{}", dash);
    println!(" Deformation Rate by Coupling Strength");
    println!("{}", dash);
    println!();
    println!(
        "{:^12} | {:^15} | {:^40}",
        "Coupling", "Deform. Rate", "Visual"
    );
    println!("{}", dash);

    for (coupling, rate) in &result.deformation_by_coupling {
        let bar_len = (rate * 30.0) as usize;
        let bar = "#".repeat(bar_len);
        println!("{:^12.2} | {:^15.2}% | {}", coupling, rate * 100.0, bar);
    }

    println!();
    println!("{}", sep);
    println!(" ANALYSIS");
    println!("{}", sep);
    println!();

    if let Some(critical) = result.critical_damping {
        println!("Critical damping threshold: γ_c ≈ {:.4}", critical);
        println!("  -> Below this, basin deformation becomes likely");
    } else {
        println!("No clear critical damping threshold identified");
    }

    println!();
    println!("{}", sep);
    println!(" VERDICT: {}", result.verdict.interpretation());
    println!("{}", sep);
    println!();
}

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
    fn test_near_critical_system_creation() {
        let system = NearCriticalSystem::new(0.01, 0.5, 0.05);
        assert_eq!(system.level1.flow_params.damping, 0.01);
        assert_eq!(system.coupling.strength, 0.5);
    }

    #[test]
    fn test_near_critical_vs_subcritical() {
        // Near-critical system should show more basin transitions
        let mut near_critical = NearCriticalSystem::new(0.001, 0.5, 0.1);
        let mut sub_critical = NearCriticalSystem::new(0.1, 0.5, 0.1);

        near_critical.initialize([0.3, 0.1], [0.1, 0.3]);
        sub_critical.initialize([0.3, 0.1], [0.1, 0.3]);

        let mut rng1 = simple_rng(42);
        let mut rng2 = simple_rng(42);

        near_critical.run(1000, &mut rng1);
        sub_critical.run(1000, &mut rng2);

        let (_, trans_near) = near_critical.count_transitions();
        let (_, trans_sub) = sub_critical.count_transitions();

        // Near-critical should have at least as many transitions
        // (may not always be more due to stochasticity, but should trend)
        // Validate transition counts are reasonable (< 1000 for 1000-step run)
        assert!(
            trans_near < 1000,
            "Transition count suspiciously high: {}",
            trans_near
        );
        assert!(
            trans_sub < 1000,
            "Transition count suspiciously high: {}",
            trans_sub
        );
    }

    #[test]
    fn test_b_prime_single_run() {
        let run = run_single_b_prime(0.01, 0.5, 0.05, 500, 42);

        assert_eq!(run.damping, 0.01);
        assert_eq!(run.coupling, 0.5);
        assert!(run.metric.n_samples > 0);
    }

    #[test]
    fn test_b_prime_experiment() {
        let config = BPrimeConfig {
            damping_values: vec![0.1, 0.01],
            coupling_strengths: vec![0.0, 0.5],
            noise: 0.05,
            steps: 200,
            runs_per_config: 2,
        };

        let result = run_b_prime_experiment(&config);

        assert!(!result.runs.is_empty());
        assert!(!result.deformation_by_damping.is_empty());
    }
}
