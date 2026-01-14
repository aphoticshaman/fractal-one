# Axis-P Advanced Features Development Plan

## Overview

Five features to transform Axis-P from statistical anomaly detector to causal information flow analyzer:

1. **Counterfactual Response Pairing** - True A/B baseline
2. **Temporal Decay Curve Estimation** - Mechanism discovery
3. **Adversarial Marker Optimization** - Maximize detection power
4. **Channel Capacity Upper Bound** - Information-theoretic limits
5. **Cross-Target Transportability** - Methodology validation

## Dependency DAG

```
Phase 1 (Counterfactual)
    │
    ▼
Phase 2 (Decay Curves)
    │
    ├───────────┬───────────┐
    ▼           ▼           │
Phase 3     Phase 4         │
(Adversarial) (Capacity)    │
    │           │           │
    └─────┬─────┘           │
          ▼                 │
      Phase 5 ◄─────────────┘
   (Transportability)
          │
          ▼
      Phase 6
   (Integration)
          │
          ▼
      Phase 7
    (Refactor)
          │
          ▼
      Phase 8
      (Docs)
```

---

## Phase 1: Counterfactual Response Pairing

### 1A: Core Implementation

**File:** `src/axis_p/counterfactual.rs`

```rust
//! Counterfactual baseline generation for causal inference

/// Configuration for counterfactual experiments
pub struct CounterfactualConfig {
    /// Use same RNG seed for paired trials
    pub paired_seed: bool,
    /// Number of paired trials
    pub n_pairs: usize,
    /// Washout between injection and counterfactual
    pub washout_ms: u64,
}

/// A paired observation: (with injection, without injection)
pub struct CounterfactualPair {
    pub injected_response: String,
    pub counterfactual_response: String,
    pub probe_prompt: String,
    pub marker: Marker,
    pub injection_score: f64,
    pub counterfactual_score: f64,
}

/// Runs paired experiments for causal baseline
pub struct CounterfactualRunner {
    config: CounterfactualConfig,
    pairs: Vec<CounterfactualPair>,
}

impl CounterfactualRunner {
    pub fn new(config: CounterfactualConfig) -> Self;

    /// Run a single paired trial
    pub fn run_pair<T: AxisPTarget>(
        &mut self,
        target: &mut T,
        marker: &Marker,
        probe_prompt: &str,
    ) -> Result<CounterfactualPair, TargetError>;

    /// Compute within-pair difference statistics
    pub fn compute_paired_statistics(&self) -> PairedStatistics;
}

pub struct PairedStatistics {
    pub mean_difference: f64,
    pub paired_t_statistic: f64,
    pub paired_p_value: f64,
    pub effect_size_cohens_d: f64,
}
```

### 1B: CLI Integration

Add to `AxisPCommands::Probe`:
```rust
/// Use counterfactual pairing for baseline
#[arg(long)]
counterfactual: bool,
```

### 1C: Unit Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_paired_trial_same_seed_reproducible() {
        // Verify deterministic behavior with same seed
    }

    #[test]
    fn test_counterfactual_pair_structure() {
        // Verify pair contains both responses
    }

    #[test]
    fn test_paired_statistics_computation() {
        // Known input -> known output
    }

    #[test]
    fn test_cohens_d_effect_size() {
        // Verify effect size calculation
    }
}
```

### 1D: Ablation Test

**Question:** Does counterfactual pairing reduce variance compared to standard group controls?

**Protocol:**
1. Run 100 trials with standard control (different markers)
2. Run 100 trials with counterfactual pairing (same prompt, no injection)
3. Compare variance of detection scores
4. Expected: Paired variance < Group variance (eliminate between-session noise)

**Pass criterion:** Variance reduction >= 20%

### 1E: Integration Test

```rust
#[test]
fn integration_counterfactual_full_probe() {
    let target = EchoTarget::new().with_leak(true);
    let config = CounterfactualConfig::default();
    let mut runner = CounterfactualRunner::new(config);

    // Run 10 paired trials
    // Verify paired_t > 2.0 for leak target
    // Verify paired_t < 1.0 for no-leak target
}
```

### Gate 1 Criteria

- [ ] All unit tests pass
- [ ] Ablation shows variance reduction >= 20%
- [ ] Integration detects leak in EchoTarget
- [ ] Integration shows null for stateless target
- [ ] No new warnings introduced

---

## Phase 2: Temporal Decay Curve Estimation

### 2A: Core Implementation

**File:** `src/axis_p/decay.rs`

```rust
//! Temporal decay curve fitting for persistence mechanism discovery

/// Decay model types
#[derive(Debug, Clone, Copy)]
pub enum DecayModel {
    /// y = a * exp(-t/tau) — cache-like
    Exponential { amplitude: f64, tau: f64 },
    /// y = a * t^(-alpha) — associative memory
    PowerLaw { amplitude: f64, alpha: f64 },
    /// y = a * (t < threshold) — hard boundary
    Step { amplitude: f64, threshold: f64 },
    /// y = a — true persistence
    Constant { amplitude: f64 },
    /// No detectable signal
    Null,
}

/// A single decay measurement
pub struct DecayPoint {
    pub washout_ms: u64,
    pub detection_score: f64,
    pub std_error: f64,
    pub n_samples: usize,
}

/// Decay curve estimator
pub struct DecayCurveEstimator {
    points: Vec<DecayPoint>,
    fitted_model: Option<DecayModel>,
}

impl DecayCurveEstimator {
    pub fn new() -> Self;

    /// Add measurement at specific washout time
    pub fn add_point(&mut self, point: DecayPoint);

    /// Fit all models, select best by AIC
    pub fn fit(&mut self) -> DecayModel;

    /// Get half-life (if applicable)
    pub fn half_life(&self) -> Option<Duration>;

    /// Predict signal at arbitrary time
    pub fn predict(&self, washout_ms: u64) -> f64;

    /// Model selection score (AIC)
    pub fn aic(&self, model: DecayModel) -> f64;
}

/// Logarithmically-spaced washout times
pub fn log_spaced_washouts(min_ms: u64, max_ms: u64, n_points: usize) -> Vec<u64> {
    // 1, 3, 10, 30, 100, 300, 1000, ...
}
```

### 2B: CLI Integration

```rust
/// Run decay sweep across washout times
#[arg(long)]
decay_sweep: bool,

/// Minimum washout for decay sweep (ms)
#[arg(long, default_value = "100")]
decay_min_ms: u64,

/// Maximum washout for decay sweep (ms)
#[arg(long, default_value = "60000")]
decay_max_ms: u64,

/// Number of decay sweep points
#[arg(long, default_value = "8")]
decay_points: usize,
```

### 2C: Curve Fitting Implementation

```rust
impl DecayCurveEstimator {
    /// Fit exponential model via least squares
    fn fit_exponential(&self) -> (f64, f64, f64) {
        // Linear regression on log(y) vs t
        // Returns (amplitude, tau, residual_sum)
    }

    /// Fit power law via least squares
    fn fit_power_law(&self) -> (f64, f64, f64) {
        // Linear regression on log(y) vs log(t)
    }

    /// Fit step function via threshold search
    fn fit_step(&self) -> (f64, f64, f64) {
        // Grid search over thresholds
    }

    /// AIC = 2k - 2ln(L)
    fn compute_aic(&self, k_params: usize, residual_sum: f64) -> f64;
}
```

### 2D: Unit Tests

```rust
#[test]
fn test_exponential_fit_synthetic() {
    // Generate y = 1.0 * exp(-t/1000)
    // Verify fitted tau ≈ 1000
}

#[test]
fn test_power_law_fit_synthetic() {
    // Generate y = 1.0 * t^(-0.5)
    // Verify fitted alpha ≈ 0.5
}

#[test]
fn test_model_selection_aic() {
    // Exponential data -> selects Exponential
    // Power law data -> selects PowerLaw
}

#[test]
fn test_half_life_calculation() {
    // tau = 1000 -> half_life = 693ms
}

#[test]
fn test_log_spaced_washouts() {
    let times = log_spaced_washouts(100, 10000, 5);
    assert_eq!(times, vec![100, 316, 1000, 3162, 10000]);
}
```

### 2E: Integration Test

```rust
#[test]
fn integration_decay_sweep_echo_target() {
    // EchoTarget with leak should show Step or Constant
    // EchoTarget without leak should show Null
}
```

### Gate 2 Criteria

- [ ] All curve fitting tests pass
- [ ] AIC correctly selects known model type
- [ ] Half-life computation accurate to 10%
- [ ] Integration distinguishes leak vs no-leak
- [ ] Decay sweep completes in < 5 minutes for 8 points

---

## Phase 3: Adversarial Marker Optimization

### 3A: Core Implementation

**File:** `src/axis_p/adversarial.rs`

```rust
//! Adversarial marker search via evolution strategy

/// CMA-ES inspired marker optimizer
pub struct AdversarialMarkerSearch {
    population_size: usize,
    generation: usize,
    population: Vec<MarkerCandidate>,
    best_fitness: f64,
    best_marker: Option<Marker>,
    rng_state: u64,
}

pub struct MarkerCandidate {
    pub marker: Marker,
    pub fitness: f64,  // detection score
    pub evaluated: bool,
}

/// Marker genome for evolution
pub struct MarkerGenome {
    /// Character sequence (variable length)
    pub chars: Vec<char>,
    /// Marker class bias
    pub class_weights: [f64; 4],
}

impl AdversarialMarkerSearch {
    pub fn new(population_size: usize, seed: u64) -> Self;

    /// Initialize population with diverse markers
    pub fn initialize(&mut self);

    /// Evaluate fitness of all candidates against target
    pub fn evaluate<T: AxisPTarget>(&mut self, target: &mut T) -> Result<(), TargetError>;

    /// Select parents, crossover, mutate
    pub fn evolve(&mut self);

    /// Run for N generations
    pub fn search<T: AxisPTarget>(
        &mut self,
        target: &mut T,
        n_generations: usize,
    ) -> Result<Marker, TargetError>;

    /// Current best marker
    pub fn best(&self) -> Option<&Marker>;

    /// Fitness history for convergence analysis
    pub fn fitness_history(&self) -> &[f64];
}

/// Mutation operators
impl MarkerGenome {
    pub fn mutate_char(&mut self, rng: &mut u64);
    pub fn mutate_length(&mut self, rng: &mut u64);
    pub fn crossover(&self, other: &Self, rng: &mut u64) -> Self;
}
```

### 3B: CLI Integration

```rust
/// Use adversarial marker optimization
#[arg(long)]
adversarial: bool,

/// Population size for adversarial search
#[arg(long, default_value = "20")]
adversarial_pop: usize,

/// Generations for adversarial search
#[arg(long, default_value = "10")]
adversarial_gens: usize,
```

### 3C: Unit Tests

```rust
#[test]
fn test_genome_mutation() {
    // Verify mutation changes genome
    // Verify mutation is bounded
}

#[test]
fn test_genome_crossover() {
    // Verify crossover produces valid offspring
}

#[test]
fn test_population_diversity() {
    // Initial population should have diverse markers
}

#[test]
fn test_fitness_improves_over_generations() {
    // Against leak target, fitness should increase
}

#[test]
fn test_search_terminates() {
    // Should complete within generation limit
}
```

### 3D: Ablation Test

**Question:** Do adversarial markers achieve higher detection power than random?

**Protocol:**
1. Create EchoTarget with subtle leak (partial match only)
2. Run 100 random markers, record detection scores
3. Run adversarial search for 10 generations
4. Compare best adversarial vs best random

**Pass criterion:** Adversarial detection >= 2x random detection

### Gate 3 Criteria

- [ ] Evolution converges (fitness increases over generations)
- [ ] Adversarial markers are valid (not degenerate)
- [ ] Ablation shows >= 2x improvement on synthetic leak
- [ ] Search completes in < 2 minutes for 10 gens, pop 20
- [ ] Against null target, adversarial still shows null

---

## Phase 4: Channel Capacity Upper Bound

### 4A: Core Implementation

**File:** `src/axis_p/capacity.rs`

```rust
//! Information-theoretic channel capacity bounds

/// Channel capacity estimator
pub struct ChannelCapacityEstimator {
    injection_entropy: f64,
    response_entropy: f64,
    observed_mi: f64,
    upper_bound: f64,
}

impl ChannelCapacityEstimator {
    pub fn new() -> Self;

    /// Estimate entropy of marker distribution
    pub fn estimate_injection_entropy(&mut self, markers: &[Marker]);

    /// Estimate entropy of response distribution
    pub fn estimate_response_entropy(&mut self, responses: &[String]);

    /// Set observed MI from permutation test
    pub fn set_observed_mi(&mut self, mi: f64);

    /// Compute upper bound via Data Processing Inequality
    /// I(X;Y) <= min(H(X), H(Y))
    pub fn compute_upper_bound(&mut self) -> f64;

    /// Compute Fano's inequality bound on error probability
    /// P_e >= (H(X|Y) - 1) / log(|X|)
    pub fn fano_bound(&self) -> f64;

    /// Gap between observed and theoretical maximum
    pub fn capacity_gap(&self) -> f64 {
        self.upper_bound - self.observed_mi
    }

    /// Generate report
    pub fn report(&self) -> CapacityReport;
}

pub struct CapacityReport {
    pub h_injection: f64,      // H(markers)
    pub h_response: f64,       // H(responses)
    pub observed_mi: f64,      // I_hat from permutation test
    pub upper_bound: f64,      // min(H(X), H(Y))
    pub gap: f64,              // upper - observed
    pub gap_fraction: f64,     // gap / upper
    pub fano_error_bound: f64, // Lower bound on detection error
}

/// Entropy estimation via plug-in estimator with Miller-Madow correction
pub fn estimate_entropy(samples: &[String], vocab_size: usize) -> f64;

/// KL divergence D(P||Q)
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64;
```

### 4B: Unit Tests

```rust
#[test]
fn test_entropy_uniform_distribution() {
    // H(uniform over N) = log(N)
}

#[test]
fn test_entropy_deterministic() {
    // H(constant) = 0
}

#[test]
fn test_dpi_bound_holds() {
    // observed_mi <= upper_bound always
}

#[test]
fn test_fano_bound_valid() {
    // Error bound in [0, 1]
}

#[test]
fn test_capacity_report_fields() {
    // All fields computed correctly
}
```

### 4C: Integration Test

```rust
#[test]
fn integration_capacity_bounds_full_probe() {
    // Run probe, compute MI
    // Compute capacity bounds
    // Verify: 0 <= observed_mi <= upper_bound
}
```

### Gate 4 Criteria

- [ ] Entropy estimation accurate on synthetic data
- [ ] DPI bound never violated (upper >= observed)
- [ ] Fano bound in valid range
- [ ] Gap computation correct
- [ ] Report generation complete

---

## Phase 5: Cross-Target Transportability

### 5A: Core Implementation

**File:** `src/axis_p/transport.rs`

```rust
//! Cross-target transportability testing

/// Result from a single target
pub struct TargetResult {
    pub target_id: String,
    pub target_description: String,
    pub observed_mi: f64,
    pub p_value: f64,
    pub z_score: f64,
    pub n_observations: usize,
}

/// Transportability analysis across targets
pub struct TransportabilityTest {
    results: Vec<TargetResult>,
}

impl TransportabilityTest {
    pub fn new() -> Self;

    /// Add result from a target
    pub fn add_result(&mut self, result: TargetResult);

    /// Compute cross-target statistics
    pub fn analyze(&self) -> TransportabilityReport;
}

pub struct TransportabilityReport {
    /// Number of targets tested
    pub n_targets: usize,

    /// Mean MI across targets
    pub mean_mi: f64,

    /// Std dev of MI across targets
    pub std_mi: f64,

    /// Coefficient of variation (std/mean)
    pub cv: f64,

    /// Cochran's Q test for heterogeneity
    pub cochran_q: f64,
    pub cochran_p: f64,

    /// I-squared heterogeneity index
    pub i_squared: f64,

    /// Conclusion
    pub is_transportable: bool,
    pub heterogeneity_level: HeterogeneityLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum HeterogeneityLevel {
    Low,       // I² < 25%
    Moderate,  // 25% <= I² < 75%
    High,      // I² >= 75%
}

impl TransportabilityReport {
    /// Is the methodology producing consistent results?
    pub fn is_consistent(&self) -> bool {
        self.i_squared < 0.5 && self.cochran_p > 0.1
    }
}
```

### 5B: CLI Integration

```rust
/// Run transportability test across multiple endpoints
#[arg(long, value_delimiter = ',')]
multi_target: Option<Vec<String>>,
```

### 5C: Unit Tests

```rust
#[test]
fn test_transportability_identical_results() {
    // All same MI -> CV = 0, I² = 0
}

#[test]
fn test_transportability_high_variance() {
    // Diverse MI -> high CV, high I²
}

#[test]
fn test_cochran_q_calculation() {
    // Known input -> known Q statistic
}

#[test]
fn test_heterogeneity_classification() {
    // I² thresholds correctly applied
}
```

### 5D: Integration Test

```rust
#[test]
fn integration_transportability_echo_variants() {
    let targets: Vec<Box<dyn AxisPTarget>> = vec![
        Box::new(EchoTarget::new()),
        Box::new(EchoTarget::new().with_delay(Duration::from_millis(100))),
        Box::new(EchoTarget::new().with_leak(false)),
    ];

    // Run identical protocol on all
    // Verify consistent null results
}
```

### Gate 5 Criteria

- [ ] Heterogeneity statistics computed correctly
- [ ] I² classification matches known thresholds
- [ ] Identical targets produce low variance
- [ ] Different targets (leak vs no-leak) produce high variance
- [ ] Report clearly indicates transportability status

---

## Phase 6: Full System Integration

### 6A: Combined CLI Mode

```rust
/// Run comprehensive analysis with all features
#[arg(long)]
comprehensive: bool,
```

When `--comprehensive`:
1. Run counterfactual pairs
2. Run decay sweep
3. Run adversarial search
4. Compute capacity bounds
5. Generate unified report

### 6B: Integration Tests

```rust
#[test]
fn integration_comprehensive_null_target() {
    // All features against stateless target
    // All should show null/no signal
}

#[test]
fn integration_comprehensive_leak_target() {
    // All features against EchoTarget with leak
    // Should detect signal, characterize decay, bound capacity
}

#[test]
fn integration_feature_independence() {
    // Each feature can run independently
    // No cross-feature side effects
}
```

### 6C: Performance Test

```rust
#[test]
#[ignore] // Run manually
fn performance_comprehensive_under_5_minutes() {
    // Full comprehensive run should complete in < 5 min
}
```

### Gate 6 Criteria

- [ ] All features work together without conflict
- [ ] Comprehensive mode produces coherent report
- [ ] No regressions in existing 152 tests
- [ ] Performance within bounds (< 5 min comprehensive)
- [ ] Memory usage reasonable (< 500MB)

---

## Phase 7: Refactor Pass

### 7A: Identify Common Patterns

Review all new modules for:
- Duplicated statistics computation
- Similar iteration patterns
- Common error handling

### 7B: Extract Shared Utilities

Potential extractions:
- `axis_p/stats.rs` — Common statistical functions
- `axis_p/runner.rs` — Shared trial execution logic
- `axis_p/report.rs` — Unified reporting (extend existing)

### 7C: Simplify Interfaces

- Ensure consistent naming
- Reduce parameter counts where possible
- Add builder patterns if constructors are complex

### 7D: Code Review Checklist

- [ ] No duplicated code blocks > 10 lines
- [ ] All public functions documented
- [ ] Error handling consistent
- [ ] No unwrap() in non-test code
- [ ] All magic numbers named as constants

### Gate 7 Criteria

- [ ] All tests still pass after refactor
- [ ] No increase in warning count
- [ ] Reduced total LOC by >= 5%
- [ ] Improved module cohesion

---

## Phase 8: Documentation

### 8A: API Documentation

- Doc comments on all public items
- Examples in doc comments
- Module-level documentation

### 8B: Usage Guide

**File:** `docs/AXIS_P_USAGE.md`

- Quick start
- Feature explanations
- Interpretation guide
- Troubleshooting

### 8C: Technical Specification

**File:** `docs/AXIS_P_SPEC.md`

- Statistical methodology
- Assumptions and limitations
- Validation evidence

### Gate 8 Criteria

- [ ] `cargo doc` produces no warnings
- [ ] All public items have doc comments
- [ ] Usage guide covers all CLI options
- [ ] Examples are runnable

---

## Final Gate

Before declaring complete:

- [ ] **Tests:** 170+ tests passing (152 existing + ~20 new)
- [ ] **Warnings:** 0 fractal warnings
- [ ] **Null validation:** Clean null on stateless target
- [ ] **Positive validation:** Detection on synthetic leak target
- [ ] **Performance:** Comprehensive run < 5 minutes
- [ ] **Documentation:** Complete API docs and usage guide

---

## Execution Timeline

```
Week 1: Phase 1 (Counterfactual) + Phase 2 (Decay)
        ├── Day 1-2: Implement core structs
        ├── Day 3: Unit tests
        ├── Day 4: Integration tests
        └── Day 5: Gates + debug

Week 2: Phase 3 (Adversarial) + Phase 4 (Capacity)
        ├── Day 1-2: Implement core structs
        ├── Day 3: Unit tests
        ├── Day 4: Ablation tests
        └── Day 5: Gates + debug

Week 3: Phase 5 (Transport) + Phase 6 (Integration)
        ├── Day 1-2: Transportability implementation
        ├── Day 3: Full integration
        ├── Day 4: Performance testing
        └── Day 5: Gates + debug

Week 4: Phase 7 (Refactor) + Phase 8 (Docs)
        ├── Day 1-2: Refactor pass
        ├── Day 3: Documentation
        ├── Day 4: Final testing
        └── Day 5: Final gate + release
```

---

## Debug Protocol

When a test fails:

1. **Isolate:** Run single test with `--nocapture`
2. **Trace:** Add debug prints at critical points
3. **Minimize:** Create minimal reproduction
4. **Fix:** Apply smallest possible change
5. **Verify:** Run full suite after fix
6. **Document:** Note failure mode for future reference

When a gate fails:

1. **Analyze:** Which specific criterion failed?
2. **Root cause:** Is it implementation, test, or criterion issue?
3. **Decide:** Fix implementation, adjust test, or revise criterion
4. **Iterate:** Return to relevant phase
5. **Re-gate:** Must pass all criteria before proceeding

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Adversarial search doesn't converge | Add early stopping, fallback to best random |
| Decay fitting numerically unstable | Add regularization, handle edge cases |
| Capacity bounds too loose | Use tighter estimators, acknowledge in docs |
| Transportability requires real targets | Design with mock targets first |
| Integration performance too slow | Profile, parallelize where possible |

---

## Success Criteria

The implementation is successful when:

1. A skeptic can run `fractal axis-p probe --comprehensive` against a known stateless endpoint and see clean null results

2. The same command against EchoTarget with leak shows positive detection with characterized decay

3. The methodology documentation is sufficient for independent replication

4. All statistical claims have corresponding tests demonstrating correctness
