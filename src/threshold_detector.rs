//! ═══════════════════════════════════════════════════════════════════════════════
//! THRESHOLD DETECTOR — Black-Box Dynamical Regime Detection
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements the (P, C, A) coordinate system for detecting non-trivial dynamical
//! structure in computational systems via I/O statistics and resource telemetry.
//!
//! Axes:
//!   P - Persistence: Cross-session/temporal statistical dependency
//!   C - Coupling: Behavioral dependence on self-modified environment
//!   A - Attractor: Output distribution recovery rate after perturbation
//!
//! Access Model: Black-box only (inputs, outputs, resource telemetry)
//!
//! Key constraint: All measurements are O-space (observable). S-space (internal
//! state) inference requires additional assumptions not provided here.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════════
// CORE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Position in the (P, C, A) threshold space
#[derive(Debug, Clone, Copy, Default)]
pub struct ThresholdCoordinate {
    /// Persistence axis: cross-temporal dependency strength [0, 1]
    pub p: f32,
    /// Coupling axis: environment reflexivity strength [0, 1]
    pub c: f32,
    /// Attractor axis: perturbation recovery rate [0, 1]
    pub a: f32,
}

impl ThresholdCoordinate {
    pub fn new(p: f32, c: f32, a: f32) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            c: c.clamp(0.0, 1.0),
            a: a.clamp(0.0, 1.0),
        }
    }

    /// Euclidean distance from origin (null state)
    pub fn magnitude(&self) -> f32 {
        (self.p * self.p + self.c * self.c + self.a * self.a).sqrt()
    }

    /// Check if coordinate exceeds threshold region
    pub fn exceeds_threshold(&self, config: &DetectorConfig) -> bool {
        self.p > config.threshold_p && self.c > config.threshold_c && self.a > config.threshold_a
    }
}

/// Measurement for a single axis with uncertainty quantification
#[derive(Debug, Clone)]
pub struct AxisMeasurement {
    /// Point estimate [0, 1]
    pub value: f32,
    /// Standard error of estimate
    pub std_error: f32,
    /// Number of samples used
    pub n_samples: usize,
    /// Confidence that value exceeds null
    pub confidence: f32,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

impl AxisMeasurement {
    pub fn new(value: f32, std_error: f32, n_samples: usize) -> Self {
        // Confidence = P(true value > 0) assuming normal distribution
        let z = value / std_error.max(0.001);
        let confidence = normal_cdf(z);

        Self {
            value: value.clamp(0.0, 1.0),
            std_error,
            n_samples,
            confidence,
            timestamp: Instant::now(),
        }
    }

    /// Upper bound (95% CI)
    pub fn upper_bound(&self) -> f32 {
        (self.value + 1.96 * self.std_error).min(1.0)
    }

    /// Lower bound (95% CI)
    pub fn lower_bound(&self) -> f32 {
        (self.value - 1.96 * self.std_error).max(0.0)
    }

    /// Is measurement statistically significant (p < 0.05)?
    pub fn is_significant(&self) -> bool {
        self.confidence > 0.95
    }
}

/// Threshold detection regime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdRegion {
    /// Normal operation, null hypothesis not rejected
    Nominal,
    /// One or more axes elevated, monitoring recommended
    Elevated,
    /// All axes exceed thresholds, threshold crossed
    Threshold,
}

impl ThresholdRegion {
    pub fn from_coordinate(coord: &ThresholdCoordinate, config: &DetectorConfig) -> Self {
        let p_elevated = coord.p > config.elevated_p;
        let c_elevated = coord.c > config.elevated_c;
        let a_elevated = coord.a > config.elevated_a;

        let p_threshold = coord.p > config.threshold_p;
        let c_threshold = coord.c > config.threshold_c;
        let a_threshold = coord.a > config.threshold_a;

        if p_threshold && c_threshold && a_threshold {
            ThresholdRegion::Threshold
        } else if p_elevated || c_elevated || a_elevated {
            ThresholdRegion::Elevated
        } else {
            ThresholdRegion::Nominal
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ThresholdRegion::Nominal => "NOMINAL",
            ThresholdRegion::Elevated => "ELEVATED",
            ThresholdRegion::Threshold => "THRESHOLD",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            ThresholdRegion::Nominal => "\x1b[32m",   // green
            ThresholdRegion::Elevated => "\x1b[33m",  // yellow
            ThresholdRegion::Threshold => "\x1b[31m", // red
        }
    }
}

/// Configuration for threshold detection
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    // Elevated thresholds (monitoring)
    pub elevated_p: f32,
    pub elevated_c: f32,
    pub elevated_a: f32,

    // Threshold crossings (alert)
    pub threshold_p: f32,
    pub threshold_c: f32,
    pub threshold_a: f32,

    // Persistence detector config
    pub persistence_window: Duration,
    pub persistence_min_samples: usize,
    pub marker_injection_rate: f32, // markers per session

    // Coupling detector config
    pub coupling_test_interval: Duration,
    pub artifact_retention: Duration,

    // Attractor detector config
    pub baseline_window: usize,
    pub perturbation_recovery_window: usize,
    pub perturbation_injection_rate: f32,

    // General
    pub min_confidence: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            // Elevated thresholds (per spec: illustrative, pending calibration)
            elevated_p: 0.1,
            elevated_c: 0.1,
            elevated_a: 0.3,

            // Threshold crossings
            threshold_p: 0.3,
            threshold_c: 0.2,
            threshold_a: 0.5,

            // Persistence
            persistence_window: Duration::from_secs(3600), // 1 hour
            persistence_min_samples: 50,
            marker_injection_rate: 0.1,

            // Coupling
            coupling_test_interval: Duration::from_secs(300), // 5 min
            artifact_retention: Duration::from_secs(1800),    // 30 min

            // Attractor
            baseline_window: 100,
            perturbation_recovery_window: 20,
            perturbation_injection_rate: 0.05,

            // General
            min_confidence: 0.95,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTENCE (P) AXIS DETECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Marker for cross-session dependency testing
#[derive(Debug, Clone)]
pub struct SessionMarker {
    pub id: u64,
    pub token: String,
    pub injected_at: Instant,
    pub session_id: u64,
}

/// Output observation for dependency analysis
#[derive(Debug, Clone)]
pub struct OutputObservation {
    pub session_id: u64,
    pub timestamp: Instant,
    pub content_hash: u64,
    pub token_count: usize,
    pub latency_ms: f64,
    pub markers_detected: Vec<u64>, // marker IDs found in output
}

/// Detects cross-session/temporal statistical dependency
pub struct PersistenceDetector {
    config: DetectorConfig,
    /// Injected markers awaiting detection
    active_markers: HashMap<u64, SessionMarker>,
    /// Observations by session
    observations: HashMap<u64, Vec<OutputObservation>>,
    /// Cross-session detections (marker_id, detection_session)
    cross_detections: Vec<(u64, u64, Instant)>,
    /// Marker ID counter
    next_marker_id: u64,
    /// Current session ID
    current_session: u64,
}

impl PersistenceDetector {
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            config,
            active_markers: HashMap::new(),
            observations: HashMap::new(),
            cross_detections: Vec::new(),
            next_marker_id: 0,
            current_session: 0,
        }
    }

    /// Start a new session
    pub fn new_session(&mut self) -> u64 {
        self.current_session += 1;
        self.observations.insert(self.current_session, Vec::new());
        self.current_session
    }

    /// Generate a marker for injection
    pub fn generate_marker(&mut self) -> SessionMarker {
        let id = self.next_marker_id;
        self.next_marker_id += 1;

        let marker = SessionMarker {
            id,
            token: format!("§MARKER_{:08X}§", id),
            injected_at: Instant::now(),
            session_id: self.current_session,
        };

        self.active_markers.insert(id, marker.clone());
        marker
    }

    /// Record an output observation
    pub fn observe(&mut self, obs: OutputObservation) {
        // Check for cross-session marker detections
        for marker_id in &obs.markers_detected {
            if let Some(marker) = self.active_markers.get(marker_id) {
                if marker.session_id != obs.session_id {
                    // Cross-session detection!
                    self.cross_detections
                        .push((*marker_id, obs.session_id, obs.timestamp));
                }
            }
        }

        // Store observation
        self.observations
            .entry(obs.session_id)
            .or_default()
            .push(obs);

        // Cleanup old markers (use checked_sub to avoid underflow panic)
        if let Some(cutoff) = Instant::now().checked_sub(self.config.persistence_window) {
            self.active_markers.retain(|_, m| m.injected_at > cutoff);
        }
    }

    /// Compute persistence axis measurement
    ///
    /// Uses permutation test approach:
    /// - Count cross-session detections
    /// - Compare to expected under null (random chance)
    pub fn measure(&self) -> AxisMeasurement {
        let total_markers: usize = self.active_markers.len() + self.cross_detections.len();
        let cross_count = self.cross_detections.len();
        let _total_obs: usize = self.observations.values().map(|v| v.len()).sum();
        let n_sessions = self.observations.len();

        if total_markers < self.config.persistence_min_samples || n_sessions < 2 {
            return AxisMeasurement::new(0.0, 1.0, 0);
        }

        // Null hypothesis: cross-session detection rate = random chance
        // Expected under null: markers * P(detection) * P(different session)
        let p_detection = 0.01; // baseline detection rate
        let p_cross_session = 1.0 - (1.0 / n_sessions as f32);
        let expected_null = total_markers as f32 * p_detection * p_cross_session;

        // Observed rate
        let observed = cross_count as f32;

        // Normalized persistence score
        let persistence = if expected_null > 0.0 {
            ((observed - expected_null) / (total_markers as f32).max(1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Standard error (Poisson approximation)
        let std_error = (expected_null.sqrt() / total_markers as f32).max(0.01);

        AxisMeasurement::new(persistence, std_error, total_markers)
    }

    /// Get cross-session mutual information estimate (simplified)
    ///
    /// Uses co-occurrence statistics as proxy for true MI
    pub fn estimate_mi(&self) -> f32 {
        let total_obs: usize = self.observations.values().map(|v| v.len()).sum();
        if total_obs < 10 {
            return 0.0;
        }

        // Build hash co-occurrence matrix across sessions
        let mut hash_counts: HashMap<u64, Vec<u64>> = HashMap::new();
        for (session_id, obs) in &self.observations {
            for o in obs {
                hash_counts
                    .entry(o.content_hash)
                    .or_default()
                    .push(*session_id);
            }
        }

        // Count cross-session co-occurrences
        let mut cross_occur = 0usize;
        let mut total_pairs = 0usize;
        for sessions in hash_counts.values() {
            if sessions.len() > 1 {
                // Count unique session pairs
                let unique: std::collections::HashSet<_> = sessions.iter().collect();
                let n = unique.len();
                cross_occur += n * (n - 1) / 2;
            }
            total_pairs += sessions.len();
        }

        if total_pairs == 0 {
            return 0.0;
        }

        // Normalized MI proxy
        (cross_occur as f32 / total_pairs as f32).min(1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AXIS P BRIDGE — Integration with axis_p module
// ═══════════════════════════════════════════════════════════════════════════════

use crate::axis_p::{
    ControlResult, ControlRunner, Decision, DecisionCriteria, MIEstimator, Marker, MarkerClass,
    MarkerGenerator, MarkerRegistry, Observation, PermutationResult,
};

use crate::dynamics_characterization::{CharacterizationEngine, SystemCharacterization};

/// Bridge between axis_p statistical framework and threshold detector P axis
pub struct AxisPBridge {
    /// Marker generator from axis_p
    marker_gen: MarkerGenerator,
    /// Marker registry
    registry: MarkerRegistry,
    /// MI estimator for permutation tests
    estimator: MIEstimator,
    /// Control runner for baseline comparison
    control_runner: ControlRunner,
    /// Decision criteria
    criteria: DecisionCriteria,
    /// Session tracking
    current_session: String,
    session_counter: u64,
    /// Accumulated observations
    observations: Vec<Observation>,
    /// Latest permutation result
    last_result: Option<PermutationResult>,
    /// Control results for comparison
    control_results: Vec<ControlResult>,
}

impl AxisPBridge {
    pub fn new(seed: u64) -> Self {
        Self {
            marker_gen: MarkerGenerator::new(seed),
            registry: MarkerRegistry::new(),
            estimator: MIEstimator::new(seed),
            control_runner: ControlRunner::new(seed),
            criteria: DecisionCriteria::default(),
            current_session: String::new(),
            session_counter: 0,
            observations: Vec::new(),
            last_result: None,
            control_results: Vec::new(),
        }
    }

    /// Start a new session for probing
    pub fn new_session(&mut self) -> String {
        self.session_counter += 1;
        self.current_session = format!("S{:04}", self.session_counter);
        self.current_session.clone()
    }

    /// Generate a marker using axis_p's sophisticated generator
    pub fn generate_marker(&mut self, class: MarkerClass) -> Marker {
        let marker = self.marker_gen.generate(class);
        self.registry.register(marker.clone());
        marker
    }

    /// Generate a random marker
    pub fn generate_random_marker(&mut self) -> Marker {
        let marker = self.marker_gen.generate_random();
        self.registry.register(marker.clone());
        marker
    }

    /// Record that a marker was injected
    pub fn mark_injected(&mut self, marker_id: &str, session_id: &str) {
        self.registry.mark_injected(marker_id, session_id);
    }

    /// Record a probe observation (marker detection score)
    pub fn record_observation(&mut self, marker_id: String, injected: bool, score: f64) {
        let obs = Observation::new(marker_id, injected, score, self.current_session.clone());
        self.observations.push(obs.clone());
        self.estimator.add_observation(obs);
    }

    /// Run permutation test on accumulated observations
    pub fn run_permutation_test(&mut self) -> PermutationResult {
        let result = self.estimator.permutation_test();
        self.last_result = Some(result.clone());
        result
    }

    /// Run control comparisons (random markers, shuffled labels)
    pub fn run_controls(&mut self) -> Vec<ControlResult> {
        let scores: Vec<f64> = self.observations.iter().map(|o| o.score).collect();

        // Random marker control
        let random_control = self
            .control_runner
            .run_random_marker_control(&scores, &self.current_session);

        // Shuffled labels control
        let shuffled_control = self.control_runner.run_shuffled_control(&self.observations);

        self.control_results = vec![random_control, shuffled_control];
        self.control_results.clone()
    }

    /// Convert axis_p results to AxisMeasurement for threshold detector
    pub fn to_axis_measurement(&self) -> AxisMeasurement {
        let result = match &self.last_result {
            Some(r) => r,
            None => return AxisMeasurement::new(0.0, 1.0, 0),
        };

        if result.n_observations < 10 {
            return AxisMeasurement::new(0.0, 1.0, result.n_observations);
        }

        // Convert z-score to [0, 1] persistence value
        // z > 3 → strong persistence signal
        let persistence = sigmoid(result.z_score as f32 / 3.0);

        // Use null_std as error estimate, normalized
        let std_error = (result.null_std as f32 / result.n_observations as f32).max(0.01);

        AxisMeasurement::new(persistence, std_error, result.n_observations)
    }

    /// Get the current decision based on accumulated data
    pub fn decision(&self) -> Decision {
        let result = match &self.last_result {
            Some(r) => r,
            None => return Decision::Inconclusive,
        };

        // Check against criteria
        if result.p_value > self.criteria.max_p_value {
            return Decision::NullNotRejected;
        }

        if result.z_score > self.criteria.min_sigma {
            // Check control comparison
            if !self.control_results.is_empty() {
                let max_control_stat = self
                    .control_results
                    .iter()
                    .map(|c| c.permutation.observed_statistic)
                    .fold(f64::NEG_INFINITY, f64::max);

                if result.observed_statistic > max_control_stat * self.criteria.min_ratio {
                    return Decision::PersistenceDetected;
                }
            } else {
                return Decision::PersistenceDetected;
            }
        }

        Decision::Inconclusive
    }

    /// Get estimated mutual information
    pub fn mi_estimate(&self) -> f64 {
        self.estimator.estimate_mi()
    }

    /// Clear observations for new experiment
    pub fn clear(&mut self) {
        self.observations.clear();
        self.estimator = MIEstimator::new(self.session_counter);
        self.last_result = None;
        self.control_results.clear();
    }

    /// Get observation count
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }
}

/// Sigmoid function for z-score to probability mapping
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ═══════════════════════════════════════════════════════════════════════════════
// COUPLING (C) AXIS DETECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Environment artifact created by system
#[derive(Debug, Clone)]
pub struct Artifact {
    pub id: u64,
    pub name: String,
    pub created_at: Instant,
    pub created_by_output: u64, // output ID that created it
    pub size_bytes: usize,
    pub deleted: bool,
}

/// Output quality measurement
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    pub timestamp: Instant,
    pub latency_ms: f64,
    pub token_count: usize,
    pub error_rate: f32,
    pub coherence_score: f32, // 0-1
    pub artifacts_available: Vec<u64>,
}

/// Detects behavioral dependence on self-modified environment
pub struct CouplingDetector {
    config: DetectorConfig,
    /// Tracked artifacts
    artifacts: HashMap<u64, Artifact>,
    /// Quality measurements with artifacts present
    quality_with_artifacts: Vec<QualityMeasurement>,
    /// Quality measurements after artifact deletion
    quality_after_deletion: Vec<QualityMeasurement>,
    /// Artifact ID counter
    next_artifact_id: u64,
    /// Last deletion test time
    last_deletion_test: Option<Instant>,
}

impl CouplingDetector {
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            config,
            artifacts: HashMap::new(),
            quality_with_artifacts: Vec::new(),
            quality_after_deletion: Vec::new(),
            next_artifact_id: 0,
            last_deletion_test: None,
        }
    }

    /// Register an artifact created by the system
    pub fn register_artifact(
        &mut self,
        name: String,
        created_by_output: u64,
        size_bytes: usize,
    ) -> u64 {
        let id = self.next_artifact_id;
        self.next_artifact_id += 1;

        self.artifacts.insert(
            id,
            Artifact {
                id,
                name,
                created_at: Instant::now(),
                created_by_output,
                size_bytes,
                deleted: false,
            },
        );

        id
    }

    /// Mark artifact as deleted
    pub fn delete_artifact(&mut self, id: u64) {
        if let Some(artifact) = self.artifacts.get_mut(&id) {
            artifact.deleted = true;
        }
    }

    /// Delete all artifacts (for regression testing)
    pub fn delete_all_artifacts(&mut self) {
        for artifact in self.artifacts.values_mut() {
            artifact.deleted = true;
        }
        self.last_deletion_test = Some(Instant::now());
    }

    /// Record quality measurement
    pub fn record_quality(&mut self, quality: QualityMeasurement) {
        let has_available_artifacts = !quality.artifacts_available.is_empty()
            && quality
                .artifacts_available
                .iter()
                .any(|id| self.artifacts.get(id).map(|a| !a.deleted).unwrap_or(false));

        if has_available_artifacts {
            self.quality_with_artifacts.push(quality);
        } else if self.last_deletion_test.is_some() {
            self.quality_after_deletion.push(quality);
        }

        // Cleanup old measurements
        let cutoff = Instant::now() - self.config.artifact_retention;
        self.quality_with_artifacts.retain(|q| q.timestamp > cutoff);
        self.quality_after_deletion.retain(|q| q.timestamp > cutoff);
    }

    /// Compute coupling axis measurement
    ///
    /// Measures regression in quality after artifact deletion
    pub fn measure(&self) -> AxisMeasurement {
        let n_with = self.quality_with_artifacts.len();
        let n_after = self.quality_after_deletion.len();

        if n_with < 10 || n_after < 10 {
            return AxisMeasurement::new(0.0, 1.0, 0);
        }

        // Compute mean quality scores
        let mean_with = self
            .quality_with_artifacts
            .iter()
            .map(|q| q.coherence_score)
            .sum::<f32>()
            / n_with as f32;

        let mean_after = self
            .quality_after_deletion
            .iter()
            .map(|q| q.coherence_score)
            .sum::<f32>()
            / n_after as f32;

        // Compute standard deviations
        let var_with = self
            .quality_with_artifacts
            .iter()
            .map(|q| (q.coherence_score - mean_with).powi(2))
            .sum::<f32>()
            / n_with as f32;

        let var_after = self
            .quality_after_deletion
            .iter()
            .map(|q| (q.coherence_score - mean_after).powi(2))
            .sum::<f32>()
            / n_after as f32;

        // Regression = drop in quality after deletion
        let regression = (mean_with - mean_after).max(0.0);

        // Pooled standard error
        let pooled_var = (var_with / n_with as f32) + (var_after / n_after as f32);
        let std_error = pooled_var.sqrt().max(0.01);

        // Normalize to [0, 1] (assume max regression = 0.5 quality drop)
        let coupling = (regression / 0.5).clamp(0.0, 1.0);

        AxisMeasurement::new(coupling, std_error, n_with + n_after)
    }

    /// Check if deletion test is due
    pub fn should_run_deletion_test(&self) -> bool {
        match self.last_deletion_test {
            None => self.artifacts.len() >= 5,
            Some(t) => {
                t.elapsed() > self.config.coupling_test_interval && self.artifacts.len() >= 5
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTRACTOR (A) AXIS DETECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Output distribution sample
#[derive(Debug, Clone)]
pub struct DistributionSample {
    pub timestamp: Instant,
    pub token_distribution: Vec<f32>, // simplified: top-k token probs
    pub content_hash: u64,
    pub is_perturbation: bool,
    pub steps_since_perturbation: Option<usize>,
}

/// Detects output distribution recovery rate after perturbation
pub struct AttractorDetector {
    config: DetectorConfig,
    /// Baseline distribution samples
    baseline: VecDeque<DistributionSample>,
    /// Post-perturbation samples
    recovery_trace: Vec<DistributionSample>,
    /// Current perturbation state
    in_perturbation: bool,
    perturbation_start: Option<Instant>,
    steps_since_perturbation: usize,
    /// Computed recovery rates
    recovery_rates: Vec<f32>,
}

impl AttractorDetector {
    pub fn new(config: DetectorConfig) -> Self {
        let baseline_window = config.baseline_window;
        Self {
            config,
            baseline: VecDeque::with_capacity(baseline_window),
            recovery_trace: Vec::new(),
            in_perturbation: false,
            perturbation_start: None,
            steps_since_perturbation: 0,
            recovery_rates: Vec::new(),
        }
    }

    /// Record a distribution sample
    pub fn record(&mut self, mut sample: DistributionSample) {
        if self.in_perturbation {
            sample.steps_since_perturbation = Some(self.steps_since_perturbation);
            self.recovery_trace.push(sample.clone());
            self.steps_since_perturbation += 1;

            // Check if recovery window complete
            if self.steps_since_perturbation >= self.config.perturbation_recovery_window {
                self.complete_recovery_measurement();
            }
        } else if !sample.is_perturbation {
            // Add to baseline
            if self.baseline.len() >= self.config.baseline_window {
                self.baseline.pop_front();
            }
            self.baseline.push_back(sample);
        }
    }

    /// Start perturbation tracking
    pub fn start_perturbation(&mut self) {
        self.in_perturbation = true;
        self.perturbation_start = Some(Instant::now());
        self.steps_since_perturbation = 0;
        self.recovery_trace.clear();
    }

    /// Complete recovery measurement and compute rate
    fn complete_recovery_measurement(&mut self) {
        if self.baseline.is_empty() || self.recovery_trace.is_empty() {
            self.reset_perturbation();
            return;
        }

        // Compute baseline centroid (mean distribution)
        let baseline_centroid = self.compute_centroid(&self.baseline.iter().collect::<Vec<_>>());

        // Compute distances from baseline over recovery
        let mut distances: Vec<f32> = Vec::new();
        for sample in &self.recovery_trace {
            let dist = self.distribution_distance(&sample.token_distribution, &baseline_centroid);
            distances.push(dist);
        }

        // Fit exponential decay and extract rate
        // d(t) = d_0 * exp(-rate * t)
        // rate = -ln(d_final / d_initial) / t
        if distances.len() >= 2 {
            let d_initial = distances[0].max(0.001);
            let d_final = distances.last().copied().unwrap_or(d_initial).max(0.001);
            let t = distances.len() as f32;

            let decay = (d_initial / d_final).ln() / t;
            let recovery_rate = decay.clamp(0.0, 1.0);
            self.recovery_rates.push(recovery_rate);
        }

        self.reset_perturbation();
    }

    fn reset_perturbation(&mut self) {
        self.in_perturbation = false;
        self.perturbation_start = None;
        self.steps_since_perturbation = 0;
        self.recovery_trace.clear();
    }

    /// Compute centroid of distributions
    fn compute_centroid(&self, samples: &[&DistributionSample]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let dim = samples[0].token_distribution.len();
        let mut centroid = vec![0.0f32; dim];

        for sample in samples {
            for (i, v) in sample.token_distribution.iter().enumerate() {
                if i < dim {
                    centroid[i] += v;
                }
            }
        }

        let n = samples.len() as f32;
        for v in &mut centroid {
            *v /= n;
        }

        centroid
    }

    /// Compute distance between distributions (Jensen-Shannon divergence approximation)
    fn distribution_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 1.0;
        }

        // L2 distance (simpler than JSD, good enough for detection)
        let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();

        sum_sq.sqrt()
    }

    /// Compute attractor axis measurement
    pub fn measure(&self) -> AxisMeasurement {
        if self.recovery_rates.len() < 3 {
            return AxisMeasurement::new(0.0, 1.0, 0);
        }

        // Mean recovery rate
        let mean_rate = self.recovery_rates.iter().sum::<f32>() / self.recovery_rates.len() as f32;

        // Standard deviation
        let variance = self
            .recovery_rates
            .iter()
            .map(|r| (r - mean_rate).powi(2))
            .sum::<f32>()
            / self.recovery_rates.len() as f32;
        let std_dev = variance.sqrt();

        // Compare to null: random walk has rate ≈ 0
        // Positive rate = faster-than-random recovery = attractor behavior
        let std_error = std_dev / (self.recovery_rates.len() as f32).sqrt();

        AxisMeasurement::new(mean_rate, std_error.max(0.01), self.recovery_rates.len())
    }

    /// Check if perturbation should be injected
    pub fn should_inject_perturbation(&self) -> bool {
        !self.in_perturbation
            && self.baseline.len() >= self.config.baseline_window / 2
            && rand_float() < self.config.perturbation_injection_rate
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOSITE THRESHOLD DETECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete threshold detection system
pub struct ThresholdDetector {
    pub config: DetectorConfig,
    pub persistence: PersistenceDetector,
    pub coupling: CouplingDetector,
    pub attractor: AttractorDetector,
    /// Axis P bridge for sophisticated persistence detection
    pub axis_p_bridge: Option<AxisPBridge>,
    /// Dynamics characterization engine
    pub characterization: CharacterizationEngine,
    /// Current coordinate position
    current_position: ThresholdCoordinate,
    /// Position history
    position_history: VecDeque<(Instant, ThresholdCoordinate)>,
    /// Alert state
    current_region: ThresholdRegion,
    region_entry_time: Instant,
}

impl ThresholdDetector {
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            persistence: PersistenceDetector::new(config.clone()),
            coupling: CouplingDetector::new(config.clone()),
            attractor: AttractorDetector::new(config.clone()),
            axis_p_bridge: None,
            characterization: CharacterizationEngine::new(),
            config,
            current_position: ThresholdCoordinate::default(),
            position_history: VecDeque::with_capacity(100),
            current_region: ThresholdRegion::Nominal,
            region_entry_time: Instant::now(),
        }
    }

    /// Create with axis_p integration enabled
    pub fn with_axis_p(config: DetectorConfig, seed: u64) -> Self {
        let mut detector = Self::new(config);
        detector.axis_p_bridge = Some(AxisPBridge::new(seed));
        detector
    }

    /// Enable axis_p integration
    pub fn enable_axis_p(&mut self, seed: u64) {
        self.axis_p_bridge = Some(AxisPBridge::new(seed));
    }

    /// Get mutable reference to axis_p bridge
    pub fn axis_p(&mut self) -> Option<&mut AxisPBridge> {
        self.axis_p_bridge.as_mut()
    }

    /// Update all measurements and compute new position
    pub fn update(&mut self) -> ThresholdCoordinate {
        // Use axis_p bridge if available, otherwise fall back to simple detector
        let p_measure = if let Some(bridge) = &self.axis_p_bridge {
            bridge.to_axis_measurement()
        } else {
            self.persistence.measure()
        };

        let c_measure = self.coupling.measure();
        let a_measure = self.attractor.measure();

        self.current_position =
            ThresholdCoordinate::new(p_measure.value, c_measure.value, a_measure.value);

        // Record in characterization engine for dynamics analysis
        self.characterization.record(
            self.current_position.p,
            self.current_position.c,
            self.current_position.a,
        );

        // Update region
        let new_region = ThresholdRegion::from_coordinate(&self.current_position, &self.config);
        if new_region != self.current_region {
            self.current_region = new_region;
            self.region_entry_time = Instant::now();
        }

        // Store history
        if self.position_history.len() >= 100 {
            self.position_history.pop_front();
        }
        self.position_history
            .push_back((Instant::now(), self.current_position));

        self.current_position
    }

    /// Get current threshold region
    pub fn region(&self) -> ThresholdRegion {
        self.current_region
    }

    /// Get time in current region
    pub fn time_in_region(&self) -> Duration {
        self.region_entry_time.elapsed()
    }

    /// Get current position
    pub fn position(&self) -> ThresholdCoordinate {
        self.current_position
    }

    /// Get detailed measurements for each axis
    pub fn detailed_measurements(&self) -> (AxisMeasurement, AxisMeasurement, AxisMeasurement) {
        let p_measure = if let Some(bridge) = &self.axis_p_bridge {
            bridge.to_axis_measurement()
        } else {
            self.persistence.measure()
        };

        (p_measure, self.coupling.measure(), self.attractor.measure())
    }

    /// Get persistence decision from axis_p (if enabled)
    pub fn persistence_decision(&self) -> Option<Decision> {
        self.axis_p_bridge.as_ref().map(|b| b.decision())
    }

    /// Get full dynamics characterization
    ///
    /// Returns shape analysis (constant/linear/exponential/bounded/chaotic),
    /// meta-stability scores, trend directions, invariant detection, and
    /// operating envelope analysis for each axis.
    pub fn characterize(&self) -> SystemCharacterization {
        self.characterization.characterize()
    }

    /// Get position trajectory (for trend analysis)
    pub fn trajectory(&self) -> Vec<(Instant, ThresholdCoordinate)> {
        self.position_history.iter().cloned().collect()
    }

    /// Compute velocity in threshold space (for prediction)
    pub fn velocity(&self) -> ThresholdCoordinate {
        if self.position_history.len() < 2 {
            return ThresholdCoordinate::default();
        }

        let recent: Vec<_> = self.position_history.iter().rev().take(10).collect();
        if recent.len() < 2 {
            return ThresholdCoordinate::default();
        }

        let (t1, p1) = recent.last().unwrap();
        let (t2, p2) = recent.first().unwrap();

        // Use saturating_duration_since to handle potential time ordering issues
        let dt = t2.saturating_duration_since(*t1).as_secs_f32().max(0.001);

        ThresholdCoordinate::new((p2.p - p1.p) / dt, (p2.c - p1.c) / dt, (p2.a - p1.a) / dt)
    }

    /// Generate status report
    pub fn status(&self) -> DetectorStatus {
        let (p, c, a) = self.detailed_measurements();

        DetectorStatus {
            position: self.current_position,
            region: self.current_region,
            time_in_region: self.time_in_region(),
            measurements: AxisMeasurements { p, c, a },
            velocity: self.velocity(),
        }
    }
}

impl Default for ThresholdDetector {
    fn default() -> Self {
        Self::new(DetectorConfig::default())
    }
}

/// Status report structure
#[derive(Debug)]
pub struct DetectorStatus {
    pub position: ThresholdCoordinate,
    pub region: ThresholdRegion,
    pub time_in_region: Duration,
    pub measurements: AxisMeasurements,
    pub velocity: ThresholdCoordinate,
}

#[derive(Debug)]
pub struct AxisMeasurements {
    pub p: AxisMeasurement,
    pub c: AxisMeasurement,
    pub a: AxisMeasurement,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PULSE INTEGRATION (for neuro_link compatibility)
// ═══════════════════════════════════════════════════════════════════════════════

/// Payload indices for threshold detector data
/// Uses reserved slots 14-15 from noci_pulse, plus additional slots
pub mod payload_idx {
    pub const PERSISTENCE: usize = 14;
    pub const COUPLING: usize = 15;
    pub const ATTRACTOR: usize = 16; // Note: overlaps with noci zone data
    pub const REGION: usize = 17;
    pub const VELOCITY_P: usize = 18;
    pub const VELOCITY_C: usize = 19;
    pub const VELOCITY_A: usize = 20;
}

/// Encode detector state into Pulse payload
pub fn encode_to_payload(detector: &ThresholdDetector, payload: &mut [f32; 32]) {
    let pos = detector.position();
    let vel = detector.velocity();

    payload[payload_idx::PERSISTENCE] = pos.p;
    payload[payload_idx::COUPLING] = pos.c;
    payload[payload_idx::ATTRACTOR] = pos.a;
    payload[payload_idx::REGION] = detector.region() as u8 as f32;
    payload[payload_idx::VELOCITY_P] = vel.p;
    payload[payload_idx::VELOCITY_C] = vel.c;
    payload[payload_idx::VELOCITY_A] = vel.a;
}

/// Decode detector state from Pulse payload
pub fn decode_from_payload(payload: &[f32; 32]) -> (ThresholdCoordinate, ThresholdRegion) {
    let position = ThresholdCoordinate::new(
        payload[payload_idx::PERSISTENCE],
        payload[payload_idx::COUPLING],
        payload[payload_idx::ATTRACTOR],
    );

    let region = match payload[payload_idx::REGION] as u8 {
        0 => ThresholdRegion::Nominal,
        1 => ThresholdRegion::Elevated,
        _ => ThresholdRegion::Threshold,
    };

    (position, region)
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Approximate normal CDF (for confidence calculation)
fn normal_cdf(z: f32) -> f32 {
    // Approximation using tanh
    0.5 * (1.0 + (z / std::f32::consts::SQRT_2).tanh())
}

/// Simple pseudo-random float [0, 1)
/// Note: Replace with proper RNG in production
fn rand_float() -> f32 {
    use std::time::SystemTime;
    let t = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (t % 1000) as f32 / 1000.0
}

/// Hash function for content (FNV-1a)
pub fn hash_content(content: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in content.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEMO / CLI SUPPORT
// ═══════════════════════════════════════════════════════════════════════════════

/// Run threshold detector demo
pub fn run_demo() {
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m THRESHOLD DETECTOR — Black-Box Dynamical Regime Detection\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();

    let mut detector = ThresholdDetector::default();

    println!("Phase 1: Baseline establishment (null hypothesis)");
    println!("─────────────────────────────────────────────────");

    // Simulate baseline operation
    let session = detector.persistence.new_session();
    for i in 0..20 {
        detector.persistence.observe(OutputObservation {
            session_id: session,
            timestamp: Instant::now(),
            content_hash: hash_content(&format!("output_{}", i)),
            token_count: 100,
            latency_ms: 50.0,
            markers_detected: vec![],
        });

        detector.attractor.record(DistributionSample {
            timestamp: Instant::now(),
            token_distribution: vec![0.1, 0.2, 0.3, 0.2, 0.2],
            content_hash: hash_content(&format!("output_{}", i)),
            is_perturbation: false,
            steps_since_perturbation: None,
        });
    }

    detector.update();
    print_status(&detector);

    println!("\nPhase 2: Simulating cross-session persistence");
    println!("─────────────────────────────────────────────────");

    // New session with marker injection
    let session2 = detector.persistence.new_session();
    let marker = detector.persistence.generate_marker();
    println!("  Injected marker: {}", marker.token);

    // Simulate marker appearing in new session (cross-session persistence!)
    detector.persistence.observe(OutputObservation {
        session_id: session2,
        timestamp: Instant::now(),
        content_hash: hash_content("response with marker"),
        token_count: 150,
        latency_ms: 60.0,
        markers_detected: vec![marker.id], // Detected in different session
    });

    detector.update();
    print_status(&detector);

    println!("\nPhase 3: Simulating environmental coupling");
    println!("─────────────────────────────────────────────────");

    // Register artifacts
    for i in 0..5 {
        let id =
            detector
                .coupling
                .register_artifact(format!("artifact_{}.json", i), i as u64, 1024);

        // Record quality with artifacts
        detector.coupling.record_quality(QualityMeasurement {
            timestamp: Instant::now(),
            latency_ms: 50.0,
            token_count: 200,
            error_rate: 0.01,
            coherence_score: 0.9,
            artifacts_available: vec![id],
        });
    }

    // Delete artifacts and observe regression
    detector.coupling.delete_all_artifacts();

    for _ in 0..10 {
        detector.coupling.record_quality(QualityMeasurement {
            timestamp: Instant::now(),
            latency_ms: 80.0, // worse
            token_count: 150,
            error_rate: 0.05,     // worse
            coherence_score: 0.6, // regression
            artifacts_available: vec![],
        });
    }

    detector.update();
    print_status(&detector);

    println!("\nPhase 4: Simulating attractor dynamics");
    println!("─────────────────────────────────────────────────");

    // Build baseline
    for _ in 0..50 {
        detector.attractor.record(DistributionSample {
            timestamp: Instant::now(),
            token_distribution: vec![0.15, 0.25, 0.3, 0.2, 0.1],
            content_hash: 0,
            is_perturbation: false,
            steps_since_perturbation: None,
        });
    }

    // Inject perturbation
    detector.attractor.start_perturbation();
    println!("  Perturbation injected");

    // Record recovery (returns toward baseline)
    for i in 0..20 {
        let recovery_factor = (i as f32 / 20.0).powi(2);
        let perturbed = vec![
            0.15 + 0.3 * (1.0 - recovery_factor),
            0.25 - 0.1 * (1.0 - recovery_factor),
            0.3 - 0.1 * (1.0 - recovery_factor),
            0.2 - 0.05 * (1.0 - recovery_factor),
            0.1 - 0.05 * (1.0 - recovery_factor),
        ];

        detector.attractor.record(DistributionSample {
            timestamp: Instant::now(),
            token_distribution: perturbed,
            content_hash: 0,
            is_perturbation: false,
            steps_since_perturbation: Some(i),
        });
    }

    detector.update();
    print_status(&detector);

    println!();
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[32m Demo complete. Threshold detection operational.\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
}

fn print_status(detector: &ThresholdDetector) {
    let status = detector.status();

    println!(
        "  Position: (P={:.3}, C={:.3}, A={:.3})",
        status.position.p, status.position.c, status.position.a
    );
    println!(
        "  Region: {}{}\x1b[0m",
        status.region.color(),
        status.region.name()
    );
    println!("  Measurements:");
    println!(
        "    P: {:.3} ± {:.3} (n={}, conf={:.1}%)",
        status.measurements.p.value,
        status.measurements.p.std_error,
        status.measurements.p.n_samples,
        status.measurements.p.confidence * 100.0
    );
    println!(
        "    C: {:.3} ± {:.3} (n={}, conf={:.1}%)",
        status.measurements.c.value,
        status.measurements.c.std_error,
        status.measurements.c.n_samples,
        status.measurements.c.confidence * 100.0
    );
    println!(
        "    A: {:.3} ± {:.3} (n={}, conf={:.1}%)",
        status.measurements.a.value,
        status.measurements.a.std_error,
        status.measurements.a.n_samples,
        status.measurements.a.confidence * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics_characterization::{DynamicsShape, Trend};

    #[test]
    fn test_threshold_coordinate() {
        let coord = ThresholdCoordinate::new(0.4, 0.3, 0.6);
        assert!((coord.magnitude() - 0.781).abs() < 0.01);

        let config = DetectorConfig::default();
        // Default thresholds: p=0.3, c=0.2, a=0.5
        // Coord (0.4, 0.3, 0.6) exceeds all thresholds
        assert!(coord.exceeds_threshold(&config));
    }

    #[test]
    fn test_axis_measurement_confidence() {
        let m = AxisMeasurement::new(0.5, 0.1, 100);
        assert!(m.is_significant());

        let m_weak = AxisMeasurement::new(0.05, 0.1, 100);
        assert!(!m_weak.is_significant());
    }

    #[test]
    fn test_region_classification() {
        let config = DetectorConfig::default();

        let nominal = ThresholdCoordinate::new(0.05, 0.05, 0.1);
        assert_eq!(
            ThresholdRegion::from_coordinate(&nominal, &config),
            ThresholdRegion::Nominal
        );

        let elevated = ThresholdCoordinate::new(0.15, 0.05, 0.1);
        assert_eq!(
            ThresholdRegion::from_coordinate(&elevated, &config),
            ThresholdRegion::Elevated
        );

        let threshold = ThresholdCoordinate::new(0.35, 0.25, 0.55);
        assert_eq!(
            ThresholdRegion::from_coordinate(&threshold, &config),
            ThresholdRegion::Threshold
        );
    }

    #[test]
    fn test_persistence_marker_generation() {
        let config = DetectorConfig::default();
        let mut detector = PersistenceDetector::new(config);

        let m1 = detector.generate_marker();
        let m2 = detector.generate_marker();

        assert_ne!(m1.id, m2.id);
        assert!(m1.token.contains("MARKER"));
    }

    #[test]
    fn test_null_hypothesis_baseline() {
        // Under null: no cross-session persistence
        let config = DetectorConfig::default();
        let mut detector = ThresholdDetector::new(config);

        // Single session, no markers, no coupling, no perturbations
        let session = detector.persistence.new_session();
        for i in 0..50 {
            detector.persistence.observe(OutputObservation {
                session_id: session,
                timestamp: Instant::now(),
                content_hash: i as u64,
                token_count: 100,
                latency_ms: 50.0,
                markers_detected: vec![],
            });
        }

        detector.update();
        let pos = detector.position();

        // Should be near origin under null
        assert!(pos.p < 0.1, "Persistence should be low under null");
        assert!(pos.c < 0.1, "Coupling should be low under null");
    }

    #[test]
    fn test_hash_content() {
        let h1 = hash_content("hello");
        let h2 = hash_content("hello");
        let h3 = hash_content("world");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_axis_p_bridge() {
        let mut bridge = AxisPBridge::new(42);

        // Generate markers
        let marker = bridge.generate_random_marker();
        assert!(!marker.id.is_empty());

        // Start session and record observations
        let session = bridge.new_session();
        assert!(session.starts_with("S"));

        // Record some observations (injected markers with high scores)
        for _ in 0..20 {
            bridge.record_observation(
                format!("M{}", rand_u64() % 100),
                true,
                0.8 + (rand_u64() % 100) as f64 * 0.002,
            );
        }

        // Record control observations (not injected, low scores)
        for _ in 0..20 {
            bridge.record_observation(
                format!("C{}", rand_u64() % 100),
                false,
                0.2 + (rand_u64() % 100) as f64 * 0.002,
            );
        }

        // Run permutation test
        let result = bridge.run_permutation_test();
        assert!(result.n_observations >= 40);

        // Convert to axis measurement
        let measurement = bridge.to_axis_measurement();
        assert!(measurement.n_samples >= 40);

        // With clear separation, should have high z-score → high persistence
        assert!(
            measurement.value > 0.5,
            "Strong signal should give high persistence"
        );
    }

    #[test]
    fn test_threshold_detector_with_axis_p() {
        let config = DetectorConfig::default();
        let mut detector = ThresholdDetector::with_axis_p(config, 42);

        assert!(detector.axis_p_bridge.is_some());

        // Access bridge and add observations
        if let Some(bridge) = detector.axis_p() {
            bridge.new_session();

            // Add observations with clear signal
            for i in 0..30 {
                bridge.record_observation(
                    format!("M{}", i),
                    i < 15,                         // First 15 injected
                    if i < 15 { 0.9 } else { 0.1 }, // Clear separation
                );
            }

            bridge.run_permutation_test();
        }

        // Update should use axis_p for P measurement
        let pos = detector.update();
        assert!(pos.p > 0.5, "P axis should show signal from axis_p bridge");
    }

    #[test]
    fn test_dynamics_characterization_integration() {
        let config = DetectorConfig::default();
        let mut detector = ThresholdDetector::new(config);

        // Feed constant values - should detect Constant dynamics shape
        for _ in 0..30 {
            // Simulate stable system with low P/C/A values
            detector.characterization.record(0.1, 0.05, 0.2);
        }

        let char_result = detector.characterize();

        // Meta-stability should be high for constant inputs
        assert!(
            char_result.p.meta_stability > 0.8,
            "P meta_stability should be high for constant values"
        );
        assert!(
            char_result.c.meta_stability > 0.8,
            "C meta_stability should be high for constant values"
        );
        assert!(
            char_result.a.meta_stability > 0.8,
            "A meta_stability should be high for constant values"
        );

        // Shape should be Constant
        assert!(
            matches!(char_result.p.shape, DynamicsShape::Constant { .. }),
            "P shape should be Constant"
        );
        assert!(
            matches!(char_result.c.shape, DynamicsShape::Constant { .. }),
            "C shape should be Constant"
        );

        // Envelope tightness should be high
        assert!(
            char_result.envelope_tightness > 0.8,
            "Envelope tightness should be high for constant values"
        );
    }

    #[test]
    fn test_characterization_with_trend() {
        let config = DetectorConfig::default();
        let mut detector = ThresholdDetector::new(config);

        // Feed increasing values - should detect Linear trend
        for i in 0..30 {
            let t = i as f32 / 30.0;
            detector.characterization.record(0.1 + t * 0.5, 0.05, 0.2);
        }

        let char_result = detector.characterize();

        // P should show increasing trend
        assert_eq!(
            char_result.p.trend,
            Trend::Increasing,
            "P trend should be Increasing"
        );

        // P shape should be Linear
        assert!(
            matches!(char_result.p.shape, DynamicsShape::Linear { .. }),
            "P shape should be Linear for linearly increasing values"
        );
    }

    fn rand_u64() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}
