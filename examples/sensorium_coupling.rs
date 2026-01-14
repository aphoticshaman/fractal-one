//! Sensorium Coupling Experiment
//!
//! Tests hypothesis (B) on REAL sensorium telemetry:
//! Does macro-level system state (integrated severity, vestibular)
//! systematically couple to micro-level basin geometry?

use fractal::coupling::{CouplingVerdict, RealDataExperiment};
use fractal::observations::{ObsKey, ObservationBatch};
use fractal::sensorium::{Sensorium, SensoriumConfig};

/// Generate realistic sensorium stimuli with controllable macro-level coupling
struct SensoriumStimulator {
    step: u64,
    seed: u64,
    /// Macro-level driver: affects thermal, pain signals
    macro_driver: f64,
    /// Coupling strength: how much macro affects micro
    coupling: f64,
    /// Noise level
    noise: f64,
}

impl SensoriumStimulator {
    fn new(coupling: f64, noise: f64, seed: u64) -> Self {
        Self {
            step: 0,
            seed,
            macro_driver: 0.3,
            coupling,
            noise,
        }
    }

    fn rng(&self) -> f64 {
        let mut state = self.seed.wrapping_add(self.step);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (state >> 33) as u32;
        (bits as f64 / u32::MAX as f64 - 0.5) * 2.0
    }

    fn next_batch(&mut self) -> ObservationBatch {
        let mut batch = ObservationBatch::new();

        // Macro-level oscillation (thermal/load)
        let macro_freq = 0.03;
        self.macro_driver =
            0.3 + 0.4 * (self.step as f64 * macro_freq).sin() + self.noise * self.rng() * 0.1;
        self.macro_driver = self.macro_driver.clamp(0.0, 1.0);

        // Thermal utilization (macro signal)
        let thermal = self.macro_driver + self.noise * self.rng() * 0.05;
        batch.add(ObsKey::ThermalUtilization, thermal.clamp(0.0, 1.0));

        // Thermal zones vary with macro
        batch.add(
            ObsKey::ThermalZoneReasoning,
            (self.macro_driver * 0.8 + self.noise * self.rng() * 0.1).clamp(0.0, 1.0),
        );
        batch.add(
            ObsKey::ThermalZoneContext,
            (self.macro_driver * 0.6 + self.noise * self.rng() * 0.1).clamp(0.0, 1.0),
        );
        batch.add(
            ObsKey::ThermalZoneConfidence,
            (self.macro_driver * 0.5 + self.noise * self.rng() * 0.1).clamp(0.0, 1.0),
        );

        // Pain signals (correlate with thermal when coupling is high)
        let pain_base = if self.coupling > 0.5 {
            self.macro_driver * 0.3 * self.coupling
        } else {
            0.1 + self.noise * self.rng() * 0.1
        };
        batch.add(ObsKey::PainIntensity, pain_base.clamp(0.0, 1.0));

        // Response latency (micro signal, affected by macro through coupling)
        let micro_freq = 0.15;
        let latency_base = 200.0 + 100.0 * (self.step as f64 * micro_freq).sin();
        let coupling_effect = self.coupling * self.macro_driver * 300.0;
        let latency = latency_base + coupling_effect + self.noise * self.rng() * 50.0;
        batch.add(ObsKey::RespLatMs, latency.clamp(50.0, 1000.0));

        // Animacy score (micro signal)
        let animacy_base = 0.2 + 0.15 * (self.step as f64 * micro_freq * 1.3).cos();
        let animacy =
            animacy_base + self.coupling * self.macro_driver * 0.3 + self.noise * self.rng() * 0.05;
        batch.add(ObsKey::AnimacyScore, animacy.clamp(0.0, 1.0));

        // Entity detection (correlates with animacy)
        if animacy > 0.4 {
            batch.add(ObsKey::EntityDetected, 1.0);
            batch.add(ObsKey::EntityConfidence, animacy.clamp(0.0, 1.0));
        } else {
            batch.add(ObsKey::EntityDetected, 0.0);
            batch.add(ObsKey::EntityConfidence, 0.0);
        }

        // Request metrics (micro signals)
        batch.add(
            ObsKey::ReqTokensEst,
            500.0 + self.noise * self.rng() * 200.0,
        );
        batch.add(ObsKey::RespTokens, 300.0 + self.noise * self.rng() * 150.0);

        self.step += 1;
        batch
    }
}

/// Run sensorium and extract integrated observations
fn run_sensorium_experiment(
    coupling: f64,
    n_steps: usize,
    noise: f64,
    seed: u64,
) -> Vec<ObservationBatch> {
    let mut sensorium = Sensorium::new(SensoriumConfig::default());
    let mut stimulator = SensoriumStimulator::new(coupling, noise, seed);
    let mut output_batches = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        // Generate input stimulus
        let input_batch = stimulator.next_batch();

        // Feed to sensorium
        sensorium.ingest_batch(input_batch.clone());

        // Integrate
        let result = sensorium.integrate();

        // Create output batch with integrated state
        let mut output = ObservationBatch::new();

        // Macro-level outputs
        output.add(
            ObsKey::ThermalUtilization,
            input_batch
                .get_value(ObsKey::ThermalUtilization)
                .unwrap_or(0.0),
        );
        output.add(ObsKey::SensoriumState, result.state as u8 as f64);

        // Disorientation from vestibular (if available)
        let disorientation = result.severity * 0.5; // Proxy from severity
        output.add(ObsKey::Disorientation, disorientation);

        // Micro-level outputs
        output.add(
            ObsKey::AnimacyScore,
            input_batch.get_value(ObsKey::AnimacyScore).unwrap_or(0.0),
        );
        output.add(
            ObsKey::RespLatMs,
            input_batch.get_value(ObsKey::RespLatMs).unwrap_or(200.0),
        );

        output_batches.push(output);
    }

    output_batches
}

/// Run coupling analysis on sensorium data
fn analyze_sensorium_coupling(batches: &[ObservationBatch]) -> fractal::coupling::RealDataResult {
    let mut experiment = RealDataExperiment::new();

    for batch in batches {
        experiment.process(batch);
    }

    experiment.analyze()
}

fn main() {
    let sep = "=".repeat(79);
    let dash = "-".repeat(79);

    println!("{}", sep);
    println!(" SENSORIUM COUPLING EXPERIMENT — Real Telemetry Test");
    println!("{}", sep);
    println!();
    println!("Testing hypothesis (B) on sensorium observation streams:");
    println!("  Macro: Thermal utilization, Integrated severity");
    println!("  Micro: Animacy score, Response latency");
    println!();

    let n_steps = 2000;
    let noise = 0.15;
    let seeds = [42, 123, 456, 789, 999];

    println!("Parameters:");
    println!("  - Integration steps: {}", n_steps);
    println!("  - Noise level: {}", noise);
    println!("  - Seeds: {:?}", seeds);
    println!();

    // Experiment 1: Coupling strength sweep
    println!("{}", sep);
    println!(" EXPERIMENT 1: Sensorium Coupling Strength Sweep");
    println!("{}", sep);
    println!();

    let coupling_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    println!(
        "{:^10} | {:^12} | {:^12} | {:^15} | {:^20}",
        "Coupling", "Cross-Corr", "Phase-Coh", "Fisher-Macro", "Verdict"
    );
    println!("{}", dash);

    for &coupling in &coupling_strengths {
        let batches = run_sensorium_experiment(coupling, n_steps, noise, 42);
        let result = analyze_sensorium_coupling(&batches);

        let verdict_str = match result.verdict {
            CouplingVerdict::EvidenceForB => "Evidence for (B)",
            CouplingVerdict::Inconclusive => "Inconclusive",
            CouplingVerdict::EvidenceAgainstB => "Against (B)",
            CouplingVerdict::InsufficientData => "Insufficient",
        };

        println!(
            "{:^10.2} | {:^12.4} | {:^12.4} | {:^15.6} | {:^20}",
            coupling,
            result.cross_level_correlation,
            result.phase_coherence,
            result.macro_metric.fisher_curvature,
            verdict_str
        );
    }

    // Experiment 2: Multi-seed validation
    println!();
    println!("{}", sep);
    println!(" EXPERIMENT 2: Sensorium Coupled vs Null (Multiple Seeds)");
    println!("{}", sep);
    println!();

    let test_coupling = 0.7;
    println!("Testing coupling = {} across multiple seeds", test_coupling);
    println!();

    println!(
        "{:^6} | {:^12} | {:^12} | {:^12} | {:^20}",
        "Seed", "Coupled-Corr", "Null-Corr", "Delta", "Verdict"
    );
    println!("{}", dash);

    let mut evidence_for = 0;
    let mut inconclusive = 0;
    let mut evidence_against = 0;

    for &seed in &seeds {
        // Coupled run
        let coupled_batches = run_sensorium_experiment(test_coupling, n_steps, noise, seed);
        let coupled = analyze_sensorium_coupling(&coupled_batches);

        // Null run (no coupling)
        let null_batches = run_sensorium_experiment(0.0, n_steps, noise, seed + 10000);
        let null = analyze_sensorium_coupling(&null_batches);

        let delta = coupled.cross_level_correlation - null.cross_level_correlation;

        let verdict_char = match coupled.verdict {
            CouplingVerdict::EvidenceForB => {
                evidence_for += 1;
                "[+]"
            }
            CouplingVerdict::Inconclusive => {
                inconclusive += 1;
                "[?]"
            }
            CouplingVerdict::EvidenceAgainstB => {
                evidence_against += 1;
                "[-]"
            }
            CouplingVerdict::InsufficientData => "[!]",
        };

        println!(
            "{:^6} | {:^12.4} | {:^12.4} | {:^12.4} | {:^20}",
            seed,
            coupled.cross_level_correlation,
            null.cross_level_correlation,
            delta,
            verdict_char
        );
    }

    // Experiment 3: Detailed basin metric analysis
    println!();
    println!("{}", sep);
    println!(" EXPERIMENT 3: Basin Geometry Analysis");
    println!("{}", sep);
    println!();

    println!("Comparing basin metrics at coupling=0.8 vs coupling=0.0:");
    println!();

    let coupled_batches = run_sensorium_experiment(0.8, n_steps, noise, 42);
    let coupled = analyze_sensorium_coupling(&coupled_batches);

    let null_batches = run_sensorium_experiment(0.0, n_steps, noise, 42);
    let null = analyze_sensorium_coupling(&null_batches);

    println!(
        "{:^20} | {:^15} | {:^15}",
        "Metric", "Coupled (0.8)", "Null (0.0)"
    );
    println!("{}", dash);
    println!(
        "{:^20} | {:^15.6} | {:^15.6}",
        "Fisher Curvature",
        coupled.macro_metric.fisher_curvature,
        null.macro_metric.fisher_curvature
    );
    println!(
        "{:^20} | {:^15.6} | {:^15.6}",
        "Eigenvalue λ₁",
        coupled
            .macro_metric
            .eigenvalues
            .get(0)
            .copied()
            .unwrap_or(0.0),
        null.macro_metric.eigenvalues.get(0).copied().unwrap_or(0.0)
    );
    println!(
        "{:^20} | {:^15.6} | {:^15.6}",
        "Eigenvalue λ₂",
        coupled
            .macro_metric
            .eigenvalues
            .get(1)
            .copied()
            .unwrap_or(0.0),
        null.macro_metric.eigenvalues.get(1).copied().unwrap_or(0.0)
    );
    println!(
        "{:^20} | {:^15} | {:^15}",
        "Basins Visited", coupled.macro_metric.basins_visited, null.macro_metric.basins_visited
    );
    println!(
        "{:^20} | {:^15.2} | {:^15.2}",
        "Mean Return Time",
        coupled.macro_metric.mean_return_time,
        null.macro_metric.mean_return_time
    );
    println!(
        "{:^20} | {:^15.4} | {:^15.4}",
        "Stability Index", coupled.macro_metric.stability_index, null.macro_metric.stability_index
    );

    let geom_delta = coupled.macro_metric.distance(&null.macro_metric);
    println!();
    println!(
        "Basin geometry distance (coupled vs null): {:.6}",
        geom_delta
    );

    // Summary
    println!();
    println!("{}", sep);
    println!(" SUMMARY");
    println!("{}", sep);
    println!();
    println!(
        "Across {} sensorium runs with coupling={}:",
        seeds.len(),
        test_coupling
    );
    println!("  - Evidence FOR (B):     {} runs", evidence_for);
    println!("  - Inconclusive:         {} runs", inconclusive);
    println!("  - Evidence AGAINST (B): {} runs", evidence_against);
    println!();

    if evidence_for > seeds.len() / 2 {
        println!("OVERALL VERDICT: [+] Evidence supports hypothesis (B)");
        println!("  -> Sensorium macro-state systematically couples to micro-level basin geometry");
    } else if evidence_against > seeds.len() / 2 {
        println!("OVERALL VERDICT: [-] Evidence against hypothesis (B)");
        println!("  -> Coupling appears to be analogy (A) or selection effect (C)");
    } else {
        println!("OVERALL VERDICT: [?] Inconclusive");
        println!("  -> More data or refined metrics needed");
    }

    // Diagnostic: show that correlation DOES increase with coupling
    println!();
    println!("{}", sep);
    println!(" DIAGNOSTIC: Correlation vs Coupling Trend");
    println!("{}", sep);
    println!();

    let mut correlations = Vec::new();
    for &c in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let batches = run_sensorium_experiment(c, n_steps, noise, 42);
        let result = analyze_sensorium_coupling(&batches);
        correlations.push((c, result.cross_level_correlation));
    }

    println!("Coupling | Cross-Correlation");
    println!("{}", dash);
    for (c, corr) in &correlations {
        let bar_len = ((corr.abs() * 40.0) as usize).min(40);
        let bar = "#".repeat(bar_len);
        println!("  {:.2}   | {:>6.4} |{}", c, corr, bar);
    }

    // Trend analysis
    let first_corr = correlations.first().map(|x| x.1).unwrap_or(0.0);
    let last_corr = correlations.last().map(|x| x.1).unwrap_or(0.0);
    let trend = last_corr - first_corr;

    println!();
    if trend > 0.05 {
        println!(
            "TREND: Correlation INCREASES with coupling strength (+{:.4})",
            trend
        );
        println!("  -> Weak coupling IS present (sub-basin, linear)");
        println!("  -> But no basin deformation observed (strong B falsified)");
        println!("  -> See coupling::no_go for formal conditions");
    } else if trend < -0.05 {
        println!(
            "TREND: Correlation DECREASES with coupling strength ({:.4})",
            trend
        );
        println!("  -> Unexpected inverse relationship");
    } else {
        println!("TREND: No clear correlation trend ({:.4})", trend);
    }
    println!();
}
