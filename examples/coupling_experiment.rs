//! Real Data Coupling Experiment
//!
//! Tests hypothesis (B): whether macro-level dynamics (thermal, vestibular)
//! systematically couple to micro-level basin geometry (animacy, latency).

use fractal::coupling::{run_comparative_experiment, run_real_data_experiment, CouplingVerdict};

fn main() {
    let sep = "=".repeat(79);
    let dash = "-".repeat(79);

    println!("{}", sep);
    println!(" COUPLING EXPERIMENT — Testing Hypothesis (B)");
    println!("{}", sep);
    println!();
    println!("Hypothesis (B): Cross-level basin coupling is real, systematic, parameter-linked");
    println!("Alternative (A): Mere analogy — same statistics, no structural coupling");
    println!("Alternative (C): Selection effect — only interesting runs show deformation");
    println!();

    // Parameters
    let n_observations = 1000;
    let noise = 0.1;
    let seeds = [42, 123, 456, 789, 999];

    println!("Parameters:");
    println!("  - Observations per run: {}", n_observations);
    println!("  - Noise level: {}", noise);
    println!("  - Seeds: {:?}", seeds);
    println!();

    println!("{}", sep);
    println!(" EXPERIMENT 1: Coupling Strength Sweep");
    println!("{}", sep);
    println!();

    let coupling_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

    println!(
        "{:^10} | {:^12} | {:^12} | {:^15} | {:^20}",
        "Coupling", "Cross-Corr", "Phase-Coh", "Fisher-Macro", "Verdict"
    );
    println!("{}", dash);

    for &strength in &coupling_strengths {
        let result = run_real_data_experiment(strength, n_observations, noise, 42);

        let verdict_str = match result.verdict {
            CouplingVerdict::EvidenceForB => "Evidence for (B)",
            CouplingVerdict::Inconclusive => "Inconclusive",
            CouplingVerdict::EvidenceAgainstB => "Against (B)",
            CouplingVerdict::InsufficientData => "Insufficient",
        };

        println!(
            "{:^10.2} | {:^12.4} | {:^12.4} | {:^15.6} | {:^20}",
            strength,
            result.cross_level_correlation,
            result.phase_coherence,
            result.macro_metric.fisher_curvature,
            verdict_str
        );
    }

    println!();
    println!("{}", sep);
    println!(" EXPERIMENT 2: Coupled vs Null (Multiple Seeds)");
    println!("{}", sep);
    println!();

    let test_strength = 0.7;
    println!("Testing coupling strength = {}", test_strength);
    println!();

    println!(
        "{:^6} | {:^12} | {:^12} | {:^12} | {:^12} | {:^10}",
        "Seed", "Coupled-Corr", "Null-Corr", "Delta-Corr", "GeomDelta", "Verdict"
    );
    println!("{}", dash);

    let mut evidence_for_b = 0;
    let mut inconclusive = 0;
    let mut evidence_against = 0;

    for &seed in &seeds {
        let (coupled, null, delta) =
            run_comparative_experiment(test_strength, n_observations, noise, seed);

        let corr_diff = coupled.cross_level_correlation - null.cross_level_correlation;

        let verdict_char = match coupled.verdict {
            CouplingVerdict::EvidenceForB => {
                evidence_for_b += 1;
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
            "{:^6} | {:^12.4} | {:^12.4} | {:^12.4} | {:^12.6} | {:^10}",
            seed,
            coupled.cross_level_correlation,
            null.cross_level_correlation,
            corr_diff,
            delta,
            verdict_char
        );
    }

    println!();
    println!("{}", sep);
    println!(" SUMMARY");
    println!("{}", sep);
    println!();
    println!(
        "Across {} runs with coupling strength {}:",
        seeds.len(),
        test_strength
    );
    println!("  - Evidence FOR (B):     {} runs", evidence_for_b);
    println!("  - Inconclusive:         {} runs", inconclusive);
    println!("  - Evidence AGAINST (B): {} runs", evidence_against);
    println!();

    // Final verdict
    if evidence_for_b > seeds.len() / 2 {
        println!("OVERALL VERDICT: [+] Evidence supports hypothesis (B)");
        println!("  -> Macro-level dynamics systematically couple to micro-level basin geometry");
    } else if evidence_against > seeds.len() / 2 {
        println!("OVERALL VERDICT: [-] Evidence against hypothesis (B)");
        println!("  -> Cross-level coupling appears to be analogy (A) or selection effect (C)");
    } else {
        println!("OVERALL VERDICT: [?] Inconclusive");
        println!("  -> More data or refined metrics needed");
    }
    println!();
}
