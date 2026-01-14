//! B′ Experiment: Near-Critical System Test
//!
//! Tests whether basin deformation occurs when the spectral gap γ → 0.

use fractal::coupling::b_prime::{
    print_b_prime_results, run_b_prime_experiment, BPrimeConfig, BPrimeVerdict,
};

fn main() {
    let sep = "=".repeat(79);

    println!("{}", sep);
    println!(" B′ HYPOTHESIS TEST");
    println!("{}", sep);
    println!();
    println!("Hypothesis B′: Basin deformation IS possible when no-go conditions violated.");
    println!();
    println!("Test: Violate 'strong dissipation' by setting γ → 0 (near-critical system).");
    println!();
    println!("Prediction: If B′ correct, deformation rate should INCREASE as γ → 0.");
    println!();

    // Test the sub-critical vs supercritical boundary
    // Key: coupling strength ‖C‖ vs damping γ
    // Sub-critical: ‖C‖ < γ → no deformation expected
    // Supercritical: ‖C‖ > γ → deformation possible
    let config = BPrimeConfig {
        damping_values: vec![0.2, 0.1, 0.05, 0.02, 0.01], // γ values
        coupling_strengths: vec![0.0, 0.05, 0.1, 0.15, 0.2, 0.3], // Weaker coupling
        noise: 0.01,
        steps: 3000,
        runs_per_config: 5,
    };

    println!("Configuration:");
    println!("  Damping values (γ): {:?}", config.damping_values);
    println!("  Coupling strengths: {:?}", config.coupling_strengths);
    println!("  Noise: {}", config.noise);
    println!("  Steps per run: {}", config.steps);
    println!("  Runs per config: {}", config.runs_per_config);
    println!();

    println!("Running experiment...");
    println!();

    let result = run_b_prime_experiment(&config);

    print_b_prime_results(&result);

    // Additional analysis: interaction between damping and coupling
    println!("{}", sep);
    println!(" INTERACTION ANALYSIS: Damping × Coupling");
    println!("{}", sep);
    println!();

    // Group runs by (damping, coupling) and show deformation rate
    println!(
        "{:^10} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8}",
        "Damping", "C=0.0", "C=0.3", "C=0.5", "C=0.7", "C=1.0"
    );
    println!("{}", "-".repeat(79));

    for &damping in &config.damping_values {
        let mut row = format!("{:^10.4} |", damping);

        for &coupling in &config.coupling_strengths {
            let runs: Vec<_> = result
                .runs
                .iter()
                .filter(|r| {
                    (r.damping - damping).abs() < 0.0001 && (r.coupling - coupling).abs() < 0.0001
                })
                .collect();

            if runs.is_empty() {
                row.push_str(" ------ |");
            } else {
                let rate = runs.iter().filter(|r| r.deformation_detected).count() as f64
                    / runs.len() as f64;
                row.push_str(&format!(" {:>5.1}% |", rate * 100.0));
            }
        }

        println!("{}", row);
    }

    println!();

    // Final status
    println!("{}", sep);
    println!(" FINAL STATUS");
    println!("{}", sep);
    println!();

    match result.verdict {
        BPrimeVerdict::Confirmed => {
            println!("B′ CONFIRMED");
            println!();
            println!("Near-critical systems (γ → 0) DO show basin deformation under coupling.");
            println!("This validates the escape route identified in the no-go theorem.");
            println!();
            println!("Updated status table:");
            println!();
            println!("  | B′ (near-critical)     | ✓ Confirmed |");
        }
        BPrimeVerdict::PartiallyConfirmed => {
            println!("B′ PARTIALLY CONFIRMED");
            println!();
            println!("Some near-critical configurations show deformation.");
            println!("The escape route exists but may require specific conditions.");
            println!();
            println!("Updated status table:");
            println!();
            println!("  | B′ (near-critical)     | ~ Partial   |");
        }
        BPrimeVerdict::NotConfirmed => {
            println!("B′ NOT CONFIRMED");
            println!();
            println!("Even near-critical systems do not show basin deformation.");
            println!(
                "The escape route may require different conditions (adaptive π, kernel alignment)."
            );
            println!();
            println!("Updated status table:");
            println!();
            println!("  | B′ (near-critical)     | ✗ Failed    |");
        }
        BPrimeVerdict::InsufficientData => {
            println!("INSUFFICIENT DATA");
            println!();
            println!("Not enough runs to draw conclusions.");
        }
    }

    println!();

    // Summary statistics
    let total_runs = result.runs.len();
    let deformed_runs = result
        .runs
        .iter()
        .filter(|r| r.deformation_detected)
        .count();
    let total_transitions: usize = result.runs.iter().map(|r| r.basin_transitions).sum();

    println!("Summary statistics:");
    println!("  Total runs: {}", total_runs);
    println!(
        "  Runs with deformation: {} ({:.1}%)",
        deformed_runs,
        100.0 * deformed_runs as f64 / total_runs as f64
    );
    println!("  Total basin transitions: {}", total_transitions);

    if let Some(critical) = result.critical_damping {
        println!("  Critical damping: γ_c ≈ {:.4}", critical);
    }

    println!();
}
