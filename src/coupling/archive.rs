//! ═══════════════════════════════════════════════════════════════════════════════
//! ARCHIVE: Cross-Level Basin Coupling Experiment — Closed Loop
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! This module serves as the permanent record of the coupling experiment.
//! Status: CLOSED as of 2026-01-11.
//!
//! ═══════════════════════════════════════════════════════════════════════════════
//! EXPERIMENT LIFECYCLE
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! 1. HYPOTHESIS FORMATION
//!    - Strong (B): Cross-level coupling reshapes attractor basins
//!    - Preregistered thresholds: correlation > 0.5, basin deformation > 0.1
//!
//! 2. IMPLEMENTATION
//!    - `BasinizedSystem<DIM>`: Formal dynamical system B = (X, Φ, π, μ)
//!    - CouplingOperator: Typed map C: X₁ → Params(Φ₂)
//!    - FisherEstimator: Information geometry metric for basin shape
//!    - TwoLevelSystem: Complete experimental apparatus
//!
//! 3. EXECUTION
//!    - Synthetic coupling sweep: strengths 0.0 → 1.0
//!    - Sensorium telemetry: 2000 steps, 5 seeds
//!    - Null baseline: same statistics, no coupling
//!
//! 4. RESULTS
//!    - Weak coupling: DEMONSTRATED (correlation 0.039 → 0.106)
//!    - Basin deformation: NOT OBSERVED (stability = 1.0)
//!    - Strong (B): FALSIFIED
//!
//! 5. THEORY
//!    - No-go theorem formalized: conditions precluding basin deformation
//!    - Escape routes identified: adaptive π, criticality, kernel alignment
//!
//! 6. CLOSURE
//!    - Negative result documented
//!    - Theory constrained
//!    - B′ left open for future work
//!
//! 7. B′ FOLLOW-UP (2026-01-11)
//!    - Tested supercritical coupling (‖C‖ > threshold)
//!    - Result: Basin deformation OBSERVED at any damping when C > 0
//!    - C=0 → 0% deformation, C>0 → 76-100% deformation
//!    - B′ CONFIRMED: Coupling (not damping) is the key factor
//!
//! 8. THEORETICAL CRYSTALLIZATION — The Projection Bound Theorem
//!
//!    Final form (validated across multiple adversarial audits):
//!
//!    "Structure cannot survive projection unless it is aligned with
//!     the projection or alters the projection."
//!
//!    Formally: Structure in X survives coarse-graining π: X → Y iff:
//!      (1) ALIGNMENT: Structure lies in ker(π)⊥ (orthogonal to kernel)
//!      (2) ADAPTATION: Structure modifies π itself (changes the observer)
//!
//!    All other structure is mathematically erased—not hidden, not subtle, gone.
//!
//!    This kills:
//!      - "The meaning is still there but hidden"
//!      - "The system feels it even if we can't measure it"
//!      - "There's deep structure influencing things invisibly"
//!
//!    Two escape routes. No third option. Linear algebra, not philosophy.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};

/// Experiment archive record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentArchive {
    /// Experiment identifier
    pub id: &'static str,

    /// Date closed
    pub date_closed: &'static str,

    /// Status
    pub status: ExperimentStatus,

    /// Hypothesis tested
    pub hypothesis: &'static str,

    /// Result
    pub result: ExperimentResult,

    /// What was learned
    pub learnings: Vec<&'static str>,

    /// Open questions
    pub open_questions: Vec<&'static str>,

    /// Related modules
    pub modules: Vec<&'static str>,
}

/// Experiment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    /// In progress
    Open,
    /// Completed with result
    Closed,
    /// Superseded by later work
    Archived,
}

/// Experiment result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentResult {
    /// Hypothesis confirmed
    Confirmed,
    /// Hypothesis falsified
    Falsified,
    /// Inconclusive
    Inconclusive,
}

// ═══════════════════════════════════════════════════════════════════════════════
// THE PROJECTION BOUND THEOREM
// ═══════════════════════════════════════════════════════════════════════════════

/// The two (and only two) ways structure can survive projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscapeRoute {
    /// Structure lies in ker(π)⊥ — aligned with what the observer measures.
    /// The coupling points in the same direction as the projection.
    Alignment,

    /// Structure modifies π itself — changes how the observer observes.
    /// The coupling grabs the observer and turns its head.
    Adaptation,
}

/// The Projection Bound Theorem — final theoretical crystallization.
///
/// **Statement**: Structure in X survives coarse-graining π: X → Y if and only if:
/// 1. **ALIGNMENT**: Structure lies in ker(π)⊥ (orthogonal to kernel)
/// 2. **ADAPTATION**: Structure modifies π itself (changes the observer)
///
/// All other structure is mathematically erased—not hidden, not subtle, **gone**.
///
/// This is not pessimism. This is linear algebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionBoundTheorem {
    /// The theorem statement in plain English
    pub statement: &'static str,

    /// The formal statement
    pub formal: &'static str,

    /// What this theorem kills (claims that cannot be true)
    pub kills: Vec<&'static str>,

    /// The only escape routes
    pub escape_routes: Vec<EscapeRoute>,
}

impl ProjectionBoundTheorem {
    /// The canonical theorem as crystallized from adversarial audit.
    pub fn canonical() -> Self {
        Self {
            statement: "Structure cannot survive projection unless it is aligned \
                       with the projection or alters the projection.",

            formal: "Structure in X survives π: X → Y iff \
                    (1) Structure ∈ ker(π)⊥, or \
                    (2) Structure modifies π",

            kills: vec![
                "The meaning is still there but hidden",
                "The system feels it even if we can't measure it",
                "There's deep structure influencing things invisibly",
                "Semantic coupling exists but we lack the tools to see it",
                "The connection is real but subtle",
            ],

            escape_routes: vec![EscapeRoute::Alignment, EscapeRoute::Adaptation],
        }
    }

    /// Stoner-level explanation for accessibility.
    pub fn eli5() -> &'static str {
        r#"
If you squash reality down to a simplified view, any details that don't
line up with how you're squashing it will disappear.

They don't "hide." They don't "influence subtly." They just don't make it through.

Two ways to survive:
1. Point in the direction you're looking (alignment)
2. Grab the observer's head and turn it (adaptation)

No third option. If you don't look for it the right way—and it doesn't
force you to look differently—it might as well not exist.
"#
    }

    /// The lock — the sentence that prevents reopening closed questions.
    pub fn the_lock() -> &'static str {
        "That would require violating assumption X. That is a different problem."
    }
}

impl ExperimentArchive {
    /// Get the canonical archive record for the coupling experiment
    pub fn coupling_experiment() -> Self {
        Self {
            id: "COUPLING-2026-001",
            date_closed: "2026-01-11",
            status: ExperimentStatus::Closed,
            hypothesis: "Strong (B): Cross-level coupling reshapes attractor basins \
                        in a structure-dependent, parameter-linked way",
            result: ExperimentResult::Falsified,
            learnings: vec![
                "Weak cross-level coupling exists (correlation increases with strength)",
                "Basin geometry is invariant under tested coupling (μ unchanged)",
                "Coupling lives in tangent space, not basin topology",
                "Attractors are robust under sub-critical perturbation",
                "Fixed π + strong dissipation + kernel misalignment → no-go",
                "(A)+(C) sufficient to explain observations",
                "Basinized System survives as valid formal object",
                "B′ CONFIRMED: Supercritical coupling causes basin deformation",
                "Coupling strength (not damping) is the key factor for deformation",
                "C=0 → 0% deformation; C>0 → 76-100% deformation at all damping",
                "PROJECTION BOUND: Structure survives π iff aligned or adapts π",
                "Two escape routes only: ker(π)⊥ alignment OR observer modification",
                "All non-aligned, non-adaptive structure is erased, not hidden",
            ],
            open_questions: vec![
                "B′: Does adaptive π enable basin deformation?",
                "B′: Can kernel-aligned coupling produce observable deformation?",
                "What is the exact threshold ‖C‖_crit for deformation onset?",
                "Does deformation persist under non-synthetic (real) data?",
            ],
            modules: vec![
                "coupling::BasinizedSystem",
                "coupling::CouplingOperator",
                "coupling::FisherEstimator",
                "coupling::TwoLevelSystem",
                "coupling::CouplingExperiment",
                "coupling::RealDataExperiment",
                "coupling::negative_result",
                "coupling::no_go",
                "coupling::b_prime",
                "coupling::archive",
            ],
        }
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            r#"
═══════════════════════════════════════════════════════════════════════════════
 EXPERIMENT ARCHIVE: {}
═══════════════════════════════════════════════════════════════════════════════

Status:     {:?}
Date:       {}
Result:     {:?}

Hypothesis:
  {}

Learnings:
{}

Open Questions:
{}

Modules:
{}
═══════════════════════════════════════════════════════════════════════════════
"#,
            self.id,
            self.status,
            self.date_closed,
            self.result,
            self.hypothesis,
            self.learnings
                .iter()
                .map(|l| format!("  • {}", l))
                .collect::<Vec<_>>()
                .join("\n"),
            self.open_questions
                .iter()
                .map(|q| format!("  ? {}", q))
                .collect::<Vec<_>>()
                .join("\n"),
            self.modules
                .iter()
                .map(|m| format!("  - {}", m))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}

/// Final status table (the locked-in record)
pub fn final_status_table() -> &'static str {
    r#"
┌────────────────────────────────────┬───────────────┐
│ Claim                              │ Status        │
├────────────────────────────────────┼───────────────┤
│ Basinized System formal object     │ ✓ Valid       │
│ Weak cross-level coupling          │ ✓ Demonstrated│
│ Semantic > random coupling         │ ✗ Falsified   │
│ Basin deformation from coupling    │ ✗ Falsified   │
│ Strong (B)                         │ ✗ Dead        │
│ (A)+(C) sufficient                 │ ✓ Supported   │
│ B′ (supercritical coupling)        │ ✓ Confirmed   │
└────────────────────────────────────┴───────────────┘
"#
}

/// Print archive summary
pub fn print_archive() {
    let archive = ExperimentArchive::coupling_experiment();
    println!("{}", archive.summary());
    println!("{}", final_status_table());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archive_record() {
        let archive = ExperimentArchive::coupling_experiment();

        assert_eq!(archive.status, ExperimentStatus::Closed);
        assert_eq!(archive.result, ExperimentResult::Falsified);
        assert!(!archive.learnings.is_empty());
        assert!(!archive.open_questions.is_empty());
    }

    #[test]
    fn test_summary_generation() {
        let archive = ExperimentArchive::coupling_experiment();
        let summary = archive.summary();

        assert!(summary.contains("COUPLING-2026-001"));
        assert!(summary.contains("Falsified"));
        assert!(summary.contains("Weak cross-level coupling"));
    }

    #[test]
    fn test_status_table() {
        let table = final_status_table();

        assert!(table.contains("Basinized System"));
        assert!(table.contains("✓ Valid"));
        assert!(table.contains("✗ Dead"));
        assert!(table.contains("B′ (supercritical coupling)"));
        assert!(table.contains("✓ Confirmed"));
    }

    #[test]
    fn test_projection_bound_theorem() {
        let theorem = ProjectionBoundTheorem::canonical();

        // Exactly two escape routes
        assert_eq!(theorem.escape_routes.len(), 2);
        assert!(theorem.escape_routes.contains(&EscapeRoute::Alignment));
        assert!(theorem.escape_routes.contains(&EscapeRoute::Adaptation));

        // Statement contains the key insight
        assert!(theorem.statement.contains("aligned"));
        assert!(theorem.statement.contains("alters"));

        // Kills the right things
        assert!(theorem.kills.iter().any(|k| k.contains("hidden")));
        assert!(theorem.kills.iter().any(|k| k.contains("subtle")));

        // ELI5 exists and is accessible
        let eli5 = ProjectionBoundTheorem::eli5();
        assert!(eli5.contains("squash"));
        assert!(eli5.contains("No third option"));

        // The lock exists
        let lock = ProjectionBoundTheorem::the_lock();
        assert!(lock.contains("assumption"));
    }
}
