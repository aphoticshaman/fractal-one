//! Constraint Propagator — Kill branches, reduce degrees of freedom
//!
//! Given a new fact (or test result), propagate eliminations.
//! If no eliminations occur, the "fact" was irrelevant; penalize that line.

use super::crux::Crux;
use super::graph::{ClaimId, ConstraintGraph};

/// Propagates constraint eliminations through the graph
pub struct ConstraintPropagator {
    /// Track irrelevant tests (no eliminations)
    irrelevant_count: usize,
    /// Track effective tests
    effective_count: usize,
}

impl ConstraintPropagator {
    pub fn new() -> Self {
        Self {
            irrelevant_count: 0,
            effective_count: 0,
        }
    }

    /// Propagate a test result through the graph
    ///
    /// Returns number of claims killed
    pub fn propagate(
        &mut self,
        graph: &mut ConstraintGraph,
        crux: &Crux,
        test_passed: bool,
    ) -> usize {
        let mut kills = 0;

        // Update the tested claim
        graph.mark_tested(crux.target, test_passed);

        if test_passed {
            // Claim confirmed: kill conflicting claims
            kills += graph.propagate_from(crux.target);
        } else {
            // Claim falsified: it's already marked dead by mark_tested
            kills += 1; // The claim itself

            // Boost alternatives
            let conflicts = graph.get_conflicts(crux.target);
            for conflict in conflicts {
                if let Some(claim) = graph.get_mut(conflict) {
                    // Increase probability of alternatives
                    claim.probability = (claim.probability + 0.1).min(1.0);
                }
            }
        }

        // Track effectiveness
        if kills > 0 {
            self.effective_count += 1;
        } else {
            self.irrelevant_count += 1;
        }

        kills
    }

    /// Propagate a raw fact (not from a crux test)
    pub fn propagate_fact(
        &mut self,
        graph: &mut ConstraintGraph,
        claim_id: ClaimId,
        is_true: bool,
    ) -> usize {
        graph.mark_tested(claim_id, is_true);

        if is_true {
            graph.propagate_from(claim_id)
        } else {
            1 // Just the claim itself
        }
    }

    /// Get effectiveness ratio
    pub fn effectiveness(&self) -> f64 {
        let total = self.effective_count + self.irrelevant_count;
        if total == 0 {
            return 1.0;
        }
        self.effective_count as f64 / total as f64
    }

    /// Get counts
    pub fn stats(&self) -> (usize, usize) {
        (self.effective_count, self.irrelevant_count)
    }

    /// Check for contradictions in the graph
    ///
    /// Returns list of contradiction descriptions
    pub fn find_contradictions(&self, graph: &ConstraintGraph) -> Vec<String> {
        let mut contradictions = Vec::new();

        let live: Vec<_> = graph.live().iter().map(|c| c.id).collect();

        // Check each live claim against others
        for &id in &live {
            let conflicts = graph.get_conflicts(id);
            for conflict in conflicts {
                if live.contains(&conflict) {
                    if let (Some(a), Some(b)) = (graph.get(id), graph.get(conflict)) {
                        contradictions.push(format!(
                            "CONTRADICTION: '{}' AND '{}' both alive",
                            a.content, b.content
                        ));
                    }
                }
            }
        }

        // Deduplicate (A-B and B-A are same contradiction)
        contradictions.sort();
        contradictions.dedup();

        contradictions
    }
}

impl Default for ConstraintPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propagation_on_success() {
        let mut graph = ConstraintGraph::new();
        let c1 = graph.add_claim("A is true", 0.6, 1.0);
        let c2 = graph.add_claim("A is false", 0.4, 1.0);
        graph.add_exclusion("A_truth", c1, c2);

        let crux = Crux {
            target: c1,
            falsifier: "test A".into(),
            test_type: super::super::crux::TestType::Manual,
            if_true: "accept A".into(),
            if_false: "reject A".into(),
            expected: Some(true),
        };

        let mut prop = ConstraintPropagator::new();
        let kills = prop.propagate(&mut graph, &crux, true);

        assert_eq!(kills, 1); // c2 should be killed
        assert!(!graph.get(c2).unwrap().alive);
        assert!(graph.get(c1).unwrap().alive);
    }

    #[test]
    fn test_propagation_on_failure() {
        let mut graph = ConstraintGraph::new();
        let c1 = graph.add_claim("hypothesis X", 0.7, 1.0);

        let crux = Crux {
            target: c1,
            falsifier: "test X".into(),
            test_type: super::super::crux::TestType::Manual,
            if_true: "X confirmed".into(),
            if_false: "X refuted".into(),
            expected: Some(true),
        };

        let mut prop = ConstraintPropagator::new();
        let kills = prop.propagate(&mut graph, &crux, false);

        assert_eq!(kills, 1); // c1 should be killed
        assert!(!graph.get(c1).unwrap().alive);
    }
}
