//! Constraint Graph — The state space
//!
//! Everything is a typed hypergraph:
//! - Claims: propositions (can be partial)
//! - Constraints: relations that eliminate combinations
//! - Observables: measurable facts / tests
//!
//! A node only exists if it implies something testable or constraining.

use std::collections::HashMap;

/// Unique identifier for a claim
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClaimId(pub u64);

/// Unique identifier for a constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstraintId(pub u64);

/// A claim in the graph
#[derive(Debug, Clone)]
pub struct Claim {
    pub id: ClaimId,
    pub content: String,
    pub probability: f64,        // Current belief (0.0 - 1.0)
    pub value: f64,              // Importance if resolved
    pub alive: bool,             // Still viable?
    pub tested: bool,            // Has been tested?
    pub parent: Option<ClaimId>, // Derived from
    pub children: Vec<ClaimId>,  // Implies
}

/// A constraint between claims
#[derive(Debug, Clone)]
pub struct Constraint {
    pub id: ConstraintId,
    pub name: String,
    pub description: String,
    /// Claims that are mutually exclusive under this constraint
    pub excludes: Vec<(ClaimId, ClaimId)>,
    /// Claims that are required together
    pub requires: Vec<(ClaimId, ClaimId)>,
    /// Active?
    pub active: bool,
}

/// Something that can be observed/tested
#[derive(Debug, Clone)]
pub struct Observable {
    pub name: String,
    pub test_command: Option<String>,
    pub expected: Option<String>,
    pub observed: Option<String>,
}

/// The constraint graph
pub struct ConstraintGraph {
    claims: HashMap<ClaimId, Claim>,
    constraints: HashMap<ConstraintId, Constraint>,
    #[allow(dead_code)]
    observables: Vec<Observable>,
    next_claim_id: u64,
    next_constraint_id: u64,
    decision: Option<ClaimId>,
}

impl ConstraintGraph {
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
            constraints: HashMap::new(),
            observables: Vec::new(),
            next_claim_id: 1,
            next_constraint_id: 1,
            decision: None,
        }
    }

    /// Add a claim to the graph
    pub fn add_claim(&mut self, content: &str, probability: f64, value: f64) -> ClaimId {
        let id = ClaimId(self.next_claim_id);
        self.next_claim_id += 1;

        self.claims.insert(
            id,
            Claim {
                id,
                content: content.to_string(),
                probability: probability.clamp(0.0, 1.0),
                value,
                alive: true,
                tested: false,
                parent: None,
                children: Vec::new(),
            },
        );

        id
    }

    /// Add a derived claim
    pub fn add_derived(&mut self, content: &str, parent: ClaimId) -> ClaimId {
        let parent_claim = self.claims.get(&parent);
        let (prob, val) = parent_claim
            .map(|c| (c.probability * 0.8, c.value * 0.5))
            .unwrap_or((0.5, 1.0));

        let id = self.add_claim(content, prob, val);

        if let Some(c) = self.claims.get_mut(&id) {
            c.parent = Some(parent);
        }
        if let Some(p) = self.claims.get_mut(&parent) {
            p.children.push(id);
        }

        id
    }

    /// Add a mutual exclusion constraint
    pub fn add_exclusion(&mut self, name: &str, a: ClaimId, b: ClaimId) -> ConstraintId {
        let id = ConstraintId(self.next_constraint_id);
        self.next_constraint_id += 1;

        self.constraints.insert(
            id,
            Constraint {
                id,
                name: name.to_string(),
                description: format!("{:?} XOR {:?}", a, b),
                excludes: vec![(a, b)],
                requires: Vec::new(),
                active: true,
            },
        );

        id
    }

    /// Add a requirement constraint (a implies b)
    pub fn add_requirement(&mut self, name: &str, a: ClaimId, b: ClaimId) -> ConstraintId {
        let id = ConstraintId(self.next_constraint_id);
        self.next_constraint_id += 1;

        self.constraints.insert(
            id,
            Constraint {
                id,
                name: name.to_string(),
                description: format!("{:?} => {:?}", a, b),
                excludes: Vec::new(),
                requires: vec![(a, b)],
                active: true,
            },
        );

        id
    }

    /// Kill a claim (mark as not alive)
    pub fn kill(&mut self, id: ClaimId) -> bool {
        if let Some(claim) = self.claims.get_mut(&id) {
            if claim.alive {
                claim.alive = false;
                // Also kill children
                let children: Vec<ClaimId> = claim.children.clone();
                for child in children {
                    self.kill(child);
                }
                return true;
            }
        }
        false
    }

    /// Mark a claim as tested
    pub fn mark_tested(&mut self, id: ClaimId, result: bool) {
        if let Some(claim) = self.claims.get_mut(&id) {
            claim.tested = true;
            if !result {
                claim.alive = false;
            } else {
                claim.probability = 0.95; // High confidence after passing test
            }
        }
    }

    /// Get claim by ID
    pub fn get(&self, id: ClaimId) -> Option<&Claim> {
        self.claims.get(&id)
    }

    /// Get mutable claim
    pub fn get_mut(&mut self, id: ClaimId) -> Option<&mut Claim> {
        self.claims.get_mut(&id)
    }

    /// Count live claims
    pub fn live_claims(&self) -> usize {
        self.claims.values().filter(|c| c.alive).count()
    }

    /// Check if graph has collapsed to a decision
    pub fn is_decided(&self) -> bool {
        self.live_claims() <= 1 || self.decision.is_some()
    }

    /// Get the decision if made
    pub fn decision(&self) -> Option<ClaimId> {
        if let Some(d) = self.decision {
            return Some(d);
        }
        let live: Vec<_> = self.claims.values().filter(|c| c.alive).collect();
        if live.len() == 1 {
            return Some(live[0].id);
        }
        None
    }

    /// Force a decision
    pub fn commit(&mut self, id: ClaimId) {
        self.decision = Some(id);
        // Kill all other claims
        let other_ids: Vec<ClaimId> = self.claims.keys().filter(|&k| *k != id).copied().collect();
        for other in other_ids {
            self.kill(other);
        }
    }

    /// Select target with highest uncertainty × value
    ///
    /// Uncertainty = distance from 0.5 (closer = more uncertain)
    pub fn select_target(&self) -> Option<ClaimId> {
        self.claims
            .values()
            .filter(|c| c.alive && !c.tested)
            .max_by(|a, b| {
                let score_a = self.target_score(a);
                let score_b = self.target_score(b);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|c| c.id)
    }

    fn target_score(&self, claim: &Claim) -> f64 {
        // Uncertainty: highest at 0.5, lowest at 0 or 1
        let uncertainty = 1.0 - (2.0 * (claim.probability - 0.5)).abs();
        uncertainty * claim.value
    }

    /// Get all live claims
    pub fn live(&self) -> Vec<&Claim> {
        self.claims.values().filter(|c| c.alive).collect()
    }

    /// Get all claims (alive and dead)
    pub fn all_claims(&self) -> Vec<&Claim> {
        self.claims.values().collect()
    }

    /// Get dependencies of a claim (what it requires)
    pub fn get_dependencies(&self, id: ClaimId) -> Vec<ClaimId> {
        let mut deps = Vec::new();
        for constraint in self.constraints.values() {
            if !constraint.active {
                continue;
            }
            for (dependent, dependency) in &constraint.requires {
                if *dependent == id {
                    deps.push(*dependency);
                }
            }
        }
        deps
    }

    /// Get dependents of a claim (what requires it)
    pub fn get_dependents(&self, id: ClaimId) -> Vec<ClaimId> {
        let mut dependents = Vec::new();
        for constraint in self.constraints.values() {
            if !constraint.active {
                continue;
            }
            for (dependent, dependency) in &constraint.requires {
                if *dependency == id {
                    dependents.push(*dependent);
                }
            }
        }
        dependents
    }

    /// Get claims that conflict under a constraint
    pub fn get_conflicts(&self, id: ClaimId) -> Vec<ClaimId> {
        let mut conflicts = Vec::new();
        for constraint in self.constraints.values() {
            if !constraint.active {
                continue;
            }
            for (a, b) in &constraint.excludes {
                if *a == id {
                    conflicts.push(*b);
                } else if *b == id {
                    conflicts.push(*a);
                }
            }
        }
        conflicts
    }

    /// Propagate constraint: if a is alive, kill all exclusions
    pub fn propagate_from(&mut self, id: ClaimId) -> usize {
        let conflicts = self.get_conflicts(id);
        let mut kills = 0;
        for conflict in conflicts {
            if self.kill(conflict) {
                kills += 1;
            }
        }
        kills
    }
}

impl Default for ConstraintGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DAG-CLI INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

impl ConstraintGraph {
    /// Load from dag-cli DagSpec
    ///
    /// Mapping:
    /// - nodes → claims
    /// - edges → requirement constraints (from implies to)
    /// - node_meta.probability → claim probability
    pub fn from_dag_spec(spec: &dag_cli::DagSpec) -> Result<Self, String> {
        // First validate it's a valid DAG
        let _dag = dag_cli::Dag::from_spec(spec).map_err(|e| format!("Invalid DAG: {}", e))?;

        let mut graph = Self::new();
        let mut name_to_id: HashMap<String, ClaimId> = HashMap::new();

        // Add claims from nodes
        for name in &spec.nodes {
            let prob = spec
                .node_meta
                .as_ref()
                .and_then(|m| m.get(name))
                .and_then(|meta| meta.probability)
                .unwrap_or(0.5);

            let value = spec
                .node_meta
                .as_ref()
                .and_then(|m| m.get(name))
                .and_then(|meta| meta.weight)
                .unwrap_or(1.0);

            let id = graph.add_claim(name, prob, value);
            name_to_id.insert(name.clone(), id);
        }

        // Add requirement constraints from edges
        for (from, to) in &spec.edges {
            let from_id = name_to_id
                .get(from)
                .ok_or_else(|| format!("Unknown node: {}", from))?;
            let to_id = name_to_id
                .get(to)
                .ok_or_else(|| format!("Unknown node: {}", to))?;

            // Edge from A to B means "A requires B" (B must be true for A to be achievable)
            // Or we can interpret as "A enables B" (A must happen before B)
            // For TICE, use: from implies to (from requires to be possible)
            graph.add_requirement(&format!("{}->{}", from, to), *from_id, *to_id);
        }

        Ok(graph)
    }

    /// Export to dag-cli DagSpec
    pub fn to_dag_spec(&self) -> dag_cli::DagSpec {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_meta = HashMap::new();
        let mut id_to_name: HashMap<ClaimId, String> = HashMap::new();

        // Export claims as nodes
        for claim in self.claims.values() {
            let name = claim.content.clone();
            nodes.push(name.clone());
            id_to_name.insert(claim.id, name.clone());

            let meta = dag_cli::NodeMeta {
                probability: Some(claim.probability),
                weight: Some(claim.value),
                label: if claim.alive {
                    None
                } else {
                    Some("[KILLED]".into())
                },
                color: if claim.alive {
                    if claim.tested {
                        Some("#90EE90".into()) // Light green for tested
                    } else {
                        None
                    }
                } else {
                    Some("#FF6B6B".into()) // Red for killed
                },
                extra: None,
            };
            node_meta.insert(name, meta);
        }

        // Export requirement constraints as edges
        for constraint in self.constraints.values() {
            for (from_id, to_id) in &constraint.requires {
                if let (Some(from), Some(to)) = (id_to_name.get(from_id), id_to_name.get(to_id)) {
                    edges.push((from.clone(), to.clone()));
                }
            }
        }

        dag_cli::DagSpec {
            nodes,
            edges,
            metadata: Some(
                [("source".into(), "TICE export".into())]
                    .into_iter()
                    .collect(),
            ),
            node_meta: Some(node_meta),
            edge_meta: None,
        }
    }

    /// Load from JSON file
    pub fn load_dag_json(path: &str) -> Result<Self, String> {
        let json =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;
        let spec = dag_cli::DagSpec::from_json(&json)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;
        Self::from_dag_spec(&spec)
    }

    /// Export to JSON string
    pub fn to_dag_json(&self) -> Result<String, String> {
        self.to_dag_spec()
            .to_json()
            .map_err(|e| format!("Failed to serialize: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claim_lifecycle() {
        let mut graph = ConstraintGraph::new();

        let c1 = graph.add_claim("P = NP", 0.1, 10.0);
        let c2 = graph.add_claim("P ≠ NP", 0.9, 10.0);

        assert_eq!(graph.live_claims(), 2);

        graph.kill(c1);
        assert_eq!(graph.live_claims(), 1);
        assert!(graph.is_decided());
        assert_eq!(graph.decision(), Some(c2));
    }

    #[test]
    fn test_exclusion_constraint() {
        let mut graph = ConstraintGraph::new();

        let c1 = graph.add_claim("A", 0.5, 1.0);
        let c2 = graph.add_claim("B", 0.5, 1.0);

        graph.add_exclusion("A_xor_B", c1, c2);

        // Propagate from c1 should kill c2
        graph.mark_tested(c1, true);
        let kills = graph.propagate_from(c1);

        assert_eq!(kills, 1);
        assert!(!graph.get(c2).unwrap().alive);
    }
}
