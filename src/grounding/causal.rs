//! ═══════════════════════════════════════════════════════════════════════════════
//! CAUSAL MODELING — Interventional, Not Just Correlational
//! ═══════════════════════════════════════════════════════════════════════════════
//! LLMs learn correlations. They don't understand causation.
//! "Correlation is not causation" - but LLMs can't tell the difference.
//!
//! This module provides:
//! - Causal graph construction and maintenance
//! - Intervention modeling (do-calculus)
//! - Counterfactual reasoning
//! - Confounding analysis
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::ObservationBatch;
use crate::time::TimePoint;
use std::collections::{HashMap, HashSet, VecDeque};

/// A node in the causal graph
#[derive(Debug, Clone)]
pub struct CausalNode {
    pub id: String,
    pub node_type: NodeType,
    pub value: Option<f64>,
    pub confidence: f64,
    pub last_updated: TimePoint,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    /// Observable variable
    Observable,
    /// Latent/hidden variable
    Latent,
    /// Intervention target
    Intervention,
    /// Outcome of interest
    Outcome,
}

/// An edge in the causal graph
#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub from: String,
    pub to: String,
    pub strength: f64,
    pub edge_type: EdgeType,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeType {
    /// Direct causal effect
    Direct,
    /// Mediated effect (through other variables)
    Mediated,
    /// Confounded (common cause)
    Confounded,
    /// Collider (common effect)
    Collider,
}

/// The causal graph
#[derive(Debug, Clone)]
pub struct CausalGraph {
    nodes: HashMap<String, CausalNode>,
    edges: Vec<CausalEdge>,
    adjacency: HashMap<String, Vec<String>>,
    reverse_adjacency: HashMap<String, Vec<String>>,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: CausalNode) {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        self.adjacency.entry(id.clone()).or_default();
        self.reverse_adjacency.entry(id).or_default();
    }

    pub fn add_edge(&mut self, edge: CausalEdge) {
        self.adjacency
            .entry(edge.from.clone())
            .or_default()
            .push(edge.to.clone());
        self.reverse_adjacency
            .entry(edge.to.clone())
            .or_default()
            .push(edge.from.clone());
        self.edges.push(edge);
    }

    pub fn get_node(&self, id: &str) -> Option<&CausalNode> {
        self.nodes.get(id)
    }

    pub fn get_children(&self, id: &str) -> Vec<&str> {
        self.adjacency
            .get(id)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    pub fn get_parents(&self, id: &str) -> Vec<&str> {
        self.reverse_adjacency
            .get(id)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Find all ancestors of a node (transitive closure of parents)
    pub fn ancestors(&self, id: &str) -> HashSet<String> {
        let mut ancestors = HashSet::new();
        let mut queue: VecDeque<&str> = self.get_parents(id).into_iter().collect();

        while let Some(node) = queue.pop_front() {
            if ancestors.insert(node.to_string()) {
                for parent in self.get_parents(node) {
                    queue.push_back(parent);
                }
            }
        }

        ancestors
    }

    /// Find all descendants of a node
    pub fn descendants(&self, id: &str) -> HashSet<String> {
        let mut descendants = HashSet::new();
        let mut queue: VecDeque<&str> = self.get_children(id).into_iter().collect();

        while let Some(node) = queue.pop_front() {
            if descendants.insert(node.to_string()) {
                for child in self.get_children(node) {
                    queue.push_back(child);
                }
            }
        }

        descendants
    }

    /// Check if there's a directed path from A to B
    pub fn has_path(&self, from: &str, to: &str) -> bool {
        self.descendants(from).contains(to)
    }

    /// Find confounders between two variables
    pub fn find_confounders(&self, a: &str, b: &str) -> Vec<String> {
        let ancestors_a = self.ancestors(a);
        let ancestors_b = self.ancestors(b);

        ancestors_a.intersection(&ancestors_b).cloned().collect()
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// An intervention (do-operator)
#[derive(Debug, Clone)]
pub struct Intervention {
    /// Variable being intervened on
    pub variable: String,
    /// Value being set
    pub value: f64,
    /// Whether to cut incoming edges (true intervention)
    pub cut_edges: bool,
}

/// Result of an intervention
#[derive(Debug, Clone)]
pub struct InterventionResult {
    pub intervention: Intervention,
    pub affected_variables: HashMap<String, f64>,
    pub confidence: f64,
    pub timestamp: TimePoint,
}

/// A counterfactual query
#[derive(Debug, Clone)]
pub struct CounterfactualQuery {
    /// The factual observation
    pub factual: HashMap<String, f64>,
    /// The hypothetical intervention
    pub intervention: Intervention,
    /// Variable we want to know about
    pub query_variable: String,
}

/// Result of causal inference
#[derive(Debug, Clone)]
pub struct CausalInference {
    pub query: String,
    pub result: f64,
    pub confidence: f64,
    pub confounders_identified: Vec<String>,
    pub assumptions: Vec<String>,
}

/// Analysis of confounding
#[derive(Debug, Clone)]
pub struct ConfoundingAnalysis {
    pub variable_a: String,
    pub variable_b: String,
    pub confounders: Vec<String>,
    pub bias_direction: BiasDirection,
    pub adjustment_needed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BiasDirection {
    AwayFromNull,
    TowardNull,
    Unknown,
}

/// Configuration for causal modeling
#[derive(Debug, Clone)]
pub struct CausalConfig {
    /// Minimum correlation to consider as potential edge
    pub edge_threshold: f64,
    /// Minimum samples before adding edge
    pub min_samples: usize,
    /// Decay factor for edge strength over time
    pub decay_factor: f64,
    /// Maximum graph size (nodes)
    pub max_nodes: usize,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            edge_threshold: 0.3,
            min_samples: 10,
            decay_factor: 0.99,
            max_nodes: 1000,
        }
    }
}

/// The Causal Model - understands why, not just what
pub struct CausalModel {
    config: CausalConfig,
    graph: CausalGraph,
    observation_counts: HashMap<(String, String), usize>,
    correlation_estimates: HashMap<(String, String), f64>,
    last_update: TimePoint,
}

impl CausalModel {
    pub fn new(config: CausalConfig) -> Self {
        Self {
            config,
            graph: CausalGraph::new(),
            observation_counts: HashMap::new(),
            correlation_estimates: HashMap::new(),
            last_update: TimePoint::now(),
        }
    }

    /// Update the causal model from observations
    pub fn update(&mut self, observations: &ObservationBatch) -> f64 {
        let now = TimePoint::now();

        // Extract numeric values from observations
        let values: Vec<(String, f64)> = observations
            .iter()
            .map(|obs| {
                let key = format!("{:?}", obs.key);
                let value = obs.value.value;
                (key, value)
            })
            .collect();

        // Update nodes
        for (key, value) in &values {
            self.update_node(key, *value, now);
        }

        // Update correlations (potential edges)
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                self.update_correlation(&values[i].0, values[i].1, &values[j].0, values[j].1);
            }
        }

        // Discover edges from correlations
        self.discover_edges();

        // Decay old edges
        self.decay_edges();

        self.last_update = now;

        // Return overall model confidence
        self.calculate_confidence()
    }

    fn update_node(&mut self, key: &str, value: f64, now: TimePoint) {
        if let Some(node) = self.graph.nodes.get_mut(key) {
            node.value = Some(value);
            node.last_updated = now;
            // Increase confidence with more observations
            node.confidence = (node.confidence + 0.01).min(1.0);
        } else if self.graph.nodes.len() < self.config.max_nodes {
            let node = CausalNode {
                id: key.to_string(),
                node_type: NodeType::Observable,
                value: Some(value),
                confidence: 0.5,
                last_updated: now,
            };
            self.graph.add_node(node);
        }
    }

    fn update_correlation(&mut self, key_a: &str, val_a: f64, key_b: &str, val_b: f64) {
        let pair = if key_a < key_b {
            (key_a.to_string(), key_b.to_string())
        } else {
            (key_b.to_string(), key_a.to_string())
        };

        // Increment count
        *self.observation_counts.entry(pair.clone()).or_insert(0) += 1;

        // Simple running correlation estimate (would be more sophisticated in production)
        let current = self
            .correlation_estimates
            .get(&pair)
            .copied()
            .unwrap_or(0.0);
        let new_signal = (val_a * val_b).signum() * 0.1; // Simplified
        let updated = current * 0.9 + new_signal * 0.1;
        self.correlation_estimates.insert(pair, updated);
    }

    fn discover_edges(&mut self) {
        let discoveries: Vec<(String, String, f64)> = self
            .correlation_estimates
            .iter()
            .filter_map(|((a, b), corr)| {
                let count = self
                    .observation_counts
                    .get(&(a.clone(), b.clone()))
                    .copied()
                    .unwrap_or(0);
                if count >= self.config.min_samples && corr.abs() >= self.config.edge_threshold {
                    Some((a.clone(), b.clone(), *corr))
                } else {
                    None
                }
            })
            .collect();

        for (a, b, strength) in discoveries {
            // Check if edge already exists
            let exists = self
                .graph
                .edges
                .iter()
                .any(|e| (e.from == a && e.to == b) || (e.from == b && e.to == a));

            if !exists {
                // Direction heuristic: temporal ordering if available, otherwise arbitrary
                let (from, to) = if strength > 0.0 { (a, b) } else { (b, a) };

                let edge = CausalEdge {
                    from,
                    to,
                    strength: strength.abs(),
                    edge_type: EdgeType::Direct, // Assume direct, refine later
                    confidence: 0.5,
                };
                self.graph.add_edge(edge);
            }
        }
    }

    fn decay_edges(&mut self) {
        for edge in &mut self.graph.edges {
            edge.confidence *= self.config.decay_factor;
        }

        // Remove very low confidence edges
        self.graph.edges.retain(|e| e.confidence > 0.1);
    }

    fn calculate_confidence(&self) -> f64 {
        if self.graph.nodes.is_empty() {
            return 0.0;
        }

        let node_confidence: f64 = self.graph.nodes.values().map(|n| n.confidence).sum::<f64>()
            / self.graph.nodes.len() as f64;

        let edge_confidence = if self.graph.edges.is_empty() {
            0.5
        } else {
            self.graph.edges.iter().map(|e| e.confidence).sum::<f64>()
                / self.graph.edges.len() as f64
        };

        (node_confidence + edge_confidence) / 2.0
    }

    /// Perform an intervention
    pub fn intervene(&mut self, intervention: Intervention) -> InterventionResult {
        let mut affected = HashMap::new();

        // Set the intervened variable
        if let Some(node) = self.graph.nodes.get_mut(&intervention.variable) {
            node.value = Some(intervention.value);
            node.node_type = NodeType::Intervention;
            affected.insert(intervention.variable.clone(), intervention.value);
        }

        // Propagate effects through descendants
        let descendants = self.graph.descendants(&intervention.variable);
        for desc in descendants {
            if let Some(node) = self.graph.nodes.get(&desc) {
                // Simple linear propagation (would be more sophisticated)
                let parent_effects: f64 = self
                    .graph
                    .get_parents(&desc)
                    .iter()
                    .filter_map(|p| affected.get(*p))
                    .sum();

                let edge_strength: f64 = self
                    .graph
                    .edges
                    .iter()
                    .filter(|e| e.to == desc && affected.contains_key(&e.from))
                    .map(|e| e.strength)
                    .sum::<f64>()
                    .max(0.1);

                let new_value = node.value.unwrap_or(0.0) + parent_effects * edge_strength * 0.1;
                affected.insert(desc, new_value);
            }
        }

        InterventionResult {
            intervention,
            affected_variables: affected,
            confidence: self.calculate_confidence(),
            timestamp: TimePoint::now(),
        }
    }

    /// Answer a counterfactual query
    pub fn counterfactual(&self, query: CounterfactualQuery) -> CausalInference {
        // Step 1: Identify confounders
        let confounders = self
            .graph
            .find_confounders(&query.intervention.variable, &query.query_variable);

        // Step 2: Check if query is identifiable
        let has_direct_path = self
            .graph
            .has_path(&query.intervention.variable, &query.query_variable);

        // Step 3: Estimate counterfactual
        let factual_value = query
            .factual
            .get(&query.query_variable)
            .copied()
            .unwrap_or(0.0);

        let effect_estimate = if has_direct_path {
            // Find total effect through all paths
            let paths = self.find_causal_paths(&query.intervention.variable, &query.query_variable);

            let total_effect: f64 = paths
                .iter()
                .map(|path| {
                    path.windows(2)
                        .filter_map(|w| {
                            self.graph
                                .edges
                                .iter()
                                .find(|e| e.from == w[0] && e.to == w[1])
                                .map(|e| e.strength)
                        })
                        .product::<f64>()
                })
                .sum();

            let intervention_delta = query.intervention.value
                - query
                    .factual
                    .get(&query.intervention.variable)
                    .copied()
                    .unwrap_or(0.0);

            factual_value + intervention_delta * total_effect
        } else {
            factual_value // No causal path, no effect
        };

        let confidence = if has_direct_path && confounders.is_empty() {
            0.8
        } else if has_direct_path {
            0.5 // Confounded
        } else {
            0.2 // No path
        };

        CausalInference {
            query: query.query_variable,
            result: effect_estimate,
            confidence,
            confounders_identified: confounders,
            assumptions: vec![
                "Causal sufficiency".to_string(),
                "No unmeasured confounders".to_string(),
                "Faithfulness".to_string(),
            ],
        }
    }

    fn find_causal_paths(&self, from: &str, to: &str) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        let mut current_path = vec![from.to_string()];
        self.dfs_paths(from, to, &mut current_path, &mut paths);
        paths
    }

    fn dfs_paths(
        &self,
        current: &str,
        target: &str,
        path: &mut Vec<String>,
        paths: &mut Vec<Vec<String>>,
    ) {
        if current == target {
            paths.push(path.clone());
            return;
        }

        for child in self.graph.get_children(current) {
            if !path.contains(&child.to_string()) {
                path.push(child.to_string());
                self.dfs_paths(child, target, path, paths);
                path.pop();
            }
        }
    }

    /// Analyze confounding between two variables
    pub fn analyze_confounding(&self, var_a: &str, var_b: &str) -> ConfoundingAnalysis {
        let confounders = self.graph.find_confounders(var_a, var_b);

        let bias_direction = if confounders.is_empty() {
            BiasDirection::Unknown
        } else {
            // Heuristic based on confounders
            BiasDirection::AwayFromNull
        };

        ConfoundingAnalysis {
            variable_a: var_a.to_string(),
            variable_b: var_b.to_string(),
            confounders: confounders.clone(),
            bias_direction,
            adjustment_needed: !confounders.is_empty(),
        }
    }

    /// Get the causal graph
    pub fn graph(&self) -> &CausalGraph {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_graph_creation() {
        let mut graph = CausalGraph::new();

        graph.add_node(CausalNode {
            id: "A".to_string(),
            node_type: NodeType::Observable,
            value: Some(1.0),
            confidence: 0.9,
            last_updated: TimePoint::now(),
        });

        graph.add_node(CausalNode {
            id: "B".to_string(),
            node_type: NodeType::Observable,
            value: Some(2.0),
            confidence: 0.9,
            last_updated: TimePoint::now(),
        });

        graph.add_edge(CausalEdge {
            from: "A".to_string(),
            to: "B".to_string(),
            strength: 0.8,
            edge_type: EdgeType::Direct,
            confidence: 0.9,
        });

        assert!(graph.has_path("A", "B"));
        assert!(!graph.has_path("B", "A"));
    }

    #[test]
    fn test_causal_model_creation() {
        let model = CausalModel::new(CausalConfig::default());
        assert_eq!(model.graph.nodes.len(), 0);
    }
}
