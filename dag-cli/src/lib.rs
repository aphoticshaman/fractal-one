//! # dag-cli
//!
//! Lightweight library for DAG validation, cycle detection, and topological sorting.
//!
//! ## Features
//! - Cycle detection with cycle path reporting
//! - Topological sorting
//! - DOT format export for Graphviz visualization
//! - JSON import/export
//!
//! ## Example
//! ```rust
//! use dag_cli::{Dag, DagSpec};
//!
//! let spec = DagSpec {
//!     nodes: vec!["A".into(), "B".into(), "C".into()],
//!     edges: vec![("A".into(), "B".into()), ("B".into(), "C".into())],
//!     metadata: None,
//!     node_meta: None,
//!     edge_meta: None,
//! };
//!
//! let dag = Dag::from_spec(&spec).unwrap();
//! assert!(dag.is_acyclic());
//! assert_eq!(dag.topological_sort().unwrap(), vec!["A", "B", "C"]);
//! ```

// Clippy configuration - allow style lints
#![allow(clippy::for_kv_map)]
#![allow(clippy::map_clone)]
#![allow(clippy::manual_clamp)]

use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Error types for DAG operations
#[derive(Debug, Clone)]
pub enum DagError {
    /// Graph contains a cycle
    CycleDetected { cycle: Vec<String> },
    /// Node referenced in edge doesn't exist
    NodeNotFound { node: String },
    /// Duplicate node name
    DuplicateNode { node: String },
    /// JSON parsing error
    ParseError { message: String },
}

impl std::fmt::Display for DagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DagError::CycleDetected { cycle } => {
                write!(f, "Cycle detected: {}", cycle.join(" -> "))
            }
            DagError::NodeNotFound { node } => {
                write!(f, "Node not found: {}", node)
            }
            DagError::DuplicateNode { node } => {
                write!(f, "Duplicate node: {}", node)
            }
            DagError::ParseError { message } => {
                write!(f, "Parse error: {}", message)
            }
        }
    }
}

impl std::error::Error for DagError {}

/// Node metadata (optional)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeMeta {
    /// Optional label for display
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Optional probability (for TICE-style reasoning)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f64>,
    /// Optional weight/cost
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// Optional color for visualization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Arbitrary key-value metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<HashMap<String, String>>,
}

/// Edge metadata (optional)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EdgeMeta {
    /// Optional label for the edge
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Optional weight/cost
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
}

/// Specification for creating a DAG from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagSpec {
    /// List of node names
    pub nodes: Vec<String>,
    /// List of edges as (from, to) pairs
    pub edges: Vec<(String, String)>,
    /// Optional graph-level metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    /// Optional per-node metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_meta: Option<HashMap<String, NodeMeta>>,
    /// Optional per-edge metadata (key is "from->to")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_meta: Option<HashMap<String, EdgeMeta>>,
}

impl DagSpec {
    /// Create a new empty spec
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: None,
            node_meta: None,
            edge_meta: None,
        }
    }

    /// Maximum allowed JSON size (10 MB)
    const MAX_JSON_SIZE: usize = 10 * 1024 * 1024;
    /// Maximum allowed nodes in a graph
    const MAX_NODES: usize = 10_000;
    /// Maximum allowed edges in a graph
    const MAX_EDGES: usize = 100_000;

    /// Parse from JSON string with size validation
    pub fn from_json(json: &str) -> Result<Self, DagError> {
        // Validate JSON size to prevent DoS
        if json.len() > Self::MAX_JSON_SIZE {
            return Err(DagError::ParseError {
                message: format!(
                    "JSON too large: {} bytes (max {} bytes)",
                    json.len(),
                    Self::MAX_JSON_SIZE
                ),
            });
        }

        let spec: Self = serde_json::from_str(json).map_err(|e| DagError::ParseError {
            message: e.to_string(),
        })?;

        // Validate graph size to prevent resource exhaustion
        if spec.nodes.len() > Self::MAX_NODES {
            return Err(DagError::ParseError {
                message: format!(
                    "Too many nodes: {} (max {})",
                    spec.nodes.len(),
                    Self::MAX_NODES
                ),
            });
        }

        if spec.edges.len() > Self::MAX_EDGES {
            return Err(DagError::ParseError {
                message: format!(
                    "Too many edges: {} (max {})",
                    spec.edges.len(),
                    Self::MAX_EDGES
                ),
            });
        }

        Ok(spec)
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, DagError> {
        serde_json::to_string_pretty(self).map_err(|e| DagError::ParseError {
            message: e.to_string(),
        })
    }
}

impl Default for DagSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// A validated directed acyclic graph
#[derive(Debug, Clone)]
pub struct Dag {
    graph: DiGraph<String, ()>,
    name_to_idx: HashMap<String, NodeIndex>,
    idx_to_name: HashMap<NodeIndex, String>,
    node_meta: HashMap<String, NodeMeta>,
    edge_meta: HashMap<String, EdgeMeta>,
    graph_meta: HashMap<String, String>,
}

impl Dag {
    /// Create a new empty DAG
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            name_to_idx: HashMap::new(),
            idx_to_name: HashMap::new(),
            node_meta: HashMap::new(),
            edge_meta: HashMap::new(),
            graph_meta: HashMap::new(),
        }
    }

    /// Create a DAG from a specification (validates acyclicity)
    pub fn from_spec(spec: &DagSpec) -> Result<Self, DagError> {
        let mut dag = Self::new();

        // Add metadata
        if let Some(meta) = &spec.metadata {
            dag.graph_meta = meta.clone();
        }
        if let Some(node_meta) = &spec.node_meta {
            dag.node_meta = node_meta.clone();
        }
        if let Some(edge_meta) = &spec.edge_meta {
            dag.edge_meta = edge_meta.clone();
        }

        // Add nodes
        for name in &spec.nodes {
            dag.add_node(name.clone())?;
        }

        // Add edges
        for (from, to) in &spec.edges {
            dag.add_edge(from, to)?;
        }

        // Validate acyclicity
        if !dag.is_acyclic() {
            let cycle = dag.find_cycle().unwrap_or_default();
            return Err(DagError::CycleDetected { cycle });
        }

        Ok(dag)
    }

    /// Create a DAG from JSON string
    pub fn from_json(json: &str) -> Result<Self, DagError> {
        let spec = DagSpec::from_json(json)?;
        Self::from_spec(&spec)
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, name: String) -> Result<NodeIndex, DagError> {
        if self.name_to_idx.contains_key(&name) {
            return Err(DagError::DuplicateNode { node: name });
        }
        let idx = self.graph.add_node(name.clone());
        self.name_to_idx.insert(name.clone(), idx);
        self.idx_to_name.insert(idx, name);
        Ok(idx)
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: &str, to: &str) -> Result<(), DagError> {
        let from_idx = self
            .name_to_idx
            .get(from)
            .ok_or_else(|| DagError::NodeNotFound {
                node: from.to_string(),
            })?;
        let to_idx = self
            .name_to_idx
            .get(to)
            .ok_or_else(|| DagError::NodeNotFound {
                node: to.to_string(),
            })?;
        self.graph.add_edge(*from_idx, *to_idx, ());
        Ok(())
    }

    /// Check if the graph is acyclic
    pub fn is_acyclic(&self) -> bool {
        !is_cyclic_directed(&self.graph)
    }

    /// Find a cycle in the graph (if one exists)
    pub fn find_cycle(&self) -> Option<Vec<String>> {
        // Use DFS to find cycle
        let mut visited = HashMap::new();
        let mut stack = Vec::new();
        let mut on_stack = HashMap::new();

        for node in self.graph.node_indices() {
            if visited.contains_key(&node) {
                continue;
            }

            stack.push((node, false));

            while let Some((current, processed)) = stack.pop() {
                if processed {
                    on_stack.remove(&current);
                    continue;
                }

                if on_stack.contains_key(&current) {
                    // Found cycle - reconstruct it
                    let mut cycle = vec![self.idx_to_name[&current].clone()];
                    // Find path back to current
                    for &(n, _) in stack.iter().rev() {
                        cycle.push(self.idx_to_name[&n].clone());
                        if n == current {
                            break;
                        }
                    }
                    cycle.reverse();
                    cycle.push(self.idx_to_name[&current].clone());
                    return Some(cycle);
                }

                if visited.contains_key(&current) {
                    continue;
                }

                visited.insert(current, true);
                on_stack.insert(current, true);
                stack.push((current, true)); // Mark for cleanup

                for neighbor in self.graph.neighbors_directed(current, Direction::Outgoing) {
                    if on_stack.contains_key(&neighbor) {
                        // Found cycle
                        let mut cycle = vec![self.idx_to_name[&neighbor].clone()];
                        cycle.push(self.idx_to_name[&current].clone());
                        cycle.push(self.idx_to_name[&neighbor].clone());
                        return Some(cycle);
                    }
                    if !visited.contains_key(&neighbor) {
                        stack.push((neighbor, false));
                    }
                }
            }
        }

        None
    }

    /// Get topological sort of nodes
    pub fn topological_sort(&self) -> Result<Vec<String>, DagError> {
        match toposort(&self.graph, None) {
            Ok(order) => Ok(order
                .iter()
                .map(|idx| self.idx_to_name[idx].clone())
                .collect()),
            Err(cycle) => {
                let node = self.idx_to_name[&cycle.node_id()].clone();
                Err(DagError::CycleDetected { cycle: vec![node] })
            }
        }
    }

    /// Get all root nodes (nodes with no incoming edges)
    pub fn roots(&self) -> Vec<String> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .count()
                    == 0
            })
            .map(|idx| self.idx_to_name[&idx].clone())
            .collect()
    }

    /// Get all leaf nodes (nodes with no outgoing edges)
    pub fn leaves(&self) -> Vec<String> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Outgoing)
                    .count()
                    == 0
            })
            .map(|idx| self.idx_to_name[&idx].clone())
            .collect()
    }

    /// Get dependencies of a node (incoming edges)
    pub fn dependencies(&self, node: &str) -> Option<Vec<String>> {
        let idx = self.name_to_idx.get(node)?;
        Some(
            self.graph
                .neighbors_directed(*idx, Direction::Incoming)
                .map(|n| self.idx_to_name[&n].clone())
                .collect(),
        )
    }

    /// Get dependents of a node (outgoing edges)
    pub fn dependents(&self, node: &str) -> Option<Vec<String>> {
        let idx = self.name_to_idx.get(node)?;
        Some(
            self.graph
                .neighbors_directed(*idx, Direction::Outgoing)
                .map(|n| self.idx_to_name[&n].clone())
                .collect(),
        )
    }

    /// Number of nodes
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Export to DOT format for Graphviz
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph G {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box, style=rounded];\n\n");

        // Add nodes with metadata
        for (name, _idx) in &self.name_to_idx {
            let mut attrs = Vec::new();

            if let Some(meta) = self.node_meta.get(name) {
                if let Some(label) = &meta.label {
                    attrs.push(format!("label=\"{}\"", label));
                }
                if let Some(color) = &meta.color {
                    attrs.push(format!("fillcolor=\"{}\"", color));
                    attrs.push("style=\"filled,rounded\"".to_string());
                }
                if let Some(prob) = meta.probability {
                    let existing_label = attrs
                        .iter()
                        .find(|a| a.starts_with("label="))
                        .map(|a| a.clone());
                    if let Some(l) = existing_label {
                        attrs.retain(|a| !a.starts_with("label="));
                        let label_content = l.trim_start_matches("label=\"").trim_end_matches("\"");
                        attrs.push(format!(
                            "label=\"{}\\np={:.0}%\"",
                            label_content,
                            prob * 100.0
                        ));
                    } else {
                        attrs.push(format!("label=\"{}\\np={:.0}%\"", name, prob * 100.0));
                    }
                }
            }

            if attrs.is_empty() {
                dot.push_str(&format!("  \"{}\";\n", name));
            } else {
                dot.push_str(&format!("  \"{}\" [{}];\n", name, attrs.join(", ")));
            }
        }

        dot.push('\n');

        // Add edges
        for edge in self.graph.edge_references() {
            let from = &self.idx_to_name[&edge.source()];
            let to = &self.idx_to_name[&edge.target()];
            let key = format!("{}->{}", from, to);

            if let Some(meta) = self.edge_meta.get(&key) {
                let mut attrs = Vec::new();
                if let Some(label) = &meta.label {
                    attrs.push(format!("label=\"{}\"", label));
                }
                if let Some(weight) = meta.weight {
                    attrs.push(format!("penwidth={:.1}", weight.max(0.5).min(5.0)));
                }
                if attrs.is_empty() {
                    dot.push_str(&format!("  \"{}\" -> \"{}\";\n", from, to));
                } else {
                    dot.push_str(&format!(
                        "  \"{}\" -> \"{}\" [{}];\n",
                        from,
                        to,
                        attrs.join(", ")
                    ));
                }
            } else {
                dot.push_str(&format!("  \"{}\" -> \"{}\";\n", from, to));
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Export to DagSpec for JSON serialization
    pub fn to_spec(&self) -> DagSpec {
        let nodes: Vec<String> = self.name_to_idx.keys().cloned().collect();
        let edges: Vec<(String, String)> = self
            .graph
            .edge_references()
            .map(|e| {
                (
                    self.idx_to_name[&e.source()].clone(),
                    self.idx_to_name[&e.target()].clone(),
                )
            })
            .collect();

        DagSpec {
            nodes,
            edges,
            metadata: if self.graph_meta.is_empty() {
                None
            } else {
                Some(self.graph_meta.clone())
            },
            node_meta: if self.node_meta.is_empty() {
                None
            } else {
                Some(self.node_meta.clone())
            },
            edge_meta: if self.edge_meta.is_empty() {
                None
            } else {
                Some(self.edge_meta.clone())
            },
        }
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, DagError> {
        self.to_spec().to_json()
    }
}

impl Default for Dag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dag() {
        let spec = DagSpec {
            nodes: vec!["A".into(), "B".into(), "C".into()],
            edges: vec![("A".into(), "B".into()), ("B".into(), "C".into())],
            metadata: None,
            node_meta: None,
            edge_meta: None,
        };

        let dag = Dag::from_spec(&spec).unwrap();
        assert!(dag.is_acyclic());
        assert_eq!(dag.node_count(), 3);
        assert_eq!(dag.edge_count(), 2);
        assert_eq!(dag.topological_sort().unwrap(), vec!["A", "B", "C"]);
    }

    #[test]
    fn test_cycle_detection() {
        let spec = DagSpec {
            nodes: vec!["A".into(), "B".into(), "C".into()],
            edges: vec![
                ("A".into(), "B".into()),
                ("B".into(), "C".into()),
                ("C".into(), "A".into()), // Creates cycle
            ],
            metadata: None,
            node_meta: None,
            edge_meta: None,
        };

        let result = Dag::from_spec(&spec);
        assert!(result.is_err());
        if let Err(DagError::CycleDetected { cycle }) = result {
            assert!(!cycle.is_empty());
        } else {
            panic!("Expected CycleDetected error");
        }
    }

    #[test]
    fn test_roots_and_leaves() {
        let spec = DagSpec {
            nodes: vec!["A".into(), "B".into(), "C".into(), "D".into()],
            edges: vec![
                ("A".into(), "B".into()),
                ("A".into(), "C".into()),
                ("B".into(), "D".into()),
                ("C".into(), "D".into()),
            ],
            metadata: None,
            node_meta: None,
            edge_meta: None,
        };

        let dag = Dag::from_spec(&spec).unwrap();
        assert_eq!(dag.roots(), vec!["A"]);
        assert_eq!(dag.leaves(), vec!["D"]);
    }

    #[test]
    fn test_json_roundtrip() {
        let spec = DagSpec {
            nodes: vec!["X".into(), "Y".into()],
            edges: vec![("X".into(), "Y".into())],
            metadata: None,
            node_meta: None,
            edge_meta: None,
        };

        let json = spec.to_json().unwrap();
        let spec2 = DagSpec::from_json(&json).unwrap();
        assert_eq!(spec.nodes, spec2.nodes);
        assert_eq!(spec.edges, spec2.edges);
    }
}
