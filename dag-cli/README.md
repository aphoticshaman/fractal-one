# dag-cli

Lightweight CLI for DAG validation, cycle detection, and topological sorting.

## Installation

```bash
cargo install dag-cli
```

Or build from source:
```bash
git clone https://github.com/aphoticshaman/fractal-one
cd fractal-one/dag-cli
cargo build --release
```

## Usage

```bash
# Create example DAG
dag init --example > project.json

# Check for cycles
dag check project.json

# Topological sort
dag sort project.json

# Show graph info
dag info project.json

# Output DOT format for Graphviz
dag viz project.json > project.dot
dot -Tpng project.dot -o project.png

# Or directly to PNG (requires graphviz)
dag viz project.json --png project.png

# Add nodes/edges interactively
dag add-node project.json new_task
dag add-edge project.json testing new_task
```

## JSON Schema

```json
{
  "nodes": ["A", "B", "C"],
  "edges": [["A", "B"], ["B", "C"]],
  "metadata": {
    "name": "My DAG"
  },
  "node_meta": {
    "A": {
      "label": "Start",
      "probability": 0.9,
      "color": "#90EE90"
    }
  }
}
```

## Library Usage

```rust
use dag_cli::{Dag, DagSpec};

let spec = DagSpec {
    nodes: vec!["A".into(), "B".into(), "C".into()],
    edges: vec![("A".into(), "B".into()), ("B".into(), "C".into())],
    metadata: None,
    node_meta: None,
    edge_meta: None,
};

let dag = Dag::from_spec(&spec).unwrap();

assert!(dag.is_acyclic());
assert_eq!(dag.topological_sort().unwrap(), vec!["A", "B", "C"]);
assert_eq!(dag.roots(), vec!["A"]);
assert_eq!(dag.leaves(), vec!["C"]);

// Export to DOT
println!("{}", dag.to_dot());
```

## Why?

- **Lightweight**: Just petgraph + serde. No orchestration framework bloat.
- **Fast**: Cycle detection in O(V+E).
- **Composable**: Pipe JSON in, get sorted order or DOT out.
- **Library + CLI**: Use as a crate or standalone tool.

For when you need to check a dependency graph, not deploy an Airflow cluster.

## License

MIT
