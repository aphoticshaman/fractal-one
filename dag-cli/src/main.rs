//! dag-cli - Lightweight DAG validation and visualization
//!
//! Usage:
//!   dag check graph.json     # Validate acyclicity
//!   dag sort graph.json      # Topological sort
//!   dag viz graph.json       # Output DOT format
//!   dag info graph.json      # Show graph statistics

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dag_cli::{Dag, DagSpec};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Validate a file path to prevent path traversal attacks.
/// For existing files: canonicalizes and checks for `..` components.
/// For new files: validates parent directory exists.
fn validate_path(path: &Path, must_exist: bool) -> Result<PathBuf> {
    // Convert to string and check for obvious traversal attempts
    let path_str = path.to_string_lossy();
    if path_str.contains("..") {
        anyhow::bail!("Path traversal not allowed: {}", path_str);
    }

    if must_exist {
        let canonical = path
            .canonicalize()
            .with_context(|| format!("Failed to resolve path: {}", path.display()))?;
        Ok(canonical)
    } else {
        // For new files, validate parent exists
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                anyhow::bail!("Parent directory does not exist: {}", parent.display());
            }
        }
        Ok(path.to_path_buf())
    }
}

#[derive(Parser)]
#[command(name = "dag")]
#[command(about = "Lightweight DAG validation, cycle detection, and topological sorting")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Check if graph is a valid DAG (no cycles)
    Check {
        /// Input JSON file (or - for stdin)
        #[arg(default_value = "-")]
        input: String,
    },

    /// Output topological sort order
    Sort {
        /// Input JSON file (or - for stdin)
        #[arg(default_value = "-")]
        input: String,

        /// Output as JSON array
        #[arg(long)]
        json: bool,
    },

    /// Output DOT format for Graphviz visualization
    Viz {
        /// Input JSON file (or - for stdin)
        #[arg(default_value = "-")]
        input: String,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Pipe to dot command and output PNG (requires graphviz)
        #[arg(long)]
        png: Option<PathBuf>,
    },

    /// Show graph information and statistics
    Info {
        /// Input JSON file (or - for stdin)
        #[arg(default_value = "-")]
        input: String,
    },

    /// Create a new empty DAG spec template
    Init {
        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Include example nodes and edges
        #[arg(long)]
        example: bool,
    },

    /// Add a node to an existing DAG
    #[command(name = "add-node")]
    AddNode {
        /// Input JSON file
        input: PathBuf,

        /// Node name to add
        name: String,

        /// Optional probability (0.0-1.0)
        #[arg(short, long)]
        probability: Option<f64>,
    },

    /// Add an edge to an existing DAG
    #[command(name = "add-edge")]
    AddEdge {
        /// Input JSON file
        input: PathBuf,

        /// Source node
        from: String,

        /// Target node
        to: String,
    },
}

fn read_input(input: &str) -> Result<String> {
    if input == "-" {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        Ok(buf)
    } else {
        let path = validate_path(Path::new(input), true)?;
        Ok(std::fs::read_to_string(path)?)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Check { input } => {
            let json = read_input(&input)?;
            match Dag::from_json(&json) {
                Ok(dag) => {
                    println!("\x1b[32m✓ Valid DAG\x1b[0m");
                    println!("  Nodes: {}", dag.node_count());
                    println!("  Edges: {}", dag.edge_count());
                    std::process::exit(0);
                }
                Err(dag_cli::DagError::CycleDetected { cycle }) => {
                    println!("\x1b[31m✗ Cycle detected\x1b[0m");
                    println!("  {}", cycle.join(" -> "));
                    std::process::exit(1);
                }
                Err(e) => {
                    println!("\x1b[31m✗ Error: {}\x1b[0m", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::Sort { input, json } => {
            let json_str = read_input(&input)?;
            let dag = Dag::from_json(&json_str)?;
            let order = dag.topological_sort()?;

            if json {
                println!("{}", serde_json::to_string_pretty(&order)?);
            } else {
                for (i, node) in order.iter().enumerate() {
                    println!("{:3}. {}", i + 1, node);
                }
            }
        }

        Commands::Viz { input, output, png } => {
            let json = read_input(&input)?;
            let dag = Dag::from_json(&json)?;
            let dot = dag.to_dot();

            if let Some(png_path) = png {
                // Validate output path before passing to external command
                let validated_path = validate_path(&png_path, false)?;
                // Pipe through dot command
                let mut child = std::process::Command::new("dot")
                    .args(["-Tpng", "-o"])
                    .arg(&validated_path)
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to run 'dot' command (install graphviz): {}", e)
                    })?;

                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(dot.as_bytes())?;
                }

                let status = child.wait()?;
                if status.success() {
                    println!("Written to {}", validated_path.display());
                } else {
                    anyhow::bail!("dot command failed");
                }
            } else if let Some(out_path) = output {
                let validated_out = validate_path(&out_path, false)?;
                std::fs::write(&validated_out, &dot)?;
                println!("Written to {}", validated_out.display());
            } else {
                print!("{}", dot);
            }
        }

        Commands::Info { input } => {
            let json = read_input(&input)?;
            let dag = Dag::from_json(&json)?;

            println!("\x1b[36mDAG Information\x1b[0m");
            println!("{}", "-".repeat(40));
            println!("  Nodes: {}", dag.node_count());
            println!("  Edges: {}", dag.edge_count());
            println!();

            let roots = dag.roots();
            let leaves = dag.leaves();

            println!("  Roots ({}): {}", roots.len(), roots.join(", "));
            println!("  Leaves ({}): {}", leaves.len(), leaves.join(", "));
            println!();

            println!("  Topological order:");
            for (i, node) in dag.topological_sort()?.iter().enumerate() {
                let deps = dag.dependencies(node).unwrap_or_default();
                if deps.is_empty() {
                    println!("    {:3}. {}", i + 1, node);
                } else {
                    println!("    {:3}. {} <- [{}]", i + 1, node, deps.join(", "));
                }
            }
        }

        Commands::Init { output, example } => {
            let spec = if example {
                DagSpec {
                    nodes: vec![
                        "requirement_a".into(),
                        "requirement_b".into(),
                        "implementation".into(),
                        "testing".into(),
                        "deployment".into(),
                    ],
                    edges: vec![
                        ("requirement_a".into(), "implementation".into()),
                        ("requirement_b".into(), "implementation".into()),
                        ("implementation".into(), "testing".into()),
                        ("testing".into(), "deployment".into()),
                    ],
                    metadata: Some(
                        [("name".into(), "Example Project DAG".into())]
                            .into_iter()
                            .collect(),
                    ),
                    node_meta: None,
                    edge_meta: None,
                }
            } else {
                DagSpec::new()
            };

            let json = spec.to_json()?;

            if let Some(out_path) = output {
                let validated_out = validate_path(&out_path, false)?;
                std::fs::write(&validated_out, &json)?;
                println!("Created {}", validated_out.display());
            } else {
                print!("{}", json);
            }
        }

        Commands::AddNode {
            input,
            name,
            probability,
        } => {
            let input_path = validate_path(Path::new(&input), true)?;
            let json = std::fs::read_to_string(&input_path)?;
            let mut spec = DagSpec::from_json(&json)?;

            if spec.nodes.contains(&name) {
                anyhow::bail!("Node '{}' already exists", name);
            }

            spec.nodes.push(name.clone());

            if let Some(prob) = probability {
                let node_meta = spec.node_meta.get_or_insert_with(Default::default);
                node_meta.insert(
                    name.clone(),
                    dag_cli::NodeMeta {
                        probability: Some(prob),
                        ..Default::default()
                    },
                );
            }

            // Validate still valid DAG
            Dag::from_spec(&spec)?;

            std::fs::write(&input_path, spec.to_json()?)?;
            println!("Added node '{}'", name);
        }

        Commands::AddEdge { input, from, to } => {
            let input_path = validate_path(Path::new(&input), true)?;
            let json = std::fs::read_to_string(&input_path)?;
            let mut spec = DagSpec::from_json(&json)?;

            if !spec.nodes.contains(&from) {
                anyhow::bail!("Source node '{}' not found", from);
            }
            if !spec.nodes.contains(&to) {
                anyhow::bail!("Target node '{}' not found", to);
            }

            spec.edges.push((from.clone(), to.clone()));

            // Validate still valid DAG (will error if cycle created)
            match Dag::from_spec(&spec) {
                Ok(_) => {
                    std::fs::write(&input_path, spec.to_json()?)?;
                    println!("Added edge {} -> {}", from, to);
                }
                Err(dag_cli::DagError::CycleDetected { cycle }) => {
                    anyhow::bail!(
                        "Cannot add edge: would create cycle ({})",
                        cycle.join(" -> ")
                    );
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    Ok(())
}
