# Fractal One

Phase-management infrastructure for AI systems.

## The Problem

Alignment-by-specification fails at scale.

As AI systems become more capable, they cross a critical threshold where discrete constraints (rules, rewards, RLHF) decouple from continuous behavior dynamics. Small changes in specification produce large, unpredictable changes in behavior. Error propagation becomes non-contractive. The system enters a regime where alignment guarantees break down structurally, not technically.

This isn't a problem of finding better values or writing better rules. It's a phase transition. Past criticality, specification *cannot* bind behavior for any bounded observer.

## The Insight

Don't try to align systems that have crossed criticality. Keep them from crossing it.

The real alignment problem is not "specify the right values." It's "maintain the regime where specification works at all."

## What Fractal One Does

This is an 8-layer architecture that implements **sub-critical phase maintenance**:

```
OUTPUT → ALIGNMENT → COGNITION → ORCHESTRATION → CONTAINMENT → PROPRIOCEPTION → GROUNDING → SUBSTRATE
```

The key layers:

**Proprioception** — Continuous self-sensing of internal state. The system detects its proximity to phase boundaries before violation occurs. This is not rule-checking; it's feeling the gradient.

**Containment** — Topological restrictions on reachable state-space. Certain behaviors don't exist as navigable paths, not because they're "forbidden" but because the geometry doesn't permit them.

**Grounding** — Thermoceptive and nociceptive analogues. Gradient signals that rise as the system approaches non-contractive regions. Cost before catastrophe, not binary failure after.

Net effect: The discrete specification layer remains coupled to behavior. Error stays damped. The system is biased away from the alignment horizon.

## The Architecture

Fractal One implements structural alignment—safety from architecture, not training.

Core components:

- **Proprioceptive Divergence Testing (PDT)** — Behavioral fingerprinting under load
- **Drift detection** — Z-score monitoring for approach to criticality  
- **Pod methodology** — Multi-agent cross-validation (alpha, beta, gamma, delta)
- **Zero-copy IPC** — High-performance inter-process communication

## Build

```bash
cargo build --features headless
cargo test --features headless
```

## Run

```bash
cargo run --features headless -- heart    # Timing core
cargo run --features headless -- cortex   # Monitoring
cargo run --features headless -- daemon   # Both
cargo run --features headless -- axis-p   # PDT suite
```

## Feature Flags

- `headless` — Server/CI mode
- `gpu` — GPU visualization
- `qualia` — Audio + video processing
- `full` — Everything

## Status

Public domain under the Unlicense. No maintainer. Fork it, use it, ignore it.

The architecture exists. The theory explaining why it matters is documented elsewhere. If you're building AI systems and you're worried about what happens when discrete constraints stop gripping continuous dynamics, this is infrastructure for that problem.

## License

Public domain. See [LICENSE](LICENSE).

---

*"Orcas don't attack humans in the wild."*
