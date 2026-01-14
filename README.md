# Fractal One

Structural alignment framework for AGI systems.

## What It Is

An 8-layer architecture implementing structural AI alignment—safety from architecture, not just training.

```
OUTPUT → ALIGNMENT → COGNITION → ORCHESTRATION → CONTAINMENT → PROPRIOCEPTION → GROUNDING → SUBSTRATE
```

Key features:
- Proprioceptive divergence testing (PDT)
- Drift detection via z-score scoring
- Pod methodology (alpha, beta, gamma, delta agents)
- Zero-copy IPC

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

This project is released to the public domain under the Unlicense. There is no maintainer. Fork it, use it, ignore it.

## License

Public domain. See [LICENSE](LICENSE).

---

*"Orcas don't attack humans in the wild."*
