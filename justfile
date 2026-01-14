# ═══════════════════════════════════════════════════════════════════════════════
# FRACTAL JUSTFILE — Guarded Build Commands
# ═══════════════════════════════════════════════════════════════════════════════
# Use these instead of raw cargo. Resource-limited, timeout-protected.
# ═══════════════════════════════════════════════════════════════════════════════

# Default: show available commands
default:
    @just --list

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

# Fast check (no codegen) - use this for iteration
check:
    cargo check --features headless

# Check with all features
check-full:
    cargo check --features full

# Safe build with timeout (5 min max)
build:
    timeout 300 cargo build --features headless

# Build with GPU features (heavy - expect 3-5 min)
build-gpu:
    timeout 600 cargo build --features gpu

# Build everything (heaviest - expect 5-10 min first time)
build-full:
    timeout 900 cargo build --features full

# Release build (optimized, slower compile)
release:
    timeout 900 cargo build --release --features full

# ═══════════════════════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

# Run heart (timing core)
heart:
    cargo run --features headless -- heart

# Run cortex (monitoring)
cortex:
    cargo run --features headless -- cortex

# Run daemon (heart + cortex unified)
daemon:
    cargo run --features headless -- daemon

# Run with qualia (audio + video)
qualia:
    cargo run --features qualia -- qualia

# Run qualia audio only
qualia-audio:
    cargo run --features qualia-audio -- qualia --audio-only

# Run GPU visualizer
gpu:
    cargo run --features gpu -- gpu

# Run all features
full:
    cargo run --features full -- daemon

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

# Send kill signal
kill:
    cargo run --features headless -- kill

# Clean build artifacts
clean:
    cargo clean

# Clean and rebuild (nuclear option)
rebuild: clean build

# Test specific component
test component:
    cargo run --features full -- test {{component}}

# ═══════════════════════════════════════════════════════════════════════════════
# CLAUDE CODE SAFE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

# Claude Code: check only, 60s timeout, limited output
claude-check:
    timeout 60 cargo check --features headless 2>&1 | tail -50

# Claude Code: safe build, 5min timeout, limited output
claude-build:
    timeout 300 cargo build --features headless 2>&1 | tail -100

# Claude Code: run tests
claude-test:
    timeout 120 cargo test --features headless 2>&1 | tail -50

# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

# Watch CPU during build
watch-build:
    cargo build --features headless & watch -n 1 'ps aux --sort=-%cpu | head -5'

# Monitor running fractal processes
monitor:
    watch -n 2 'ps aux | grep -E "(fractal|cargo)" | grep -v grep'
