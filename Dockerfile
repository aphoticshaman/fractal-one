# ═══════════════════════════════════════════════════════════════════════════════
# FRACTAL ONE — Multi-stage Production Container
# ═══════════════════════════════════════════════════════════════════════════════
# TRL 6+ deployment-ready container with security hardening
#
# Build:   docker build -t fractal:latest .
# Run:     docker run -d --name fractal -p 8080:8080 -p 9090:9090 fractal:latest
# ═══════════════════════════════════════════════════════════════════════════════

# Stage 1: Build environment
FROM rust:1.75-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files first (for caching)
COPY Cargo.toml Cargo.lock ./
COPY dag-cli/Cargo.toml dag-cli/
COPY fractal_agent/Cargo.toml fractal_agent/
COPY neuro_link/Cargo.toml neuro_link/

# Create dummy sources for dependency caching
RUN mkdir -p src dag-cli/src fractal_agent/src neuro_link/src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    echo "pub fn dummy() {}" > dag-cli/src/lib.rs && \
    echo "pub fn dummy() {}" > fractal_agent/src/lib.rs && \
    echo "pub fn dummy() {}" > neuro_link/src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release --features headless 2>/dev/null || true

# Copy actual source code
COPY src/ src/
COPY dag-cli/ dag-cli/
COPY fractal_agent/ fractal_agent/
COPY neuro_link/ neuro_link/

# Touch files to ensure rebuild
RUN touch src/main.rs src/lib.rs

# Build release binary
RUN cargo build --release --features headless

# Stage 2: Runtime environment
FROM debian:bookworm-slim AS runtime

# Security: Create non-root user
RUN groupadd -r fractal && useradd -r -g fractal fractal

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy binary
COPY --from=builder /build/target/release/fractal /usr/local/bin/fractal

# Create directories
RUN mkdir -p /var/lib/fractal /var/log/fractal /etc/fractal && \
    chown -R fractal:fractal /var/lib/fractal /var/log/fractal /etc/fractal

# Copy default configuration
COPY deploy/config/default.toml /etc/fractal/config.toml

# Security hardening
RUN chmod 755 /usr/local/bin/fractal && \
    chmod 700 /var/lib/fractal

# Switch to non-root user
USER fractal

# Environment variables
ENV FRACTAL_CONFIG=/etc/fractal/config.toml
ENV FRACTAL_DATA=/var/lib/fractal
ENV FRACTAL_LOG=/var/log/fractal
ENV RUST_LOG=info

# Expose ports
# 8080: HTTP API
# 9090: Prometheus metrics
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/fractal", "sitrep", "--health-check"]

# Default command
ENTRYPOINT ["/usr/local/bin/fractal"]
CMD ["daemon", "--config", "/etc/fractal/config.toml"]

# Labels for container registry
LABEL org.opencontainers.image.title="Fractal One" \
      org.opencontainers.image.description="Structural alignment framework for AGI systems" \
      org.opencontainers.image.vendor="Fractal" \
      org.opencontainers.image.version="0.6.0" \
      org.opencontainers.image.source="https://github.com/aphoticshaman/fractal-one"
