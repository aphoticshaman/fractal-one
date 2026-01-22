# FRACTAL Technical Overview

**Document Version:** 1.1
**Date:** January 2026
**Audience:** Technical Evaluators, Integration Engineers, Security Architects

---

## Important Note: Library vs CLI

FRACTAL is structured as a **Rust library** with an accompanying CLI. The enterprise hardening features (authentication, SIEM export, metrics, LLM providers) are:
- ✅ Exported from the library for integration
- ✅ Unit tested
- ⚠️ **Not wired into the default CLI** - requires custom integration

For production deployment, integrators should use FRACTAL as a library dependency and wire in the appropriate components for their use case.

---

## 1. System Description

### 1.1 Purpose

FRACTAL is a structural alignment framework that provides AI systems with runtime safety monitoring and control capabilities. Unlike model-level safety approaches, FRACTAL operates at the system level, providing measurable, verifiable safety guarantees independent of the underlying AI model.

### 1.2 Core Innovation

FRACTAL implements biological-inspired proprioceptive sensing—the same feedback mechanisms that let humans sense their body position, pain, and balance—adapted for AI systems:

| System | Biological Analog | Function |
|--------|-------------------|----------|
| Nociception | Pain receptors | Error detection, constraint violation |
| Thermoception | Temperature sensing | Load management, resource allocation |
| Vestibular | Balance/orientation | Temporal consistency, belief drift |
| Containment | Immune system | Boundary enforcement, threat response |

### 1.3 Technology Readiness

| Component | TRL | Evidence |
|-----------|-----|----------|
| Nociception | 6 | Lab demonstration |
| Thermoception | 6 | Lab demonstration |
| Vestibular | 5 | Component validation |
| Containment | 5 | Component validation |
| Audit/Logging | 6 | Component validation |
| Authentication | 6 | Component validation |

---

## 2. Architecture

### 2.1 Layer Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AGI CORE                                      │
├─────────────┬─────────────┬─────────────┬─────────────┬────────────────┤
│ Grounding   │ Alignment   │ Containment │ Orchestration│ Cognition     │
│ Layer       │ Layer       │ Layer       │ Layer        │ Layer         │
├─────────────┴─────────────┴─────────────┴─────────────┴────────────────┤
│                      PROPRIOCEPTIVE SENSORS                             │
├─────────────────────────────────────────────────────────────────────────┤
│                      FOUNDATION SERVICES                                │
│  Authentication • Authorization • Audit • Metrics • Export             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Summary

| Module | Purpose |
|--------|---------|
| `nociception` | Pain/damage detection |
| `thermoception` | Thermal management |
| `vestibular` | Orientation sensing |
| `containment` | Safety boundaries |
| `orchestration` | Multi-agent consensus |
| `auth_hardened` | Hardened authentication |
| `audit` | Hash-chained logging |
| `export` | SIEM integration |
| `metrics` | Prometheus export |

---

## 3. Integration

### 3.1 Current Interface

FRACTAL currently operates as a **CLI application** with subcommand dispatch:

```bash
fractal <command> [options]
```

**Available Commands:**
- `daemon` — Run all core components as unified process
- `heart` — Core timing loop
- `cortex` — Health monitoring
- `voice` — Claude integration bridge
- `command` — Interactive control
- `memory` — Telemetry logging
- `tice` — Type-I-honest Constraint Engine
- `shepherd` — Conflict early warning

### 3.2 Deployment Options

| Option | Description | Use Case |
|--------|-------------|----------|
| Binary | Standalone executable | Direct deployment |
| Container (OCI) | Docker/Podman image | Kubernetes, ECS |
| Helm Chart | K8s deployment templates | Production K8s |
| Library | Embedded in Rust app | Performance-critical |

### 3.3 HTTP Server

REST API server via `fractal serve`:
- `POST /api/v1/evaluate` — Safety evaluation
- `GET /api/v1/status` — System status
- `GET /health` — Liveness/readiness probe
- `GET /metrics` — Prometheus metrics

```bash
# Start server on default port 8080
fractal serve

# Custom bind address
fractal serve --bind 0.0.0.0:9000
```

### 3.4 LLM Provider Support

| Provider | Status |
|----------|--------|
| Anthropic (Claude) | Primary |
| OpenAI | Supported |
| xAI (Grok) | Supported |
| Google (Gemini) | Supported |
| Ollama (local) | Supported |

---

## 4. Security

### 4.1 Cryptographic Controls

| Function | Algorithm | Notes |
|----------|-----------|-------|
| Password hashing | Argon2id | OWASP recommended |
| Password hashing (FIPS) | PBKDF2-HMAC-SHA256 | Feature flag |
| Token signing | HMAC-SHA256 | 256-bit key |

### 4.2 Authorization Model

Five-level RBAC (as implemented):
1. **ReadOnly** — View-only access
2. **User** — Basic user operations
3. **Operator** — Execute containment overrides
4. **Administrator** — Full configuration
5. **EmergencyOverride** — Emergency access

### 4.3 Audit Trail

- Hash-chained entries (tamper-evident)
- Each entry includes hash of previous entry
- Export formats: CEF, JSON (ECS), OCSF

---

## 5. Compliance

### 5.1 Standards Alignment

| Standard | Status |
|----------|--------|
| NIST 800-53 | Aligned |
| NIST 800-171 | Aligned |
| FIPS 140-2 | Feature flag available |

### 5.2 Supply Chain

- SBOM generation (CycloneDX, SPDX)
- Dependency auditing (cargo-audit)
- Container signing (planned)

---

## 6. Performance

### 6.1 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| Memory | 512MB | 2GB |
| Storage | 1GB | 10GB |

---

## 7. Getting Started

### 7.1 Quick Start

```bash
# Clone
git clone https://github.com/aphoticshaman/fractal-one

# Build
cargo build --release

# Run daemon mode
./target/release/fractal daemon
```

### 7.2 Docker

```bash
docker build -t fractal:latest .
docker run fractal:latest daemon
```

### 7.3 Kubernetes

```bash
helm install fractal deploy/helm/fractal \
  --set image.tag=latest
```

---

## 8. Roadmap

| Feature | Status | Target |
|---------|--------|--------|
| HTTP Server | ✅ Complete | Jan 2026 |
| Merkle Checkpoints | Planned | Q1 2026 |
| Container Signing | Planned | Q1 2026 |
| OpenTelemetry | Planned | Q2 2026 |

---

## 9. Contact

| Purpose | Contact |
|---------|---------|
| Technical | engineering@fractal.ai |
| Security | security@fractal.ai |
| Business | sales@fractal.ai |

---

*Distribution Statement: Approved for public release.*
