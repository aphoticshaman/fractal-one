# Procurement Readiness Tracker

**Target:** Defense/Intelligence procurement review
**Status:** Early Stage - Library Components Available, Integration Required

---

## ⚠️ IMPORTANT: Library vs Application

FRACTAL is structured as a **Rust library** with CLI examples. Enterprise features exist as **library components** that require integration into a deployment-specific application.

**What this means for procurement:**
- Core algorithms: ✅ Ready for evaluation
- Enterprise hardening: ⚠️ Library components, need integration
- Production deployment: ⚠️ Requires custom application wrapper

---

## Phase 1: Compliance Artifacts

| Item | Actual Status | Notes |
|------|---------------|-------|
| SBOM script | ✅ Available | `scripts/generate-sbom.sh` - requires cargo-sbom tool |
| cargo-deny config | ✅ Available | `deny.toml` for supply chain policy |
| cargo audit | ✅ Passing | 0 critical CVEs, 4 unmaintained warnings (accepted) |
| FIPS feature flag | ✅ Available | `--features fips` enables PBKDF2 instead of Argon2 |
| STIG compliance doc | ✅ Available | `docs/STIG_COMPLIANCE.md` |

### To Generate SBOM
```bash
cargo install cargo-sbom cargo-cyclonedx
./scripts/generate-sbom.sh
```

---

## Phase 2: Enterprise Hardening (Library Components)

| Item | Library Status | CLI Integration |
|------|----------------|-----------------|
| Hardened Auth (Argon2id) | ✅ `auth_hardened.rs` | ❌ Not wired |
| X.509 Certificate Validation | ✅ `CertificateValidator` | ❌ Not wired |
| SIEM Export (CEF) | ✅ `export/cef.rs` | ❌ Not wired |
| SIEM Export (JSON/ECS) | ✅ `export/json.rs` | ❌ Not wired |
| SIEM Export (OCSF) | ✅ `export/ocsf.rs` | ❌ Not wired |
| Prometheus Metrics | ✅ `metrics.rs` | ✅ `fractal serve` |
| HTTP Server | ✅ `server.rs` | ✅ `fractal serve` |
| LLM Provider Config | ✅ `llm_providers.rs` | ❌ Not wired |
| Audit Trail (Hash Chain) | ✅ `audit.rs` | ✅ Used in TICE |

### What "Library Component" Means
```rust
// These are available for use:
use fractal::{HardenedAuthProvider, CefExporter, MetricsRegistry};

// But NOT automatically active in the CLI
// Integrators must wire them into their application
```

---

## Phase 3: Observability

| Item | Actual Status | Notes |
|------|---------------|-------|
| Prometheus metric types | ✅ Library code | Counter, Gauge, Histogram implemented |
| Grafana dashboard JSON | ✅ Available | `deploy/grafana/fractal-dashboard.json` |
| Prometheus alert rules | ✅ Available | `deploy/prometheus/alerts.yaml` |
| HTTP metrics endpoint | ✅ Implemented | `fractal serve` exposes GET /metrics |
| Health check endpoint | ✅ Implemented | GET /health for k8s probes |
| Status API | ✅ Implemented | GET /api/v1/status |
| OpenTelemetry | ❌ Not implemented | Planned |

---

## Phase 4: Documentation

| Item | Status | Location |
|------|--------|----------|
| Technical Overview | ✅ Complete | `docs/TECHNICAL_OVERVIEW.md` |
| CISO Security Assessment | ✅ Complete | `docs/CISO_SECURITY_ASSESSMENT.md` |
| Threat Model (STRIDE) | ✅ Complete | `docs/THREAT_MODEL.md` |
| STIG Compliance Mapping | ✅ Complete | `docs/STIG_COMPLIANCE.md` |
| Executive Summary | ✅ Complete | `docs/EXECUTIVE_SUMMARY.md` |

---

## Phase 5: Delivery Artifacts

| Item | Status | Notes |
|------|--------|-------|
| Dockerfile | ✅ Available | Multi-stage, non-root user |
| Helm Chart | ✅ Available | `deploy/helm/fractal/` |
| Demo Dashboard | ✅ Available | `demo/dashboard.html` (air-gap capable) |
| Container signing | ❌ Not implemented | Planned |
| Reproducible builds | ⚠️ Partial | Cargo.lock committed, no CI verification |

---

## What's Actually Working in the CLI

The following subsystems are **actively used** in the current CLI:

| Subsystem | CLI Command | Status |
|-----------|-------------|--------|
| HTTP Server | `fractal serve` | ✅ Working |
| Nociception | `fractal test noci` | ✅ Working |
| Thermoception | `fractal test thermo` | ✅ Working |
| Containment | Library only | ⚠️ Not exposed |
| Orchestration | Library only | ⚠️ Not exposed |
| Voice Bridge (Claude) | `fractal voice` | ✅ Working |
| TICE | `fractal tice` | ✅ Working |
| Shepherd | `fractal shepherd` | ✅ Working |
| Audit Trail | Internal | ✅ Working |

### Server Endpoints (`fractal serve`)
```
GET  /health           - Liveness/readiness probe
GET  /metrics          - Prometheus metrics
GET  /api/v1/status    - System status JSON
POST /api/v1/evaluate  - Safety evaluation (basic)
```

---

## Honest Assessment for Evaluators

**Strengths:**
- Novel proprioceptive approach to AI safety
- Clean Rust codebase with good modular structure
- Core algorithms implemented and tested
- Enterprise security patterns available as library components
- HTTP server with metrics/health/API endpoints

**Gaps for Production:**
- Enterprise hardening (auth, SIEM export) not wired into CLI
- No mTLS module (would require rustls integration)
- No secret management integration (Vault/SOPS)
- No CI/CD pipeline for signing

**Recommendation:** Evaluate the core algorithms via CLI and `fractal serve`. Enterprise hardening features require integration work for production deployment.

---

## Contact

For technical evaluation: engineering@fractal.ai
