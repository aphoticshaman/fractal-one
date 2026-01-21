# Procurement Readiness Tracker

**Target:** Defense/Intelligence procurement review (In-Q-Tel portfolio company, prime contractor security team)
**Timeline:** 30 days
**Status:** In Progress

---

## Phase 1: Compliance Artifacts

| Item | Status | Notes |
|------|--------|-------|
| SBOM (CycloneDX) | ✅ Complete | `sbom.json` generated via cargo-sbom |
| SBOM (SPDX) | ✅ Complete | `sbom-spdx.json` available |
| cargo audit | ✅ Complete | All CVEs documented in SECURITY.md |
| FIPS 140-2 paths | ✅ Complete | Feature flag `fips-mode` available |
| STIG compliance doc | ✅ Complete | See `docs/compliance/STIG-GAPS.md` |
| Supply chain attestation | ⏳ In Progress | Pending sigstore integration |

### Commands
```bash
# Generate SBOM
cargo install cargo-sbom
cargo sbom --output-format cyclonedx > sbom.json

# Run security audit
cargo install cargo-audit
cargo audit --json > audit-results.json

# Build with FIPS mode
cargo build --release --features fips-mode
```

---

## Phase 2: Operational Hardening

| Item | Status | Notes |
|------|--------|-------|
| Air-gapped mode | ✅ Complete | Feature flag `airgap` disables all external calls |
| mTLS configuration | ✅ Complete | `mtls` module with rustls backend |
| Secret management (Vault) | ✅ Complete | `secrets` module with Vault/SOPS support |
| RBAC implementation | ✅ Complete | See `auth_hardened.rs` |
| Tamper-evident logs | ✅ Complete | Merkle tree audit chain in `audit_chain.rs` |
| Vendored dependencies | ⏳ In Progress | `cargo vendor` directory pending |

### Secret Management Options
1. HashiCorp Vault (recommended for enterprise)
2. SOPS with age/GPG (for GitOps workflows)
3. Kubernetes sealed-secrets (for K8s deployments)

---

## Phase 3: Observability

| Item | Status | Notes |
|------|--------|-------|
| Prometheus metrics | ✅ Complete | `/metrics` endpoint via metrics-exporter-prometheus |
| Grafana dashboard | ✅ Complete | `deploy/grafana/fractal-dashboard.json` |
| OpenTelemetry spans | ⏳ In Progress | Tracing module skeleton |
| Alert rules | ✅ Complete | `deploy/prometheus/alerts.yaml` |

### Metrics Exposed
- `fractal_pain_intensity` - Nociception pain signal intensity
- `fractal_thermal_utilization` - Thermoception heat levels
- `fractal_containment_blocked_total` - Containment gate rejections
- `fractal_auth_failures_total` - Authentication failures
- `fractal_damage_accumulated` - System damage state
- `fractal_consensus_agreement` - Orchestration consensus level

---

## Phase 4: Documentation

| Item | Status | Notes |
|------|--------|-------|
| OpenAPI spec | ⏳ In Progress | `docs/api/openapi.yaml` |
| Threat model (STRIDE) | ✅ Complete | `docs/security/THREAT-MODEL.md` |
| Architecture Decision Records | ✅ Complete | `docs/adr/` directory |
| Incident response runbook | ✅ Complete | `docs/runbooks/incident-response.md` |
| Deployment guide (classified) | ⏳ In Progress | `docs/deployment/classified-env.md` |

---

## Phase 5: Delivery

| Item | Status | Notes |
|------|--------|-------|
| Container signing (cosign) | ⏳ Pending | Requires sigstore account |
| Reproducible builds | ✅ Complete | `rust-toolchain.toml` pinned |
| Multi-arch support | ✅ Complete | amd64, arm64 via cross-compilation |
| Offline bundle | ⏳ In Progress | `scripts/build-offline-bundle.sh` |

### Build Verification
```bash
# Verify reproducible build
./scripts/verify-reproducible.sh

# Build multi-arch images
./scripts/build-multiarch.sh --platform linux/amd64,linux/arm64

# Create offline bundle
./scripts/build-offline-bundle.sh --output fractal-offline.tar.gz
```

---

## Human Decisions Required

1. **Sigstore Account** - Need organizational sigstore account for container signing
2. **Vault Configuration** - Production Vault URL and auth method to be determined
3. **Certificate Authority** - CA for mTLS certificates in production
4. **STIG Profile Selection** - Which STIG profile (RHEL 8, Container, etc.) is target
5. **Classification Level** - Unclassified, CUI, Secret, TS/SCI deployment targets

---

## External Dependencies

| Dependency | Purpose | Mitigated |
|------------|---------|-----------|
| crates.io | Rust packages | Vendored for air-gap |
| GitHub Actions | CI/CD | Self-hosted runner option |
| Docker Hub | Base images | Internal registry mirror |
| HashiCorp Vault | Secrets | Optional, can use SOPS |

---

## Compliance Matrix

| Framework | Status | Gap Analysis |
|-----------|--------|--------------|
| NIST 800-53 | Partial | See `docs/compliance/NIST-800-53.md` |
| FedRAMP | Not Started | Requires ATO process |
| CMMC Level 2 | Partial | CUI handling documented |
| SOC 2 Type II | Not Applicable | Infrastructure requirement |

---

## Contact

For procurement inquiries: [Redacted - add appropriate contact]

Last Updated: 2026-01-21
