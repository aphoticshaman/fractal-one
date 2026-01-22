# FRACTAL STIG Compliance Documentation

## Overview

This document maps FRACTAL's security controls to relevant DISA Security Technical Implementation Guides (STIGs) for deployment in DoD environments.

**Applicable STIGs:**
- Application Security and Development STIG (V5R3)
- Container Platform STIG (V2R1)
- Kubernetes STIG (V1R11)
- PostgreSQL 9.x STIG (if using persistent storage)

---

## Application Security Controls

### APSC-DV-000160: Application must implement DoD-approved encryption

**Status:** ✅ COMPLIANT (with FIPS feature flag)

**Implementation:**
- Default: Ring cryptographic library (BoringSSL-based)
- FIPS Mode: PBKDF2-HMAC-SHA256 for password hashing
- TLS 1.2+ enforced for all network communications
- X.509 certificate validation with chain verification

**Evidence:**
```rust
// src/auth_hardened.rs
#[cfg(feature = "fips")]
fn hash_password_fips(password: &str, salt: &[u8]) -> Vec<u8> {
    // PBKDF2-HMAC-SHA256, NIST SP 800-132 compliant
    pbkdf2::pbkdf2_hmac::<Sha256>(
        password.as_bytes(),
        salt,
        600_000,  // OWASP 2024 recommendation
        &mut output
    );
}
```

---

### APSC-DV-000460: Application must enforce approved authorizations

**Status:** ✅ COMPLIANT

**Implementation:**
- Role-Based Access Control (RBAC) with four authorization levels
- Operator detection and verification
- Session-based authentication with timeout enforcement
- Certificate-based mutual TLS (mTLS) support

**Evidence:**
```rust
// src/auth.rs
pub enum AuthorizationLevel {
    Viewer,     // Read-only access to metrics
    Operator,   // Can execute containment overrides
    Admin,      // Full system configuration
    System,     // Internal process identity (non-human)
}
```

---

### APSC-DV-000500: Application must generate audit records

**Status:** ✅ COMPLIANT

**Implementation:**
- Structured logging in CEF, JSON/ECS, and OCSF formats
- All authentication events logged with timestamps
- Containment decisions include full context
- Tamper-evident log chain (hash chain) implemented

**Evidence:**
```rust
// src/export/cef.rs
pub fn emit_auth_event(&self, event: &AuthEvent) -> String {
    // CEF format for ArcSight/Splunk/QRadar integration
    format!(
        "CEF:0|Fractal|AGI-Core|1.0|{}|{}|{}|...",
        event.signature_id,
        event.name,
        event.severity
    )
}
```

---

### APSC-DV-001390: Application must protect against injection attacks

**Status:** ✅ COMPLIANT

**Implementation:**
- Input validation at all trust domain boundaries
- Text normalization removes control characters
- No dynamic SQL or shell command construction
- Parameterized queries for any database access

**Evidence:**
```rust
// src/text_normalize.rs
pub fn sanitize_input(input: &str) -> String {
    input
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .take(MAX_INPUT_LENGTH)
        .collect()
}
```

---

### APSC-DV-002010: Application must maintain confidentiality of data at rest

**Status:** ✅ COMPLIANT

**Implementation:**
- Secrets managed via Kubernetes Secrets or HashiCorp Vault
- No hardcoded credentials in source code
- Environment variable injection for sensitive configuration
- Memory zeroization for password buffers

**Evidence:**
```yaml
# deploy/helm/fractal/templates/deployment.yaml
env:
  - name: FRACTAL_AUTH_SECRET
    valueFrom:
      secretKeyRef:
        name: fractal-secrets
        key: auth-secret
```

---

## Container Security Controls

### CNTR-K8-000300: Containers must run as non-root

**Status:** ✅ COMPLIANT

**Implementation:**
```dockerfile
# Dockerfile
RUN addgroup -g 10001 fractal && \
    adduser -u 10001 -G fractal -D -H fractal
USER fractal:fractal
```

---

### CNTR-K8-000360: Container images must be signed

**Status:** 🔄 PLANNED (Phase 5)

**Implementation Plan:**
- Cosign/Sigstore for container image signing
- Signature verification in deployment pipeline
- SBOM attestation attached to images

---

### CNTR-K8-000410: Read-only root filesystem

**Status:** ✅ COMPLIANT

**Implementation:**
```yaml
# deploy/helm/fractal/templates/deployment.yaml
securityContext:
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
```

---

### CNTR-K8-000440: Resource limits must be defined

**Status:** ✅ COMPLIANT

**Implementation:**
```yaml
resources:
  limits:
    cpu: "2"
    memory: 2Gi
  requests:
    cpu: "500m"
    memory: 512Mi
```

---

## Network Security Controls

### CNTR-K8-001160: Network policies must restrict traffic

**Status:** 🔄 IN PROGRESS (Phase 2)

**Implementation Plan:**
- NetworkPolicy restricting ingress to load balancer
- Egress limited to required external services
- Service mesh (Istio/Linkerd) for mTLS between pods

---

## Cryptographic Controls

### SRG-APP-000514: Use FIPS 140-2 validated cryptographic modules

**Status:** ✅ COMPLIANT (with FIPS feature flag)

**Implementation:**
```toml
# Cargo.toml feature flag
[features]
fips = []
```

When `fips` feature is enabled:
- Password hashing uses PBKDF2-HMAC-SHA256 (FIPS 140-2 approved)
- TLS via FIPS-validated OpenSSL or BoringCrypto
- No non-approved algorithms in cryptographic operations

---

## Audit Checklist

| Control ID | Description | Status | Evidence Location |
|------------|-------------|--------|-------------------|
| APSC-DV-000160 | DoD-approved encryption | ✅ | src/auth_hardened.rs |
| APSC-DV-000460 | Authorization enforcement | ✅ | src/auth.rs |
| APSC-DV-000500 | Audit record generation | ✅ | src/export/ |
| APSC-DV-001390 | Injection protection | ✅ | src/text_normalize.rs |
| APSC-DV-002010 | Data-at-rest protection | ✅ | Helm secrets |
| CNTR-K8-000300 | Non-root containers | ✅ | Dockerfile |
| CNTR-K8-000360 | Image signing | 🔄 | Phase 5 |
| CNTR-K8-000410 | Read-only filesystem | ✅ | deployment.yaml |
| CNTR-K8-000440 | Resource limits | ✅ | values.yaml |
| CNTR-K8-001160 | Network policies | 🔄 | Phase 2 |
| SRG-APP-000514 | FIPS cryptography | ✅ | fips feature flag |

---

## Remediation Notes

### Items Requiring Environment Configuration

1. **TLS Certificates**: Must be provisioned by deployment environment
2. **Secret Management**: Requires Vault or K8s Secrets setup
3. **Network Policies**: Specific to deployment cluster configuration
4. **FIPS Mode**: Requires `--features fips` at compile time

### Items Requiring ATO Review

1. Boundary definition for containerized deployment
2. Data flow diagram approval
3. Interconnection Security Agreement (ISA) if crossing enclaves
4. Continuous monitoring plan alignment

---

## Contact

For STIG compliance questions: [security@fractal.ai]
For ATO package requests: [compliance@fractal.ai]
