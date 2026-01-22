# FRACTAL CISO Security Assessment

**Classification:** UNCLASSIFIED // FOUO
**Document Version:** 1.0
**Date:** January 2026
**Prepared For:** Chief Information Security Officers, Security Architects

---

## Executive Summary

FRACTAL is a structural alignment framework for AGI systems that implements defense-in-depth security controls across the software supply chain, authentication, authorization, audit, and operational security domains. This assessment provides a comprehensive security posture overview for CISO review prior to procurement or integration.

**Overall Security Rating:** HIGH (suitable for deployment in IL-4 environments with appropriate boundary controls)

---

## 1. Security Architecture Overview

### 1.1 Defense-in-Depth Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    NETWORK BOUNDARY                         │
│  (TLS 1.3, mTLS, NetworkPolicy, Ingress Controller)        │
├─────────────────────────────────────────────────────────────┤
│                    AUTHENTICATION                           │
│  (X.509 certificates, Argon2id/PBKDF2, HMAC tokens)        │
├─────────────────────────────────────────────────────────────┤
│                    AUTHORIZATION (RBAC)                     │
│  (Viewer < Operator < Admin < System)                      │
├─────────────────────────────────────────────────────────────┤
│                    CONTAINMENT LAYER                        │
│  (Intent classification, manipulation detection)            │
├─────────────────────────────────────────────────────────────┤
│                    AUDIT & MONITORING                       │
│  (Tamper-evident logs, SIEM integration, Prometheus)       │
├─────────────────────────────────────────────────────────────┤
│                    APPLICATION CORE                         │
│  (Memory-safe Rust, input validation, secrets management)  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Zero Trust Implementation

| Principle | Implementation |
|-----------|----------------|
| Never trust, always verify | All API calls require valid authentication token |
| Least privilege | RBAC with four authorization levels |
| Assume breach | Containment layer operates as internal IDS |
| Verify explicitly | Certificate chain validation, token expiration |
| Micro-segmentation | Network policies restrict pod-to-pod communication |

---

## 2. Authentication Controls

### 2.1 Password/API Key Security

| Control | Implementation | Standard |
|---------|----------------|----------|
| Hashing Algorithm | Argon2id (default) / PBKDF2-HMAC-SHA256 (FIPS) | OWASP 2024 |
| Iteration Count | 600,000 (PBKDF2) / 64MB memory + 3 iterations (Argon2) | NIST SP 800-132 |
| Salt | 16 bytes, cryptographically random | NIST SP 800-132 |
| Timing Attack Prevention | Constant-time comparison (subtle crate) | CWE-208 |

### 2.2 Certificate-Based Authentication

| Control | Implementation |
|---------|----------------|
| Certificate Validation | X.509 path validation with chain verification |
| Key Sizes | RSA 2048+ / ECDSA P-256+ |
| Revocation | CRL/OCSP support planned |
| mTLS | Supported via Kubernetes Ingress |

### 2.3 Session Management

| Control | Value | Justification |
|---------|-------|---------------|
| Session Duration | 1 hour (configurable) | NIST 800-63B |
| API Key Expiration | 90 days (high-security mode) | Rotation enforcement |
| Max Failed Attempts | 3-5 (configurable) | Brute force prevention |
| Lockout Duration | 15-60 minutes | Exponential backoff |

---

## 3. Authorization Controls

### 3.1 Role-Based Access Control

| Level | Permissions | Use Case |
|-------|-------------|----------|
| ReadOnly | Read metrics, view status | SOC analysts, monitoring |
| User | Basic operations | Standard users |
| Operator | Execute containment overrides | On-call engineers |
| Administrator | Full configuration, user management | System administrators |
| EmergencyOverride | Emergency access (time-limited) | Incident response |

### 3.2 Containment Layer Authorization

The containment layer provides an additional authorization check beyond RBAC:

- **Intent Classification:** Analyzes request intent for potential harm
- **Operator Verification:** Confirms human operator is legitimate
- **Manipulation Detection:** Identifies social engineering attempts
- **Boundary Enforcement:** Blocks requests outside defined operational envelope

---

## 4. Audit & Logging

### 4.1 Audit Trail Properties

| Property | Implementation |
|----------|----------------|
| Tamper Evidence | Hash chain (each entry references previous entry hash) |
| Integrity Verification | Chain verification via sequential hash validation |
| Timestamp Accuracy | Monotonic + wall clock (TimePoint) |
| Non-repudiation | Principal attribution on all events |

### 4.2 SIEM Integration

| Format | Compatible Systems |
|--------|-------------------|
| CEF (Common Event Format) | ArcSight, Splunk, QRadar |
| JSON (ECS Schema) | Elasticsearch, Azure Sentinel, Google Chronicle |
| OCSF | AWS Security Lake |

### 4.3 Auditable Events (NIST 800-53 AU-2)

- Authentication success/failure
- Authorization decisions
- Configuration changes
- Containment layer blocks
- Nociception pain events
- Orchestration consensus failures
- Security anomaly detection

---

## 5. Cryptographic Controls

### 5.1 Algorithms in Use

| Purpose | Algorithm | Key Size | Standard |
|---------|-----------|----------|----------|
| Password Hashing | Argon2id | N/A | OWASP 2024 |
| Password Hashing (FIPS) | PBKDF2-HMAC-SHA256 | N/A | FIPS 140-2 |
| Token Signing | HMAC-SHA256 | 256-bit | FIPS 198-1 |
| TLS | TLS 1.3 | N/A | FIPS 140-2 |
| Certificate Signing | ECDSA P-256 / RSA-2048 | 256/2048-bit | FIPS 186-4 |

### 5.2 FIPS 140-2 Compliance

FIPS mode is available via feature flag (`--features fips`):
- Disables Argon2id (not FIPS-approved)
- Uses PBKDF2-HMAC-SHA256 for password hashing
- Uses Ring library (BoringCrypto backend) for crypto primitives

---

## 6. Supply Chain Security

### 6.1 Dependency Management

| Control | Implementation |
|---------|----------------|
| SBOM Generation | CycloneDX + SPDX formats |
| Vulnerability Scanning | cargo-audit (RustSec advisory database) |
| License Compliance | cargo-deny configuration |
| Pinned Dependencies | Cargo.lock committed, reproducible builds |

### 6.2 Current Vulnerability Status

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 0 | N/A |
| High | 0 | N/A |
| Medium | 0 | N/A |
| Low | 0 | N/A |
| Unmaintained | 4 | Accepted (transitive, no security impact) |

### 6.3 Container Security

| Control | Implementation |
|---------|----------------|
| Base Image | Alpine Linux (minimal attack surface) |
| User | Non-root (UID 10001) |
| Filesystem | Read-only root |
| Capabilities | All dropped, none added |
| Image Signing | Cosign/Sigstore (planned Phase 5) |

---

## 7. Network Security

### 7.1 Exposed Services

| Port | Service | Protocol | Authentication |
|------|---------|----------|----------------|
| 8080 | HTTP API | HTTP | None (use reverse proxy for TLS) |
| 9090 | Metrics | HTTP | Network policy restricted |

**Note:** TLS termination should be handled by an external reverse proxy (nginx, envoy, ingress controller). The application server does not implement TLS directly.

### 7.2 Kubernetes Network Policies

- Ingress: Only from designated ingress controller
- Egress: DNS, required external services only
- Inter-pod: Denied by default

---

## 8. Compliance Mapping

### 8.1 NIST 800-53 Rev 5 Coverage

| Control Family | Coverage | Key Controls |
|----------------|----------|--------------|
| AC (Access Control) | High | AC-2, AC-3, AC-6, AC-7 |
| AU (Audit) | High | AU-2, AU-3, AU-6, AU-9 |
| CA (Assessment) | Medium | CA-7 (continuous monitoring) |
| CM (Configuration) | High | CM-2, CM-3, CM-7, CM-8 |
| IA (Identification) | High | IA-2, IA-5, IA-8 |
| SC (System/Comms) | High | SC-8, SC-13, SC-28 |
| SI (System Integrity) | High | SI-2, SI-3, SI-4, SI-7 |

### 8.2 STIG Compliance

See `docs/STIG_COMPLIANCE.md` for detailed STIG checklist mapping.

---

## 9. Risk Assessment

### 9.1 Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Credential stuffing | Medium | High | Rate limiting, account lockout |
| Supply chain compromise | Low | Critical | SBOM, signature verification, pinned deps |
| Container escape | Low | Critical | Non-root, read-only fs, minimal capabilities |
| Insider threat | Medium | High | RBAC, audit logging, separation of duties |
| Model manipulation | Medium | High | Containment layer, intent classification |

### 9.2 Residual Risk Acceptance

The following residual risks require acceptance:
1. Transitive dependency vulnerabilities (monitored via cargo-audit)
2. Zero-day vulnerabilities in Rust standard library (mitigated by memory safety)
3. Side-channel attacks on cryptographic operations (mitigated by constant-time ops)

---

## 10. Recommendations

### 10.1 Pre-Deployment Checklist

- [ ] Configure TLS certificates from PKI
- [ ] Enable mTLS for inter-service communication
- [ ] Integrate with enterprise identity provider (SAML/OIDC)
- [ ] Configure SIEM log forwarding
- [ ] Deploy network policies appropriate to environment
- [ ] Review and approve RBAC role assignments
- [ ] Conduct penetration test against deployed instance

### 10.2 Ongoing Security Operations

- Weekly: Review security alerts and audit logs
- Monthly: Run vulnerability scans, review dependency updates
- Quarterly: Red team exercise, tabletop incident response
- Annually: Third-party penetration test, compliance audit

---

## Appendix A: Security Contacts

| Role | Contact |
|------|---------|
| Security Team | security@fractal.ai |
| Vulnerability Disclosure | security@fractal.ai (PGP key available) |
| Compliance Questions | compliance@fractal.ai |

---

## Appendix B: Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 2026 | Security Team | Initial release |

---

*This document should be reviewed quarterly or when significant changes occur.*
