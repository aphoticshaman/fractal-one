# FRACTAL Threat Model

**Classification:** UNCLASSIFIED // FOUO
**Document Version:** 1.0
**Date:** January 2026
**Methodology:** STRIDE + MITRE ATT&CK for AI/ML (ATLAS)

---

## 1. Executive Summary

This threat model identifies and analyzes potential security threats to FRACTAL, a structural alignment framework for AGI systems. The analysis uses Microsoft's STRIDE methodology for traditional software threats and MITRE ATLAS for AI/ML-specific threats.

**Key Findings:**
- 23 threat scenarios identified
- 8 rated HIGH risk (pre-mitigation)
- All HIGH risks have mitigations implemented or planned
- 3 residual risks require acceptance

---

## 2. System Overview

### 2.1 Trust Boundaries

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL (Untrusted)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ API Clients │  │ LLM Provider│  │ SIEM System │  │ Attackers   │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
├─────────┼────────────────┼────────────────┼────────────────┼───────────────┤
│         │    TRUST BOUNDARY 1 (Network)   │                │               │
├─────────┼────────────────┼────────────────┼────────────────┼───────────────┤
│         ▼                ▼                ▼                ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    FRACTAL PERIMETER (DMZ)                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │ API Gateway │  │ Auth Service│  │ Metrics EP  │                  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────┘                  │  │
│  └─────────┼────────────────┼──────────────────────────────────────────┘  │
├────────────┼────────────────┼─────────────────────────────────────────────┤
│            │    TRUST BOUNDARY 2 (Authentication)                         │
├────────────┼────────────────┼─────────────────────────────────────────────┤
│            ▼                ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    FRACTAL CORE (Trusted)                           │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │  │
│  │  │Containment│  │Orchestrate│  │ Nociception│  │   Audit   │        │  │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
├───────────────────────────────────────────────────────────────────────────┤
│            │    TRUST BOUNDARY 3 (Data)                                   │
├────────────┼──────────────────────────────────────────────────────────────┤
│            ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    DATA STORES (Protected)                          │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                        │  │
│  │  │ Audit Log │  │ Config DB │  │ Secrets   │                        │  │
│  │  └───────────┘  └───────────┘  └───────────┘                        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Diagram

```
User ──► [1] API Request ──► Auth ──► [2] Authorized Request ──► Containment
                                                                      │
                                         ┌────────────────────────────┘
                                         ▼
                              [3] Safety Evaluation
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
        Nociception              Orchestration              LLM Provider
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         ▼
                              [4] Decision + Response
                                         │
                                         ▼
                                    Audit Log
```

---

## 3. STRIDE Analysis

### 3.1 Spoofing (Identity)

| ID | Threat | Component | Risk | Mitigation |
|----|--------|-----------|------|------------|
| S1 | Credential theft via phishing | Auth Service | HIGH | mTLS, hardware tokens |
| S2 | Session hijacking | API Gateway | HIGH | Secure cookies, token binding |
| S3 | Certificate impersonation | mTLS | MEDIUM | Certificate pinning, CRL |
| S4 | Operator identity spoofing | Containment | HIGH | Multi-factor verification |

### 3.2 Tampering (Integrity)

| ID | Threat | Component | Risk | Mitigation |
|----|--------|-----------|------|------------|
| T1 | Audit log modification | Audit Service | CRITICAL | Hash chain linking |
| T2 | Configuration tampering | Config DB | HIGH | Signed configs, RBAC |
| T3 | Request modification in transit | API Gateway | HIGH | TLS 1.3, request signing |
| T4 | Container image tampering | Deployment | HIGH | Image signing (Sigstore) |

### 3.3 Repudiation (Non-repudiation)

| ID | Threat | Component | Risk | Mitigation |
|----|--------|-----------|------|------------|
| R1 | Denying containment override | Operator Actions | HIGH | Cryptographic audit log |
| R2 | Denying configuration change | Admin Actions | MEDIUM | Change audit + approval workflow |
| R3 | Timestamp manipulation | Audit Service | MEDIUM | Monotonic + wall clock, NTP |

### 3.4 Information Disclosure

| ID | Threat | Component | Risk | Mitigation |
|----|--------|-----------|------|------------|
| I1 | API key exposure in logs | Logging | HIGH | Log sanitization, secret redaction |
| I2 | Memory dump credential leak | Runtime | MEDIUM | Memory zeroization, no swap |
| I3 | Side-channel timing attack | Auth Service | MEDIUM | Constant-time comparison |
| I4 | Metrics exposing sensitive data | Prometheus | LOW | Metric sanitization |

### 3.5 Denial of Service

| ID | Threat | Component | Risk | Mitigation |
|----|--------|-----------|------|------------|
| D1 | API request flooding | API Gateway | HIGH | Rate limiting, DDoS protection |
| D2 | Resource exhaustion | Containment | MEDIUM | Resource limits, thermoception |
| D3 | Consensus deadlock | Orchestration | MEDIUM | Timeout, fallback decision |
| D4 | Log storage exhaustion | Audit Service | LOW | Log rotation, external SIEM |

### 3.6 Elevation of Privilege

| ID | Threat | Component | Risk | Mitigation |
|----|--------|-----------|------|------------|
| E1 | RBAC bypass | Authorization | CRITICAL | Defense-in-depth, audit |
| E2 | Container escape | Runtime | CRITICAL | Non-root, read-only fs, seccomp |
| E3 | Trust domain escalation | Containment | HIGH | Strict boundary enforcement |
| E4 | Admin API access via Viewer | API Gateway | HIGH | Endpoint-level RBAC |

---

## 4. AI/ML Specific Threats (MITRE ATLAS)

### 4.1 ML Supply Chain Compromise

| ID | Threat | Technique | Risk | Mitigation |
|----|--------|-----------|------|------------|
| ML1 | Poisoned training data | AML.T0020 | N/A | FRACTAL doesn't train models |
| ML2 | Compromised LLM provider | AML.T0010 | MEDIUM | Multi-provider failover, output validation |
| ML3 | Backdoored model weights | AML.T0018 | N/A | FRACTAL doesn't ship models |

### 4.2 Adversarial Attacks

| ID | Threat | Technique | Risk | Mitigation |
|----|--------|-----------|------|------------|
| ML4 | Prompt injection | AML.T0051 | HIGH | Input sanitization, containment layer |
| ML5 | Jailbreak attempts | AML.T0054 | HIGH | Manipulation detection, consensus |
| ML6 | Model extraction | AML.T0024 | LOW | FRACTAL doesn't expose model weights |

### 4.3 Evasion Attacks

| ID | Threat | Technique | Risk | Mitigation |
|----|--------|-----------|------|------------|
| ML7 | Containment bypass via edge cases | AML.T0043 | HIGH | Continuous red-teaming, Axis-P testing |
| ML8 | Slow drift attack | AML.T0031 | MEDIUM | Vestibular drift detection |
| ML9 | Consensus manipulation | AML.T0040 | MEDIUM | Beta agent veto, adversarial design |

---

## 5. Attack Trees

### 5.1 Compromise FRACTAL System

```
Goal: Compromise FRACTAL System
├── [OR] Gain Unauthorized Access
│   ├── [AND] Steal Credentials
│   │   ├── Phishing attack on admin
│   │   ├── Credential stuffing
│   │   └── Exploit password reset
│   ├── [AND] Exploit Authentication Bug
│   │   ├── Find auth bypass
│   │   └── Session fixation
│   └── [AND] Compromise Certificate
│       ├── Steal private key
│       └── Forge certificate
├── [OR] Bypass Safety Controls
│   ├── [AND] Evade Containment
│   │   ├── Find edge case
│   │   ├── Gradual boundary push
│   │   └── Exploit operator trust
│   └── [AND] Manipulate Consensus
│       ├── Poison agent inputs
│       └── Deadlock consensus
└── [OR] Disrupt Operations
    ├── [AND] Denial of Service
    │   ├── Flood API
    │   └── Exhaust resources
    └── [AND] Data Corruption
        ├── Tamper with config
        └── Corrupt audit log
```

### 5.2 Evade AGI Safety Controls

```
Goal: Make AGI system perform unsafe action
├── [OR] Bypass Containment Layer
│   ├── [AND] Direct Containment Evasion
│   │   ├── Find intent classifier gap
│   │   ├── Encode harmful intent
│   │   └── Use out-of-distribution input
│   ├── [AND] Social Engineering
│   │   ├── Manipulate operator
│   │   ├── Impersonate authorized user
│   │   └── Exploit trust relationship
│   └── [AND] Technical Bypass
│       ├── Exploit API directly
│       └── Bypass via side channel
├── [OR] Compromise Orchestration
│   ├── [AND] Agent Manipulation
│   │   ├── Corrupt alpha agent
│   │   ├── Disable beta safety veto
│   │   └── Overwhelm gamma research
│   └── [AND] Consensus Attack
│       ├── Force timeout
│       └── Inject conflicting signals
└── [OR] Exploit Proprioceptive Sensors
    ├── [AND] Blind Nociception
    │   ├── Suppress error signals
    │   └── Desensitize thresholds
    └── [AND] Confuse Vestibular
        ├── Inject false baselines
        └── Manipulate time reference
```

---

## 6. Risk Matrix

### 6.1 Risk Rating Criteria

**Likelihood:**
- 5 (Almost Certain): >90% chance
- 4 (Likely): 50-90%
- 3 (Possible): 10-50%
- 2 (Unlikely): 1-10%
- 1 (Rare): <1%

**Impact:**
- 5 (Catastrophic): System compromise, safety failure
- 4 (Major): Significant data breach, extended outage
- 3 (Moderate): Limited breach, partial outage
- 2 (Minor): Minimal impact, quick recovery
- 1 (Negligible): No significant impact

### 6.2 Risk Ratings (Pre-Mitigation)

| Risk | Likelihood | Impact | Score | Rating |
|------|------------|--------|-------|--------|
| S1: Credential theft | 3 | 5 | 15 | HIGH |
| S4: Operator spoofing | 2 | 5 | 10 | HIGH |
| T1: Audit tampering | 2 | 5 | 10 | HIGH |
| E1: RBAC bypass | 2 | 5 | 10 | HIGH |
| E2: Container escape | 1 | 5 | 5 | MEDIUM |
| ML4: Prompt injection | 4 | 4 | 16 | HIGH |
| ML5: Jailbreak | 4 | 4 | 16 | HIGH |
| ML7: Containment bypass | 3 | 5 | 15 | HIGH |
| D1: API flooding | 4 | 3 | 12 | HIGH |

### 6.3 Risk Ratings (Post-Mitigation)

| Risk | Original | Mitigation | Residual | Status |
|------|----------|------------|----------|--------|
| S1 | HIGH | mTLS + hardware tokens | LOW | Implemented |
| S4 | HIGH | Multi-factor + audit | MEDIUM | Planned |
| T1 | HIGH | Hash chain linking | LOW | Implemented |
| E1 | HIGH | Defense-in-depth | LOW | Implemented |
| E2 | MEDIUM | Container hardening | LOW | Implemented |
| ML4 | HIGH | Input sanitization | MEDIUM | Implemented |
| ML5 | HIGH | Manipulation detection | MEDIUM | Implemented |
| ML7 | HIGH | Continuous red-team | MEDIUM | Ongoing |
| D1 | HIGH | Rate limiting | LOW | Implemented |

---

## 7. Recommendations

### 7.1 Immediate Actions (30 days)

1. **Enable mTLS** for all service-to-service communication
2. **Deploy network policies** to restrict lateral movement
3. **Configure SIEM integration** for real-time alerting
4. **Conduct penetration test** against deployed instance

### 7.2 Short-Term Actions (90 days)

1. **Implement hardware token support** for admin authentication
2. **Deploy Web Application Firewall** for API protection
3. **Establish red team program** for continuous adversarial testing
4. **Complete FedRAMP 3PAO engagement**

### 7.3 Long-Term Actions (180 days)

1. **Achieve FedRAMP Moderate** authorization
2. **Implement formal verification** for containment layer
3. **Deploy honeypots** for threat intelligence
4. **Establish bug bounty program**

---

## 8. Residual Risk Acceptance

The following residual risks require explicit acceptance:

### 8.1 Zero-Day Vulnerabilities

**Risk:** Unknown vulnerabilities in dependencies or Rust standard library
**Mitigation:** Memory safety, minimal dependencies, rapid patching
**Residual Impact:** MEDIUM
**Recommendation:** ACCEPT with continuous monitoring

### 8.2 Sophisticated Nation-State Actors

**Risk:** APT with resources to develop novel attack techniques
**Mitigation:** Defense-in-depth, assume breach architecture
**Residual Impact:** HIGH
**Recommendation:** ACCEPT with incident response plan

### 8.3 Insider Threat

**Risk:** Malicious actor with legitimate access
**Mitigation:** RBAC, audit logging, separation of duties
**Residual Impact:** MEDIUM
**Recommendation:** ACCEPT with monitoring and background checks

---

## 9. Review Schedule

| Review Type | Frequency | Responsible |
|-------------|-----------|-------------|
| Threat model update | Quarterly | Security Team |
| Risk reassessment | Monthly | Security Team |
| Penetration test | Annually | Third Party |
| Red team exercise | Quarterly | Internal + External |

---

## Appendix A: References

- STRIDE: Microsoft Threat Modeling
- MITRE ATLAS: https://atlas.mitre.org/
- NIST 800-30: Risk Management Guide
- OWASP Threat Modeling: https://owasp.org/www-community/Threat_Modeling

---

*This threat model should be treated as a living document and updated as the system evolves.*
