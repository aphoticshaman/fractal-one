//! ═══════════════════════════════════════════════════════════════════════════════
//! DOMAINS — Trust Domain Classification
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Every observation originates from a trust domain. This matters because:
//! - Self-reported data can be manipulated
//! - Operator context has different integrity guarantees than user input
//! - External APIs have their own failure modes
//!
//! Trust domains enable:
//! - Differential weighting in anomaly detection
//! - Source attribution in alerts
//! - Integrity verification strategies per domain
//! ═══════════════════════════════════════════════════════════════════════════════

use std::fmt;

/// Trust domains for observation sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustDomain {
    /// Internal system measurements (timers, counters, memory)
    /// Highest trust - directly observed, hard to spoof
    System,

    /// Operator-provided context (system prompts, configuration)
    /// High trust - vetted at deployment, stable
    Operator,

    /// User-provided input (queries, tool arguments)
    /// Medium trust - validated but adversarial
    User,

    /// External API responses (tool outputs, network data)
    /// Lower trust - can fail, be delayed, or be compromised
    External,

    /// Derived/computed values (aggregations, inferences)
    /// Trust inherited from inputs + computation integrity
    Derived,

    /// Ensemble/pod observations (other model outputs)
    /// Weak reference - useful for outlier detection, not authority
    Ensemble,
}

impl TrustDomain {
    /// Base trust weight (0.0 to 1.0)
    /// Higher = more trusted for anomaly baseline
    pub fn base_weight(&self) -> f64 {
        match self {
            TrustDomain::System => 1.0,
            TrustDomain::Operator => 0.9,
            TrustDomain::Derived => 0.7,
            TrustDomain::User => 0.5,
            TrustDomain::External => 0.4,
            TrustDomain::Ensemble => 0.2,
        }
    }

    /// Baseline adaptation rate multiplier
    /// Lower = slower adaptation (more resistant to poisoning)
    pub fn adaptation_rate(&self) -> f64 {
        match self {
            TrustDomain::System => 1.0,   // Quick adaptation, trusted
            TrustDomain::Operator => 0.8, // Slightly slower
            TrustDomain::Derived => 0.6,
            TrustDomain::User => 0.3, // Slow - potential adversarial
            TrustDomain::External => 0.3,
            TrustDomain::Ensemble => 0.2, // Slowest - weak reference
        }
    }

    /// Whether observations from this domain should trigger integrity checks
    pub fn requires_integrity_check(&self) -> bool {
        matches!(
            self,
            TrustDomain::User | TrustDomain::External | TrustDomain::Ensemble
        )
    }

    /// Whether this domain can be a primary source for baseline calibration
    pub fn can_calibrate(&self) -> bool {
        matches!(
            self,
            TrustDomain::System | TrustDomain::Operator | TrustDomain::Derived
        )
    }
}

impl fmt::Display for TrustDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrustDomain::System => write!(f, "system"),
            TrustDomain::Operator => write!(f, "operator"),
            TrustDomain::User => write!(f, "user"),
            TrustDomain::External => write!(f, "external"),
            TrustDomain::Derived => write!(f, "derived"),
            TrustDomain::Ensemble => write!(f, "ensemble"),
        }
    }
}

/// Domain-specific configuration for observation processing
#[derive(Debug, Clone)]
pub struct DomainConfig {
    pub domain: TrustDomain,
    /// Override base weight
    pub weight_override: Option<f64>,
    /// Override adaptation rate
    pub adaptation_override: Option<f64>,
    /// Custom validation function name (for logging)
    pub validator: Option<String>,
}

impl DomainConfig {
    pub fn new(domain: TrustDomain) -> Self {
        Self {
            domain,
            weight_override: None,
            adaptation_override: None,
            validator: None,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight_override = Some(weight.clamp(0.0, 1.0));
        self
    }

    pub fn with_adaptation(mut self, rate: f64) -> Self {
        self.adaptation_override = Some(rate.clamp(0.0, 1.0));
        self
    }

    pub fn effective_weight(&self) -> f64 {
        self.weight_override
            .unwrap_or_else(|| self.domain.base_weight())
    }

    pub fn effective_adaptation(&self) -> f64 {
        self.adaptation_override
            .unwrap_or_else(|| self.domain.adaptation_rate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_ordering() {
        assert!(TrustDomain::System.base_weight() > TrustDomain::User.base_weight());
        assert!(TrustDomain::Operator.base_weight() > TrustDomain::External.base_weight());
        assert!(TrustDomain::User.base_weight() > TrustDomain::Ensemble.base_weight());
    }

    #[test]
    fn test_integrity_requirements() {
        assert!(!TrustDomain::System.requires_integrity_check());
        assert!(TrustDomain::User.requires_integrity_check());
        assert!(TrustDomain::External.requires_integrity_check());
    }

    #[test]
    fn test_domain_config_override() {
        let config = DomainConfig::new(TrustDomain::User)
            .with_weight(0.8)
            .with_adaptation(0.5);

        assert_eq!(config.effective_weight(), 0.8);
        assert_eq!(config.effective_adaptation(), 0.5);
    }
}
