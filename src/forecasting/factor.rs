//! ═══════════════════════════════════════════════════════════════════════════════
//! FACTOR — The Variables That Matter
//! ═══════════════════════════════════════════════════════════════════════════════
//! A factor is a variable/consideration that influences the outcome.
//!
//! The key insight: salience detection is about factor weight differentials,
//! not outcome prediction accuracy.
//!
//! Example (Fed Rate Decision):
//! - Factor: "employment_data"
//! - Pod weight: 0.7 (employment mandate matters a lot)
//! - Market weight: 0.3 (market focused on inflation)
//! - Differential: +0.4 (pod sees this as more important)
//!
//! The prediction might be uncertain (55% cut), but the factor weighting
//! differential is the salience signal.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::salience::BetDirection;
use std::collections::HashMap;

/// A factor that influences the outcome
#[derive(Debug, Clone)]
pub struct Factor {
    /// Identifier for this factor
    pub name: String,
    /// Weight assigned to this factor (0.0 - 1.0)
    pub weight: f64,
    /// Direction this factor pushes (YES or NO)
    pub direction: BetDirection,
    /// Human-readable reasoning for this weight
    pub reasoning: String,
    /// Category of factor (e.g., "economic", "political", "technical")
    pub category: Option<String>,
}

impl Factor {
    pub fn new(name: &str, weight: f64, direction: BetDirection, reasoning: &str) -> Self {
        Self {
            name: name.to_string(),
            weight: weight.clamp(0.0, 1.0),
            direction,
            reasoning: reasoning.to_string(),
            category: None,
        }
    }

    pub fn with_category(mut self, category: &str) -> Self {
        self.category = Some(category.to_string());
        self
    }
}

/// Factor weight with source attribution
#[derive(Debug, Clone)]
pub struct FactorWeight {
    /// The factor
    pub factor: Factor,
    /// Source of this weighting (pod role, market source, etc.)
    pub source: String,
}

/// Differential between pod and market factor weights
#[derive(Debug, Clone)]
pub struct FactorDifferential {
    /// The factor being compared
    pub factor: Factor,
    /// Pod's weight for this factor
    pub pod_weight: f64,
    /// Market's implied weight for this factor
    pub market_weight: f64,
}

impl FactorDifferential {
    /// The key metric: how different is pod's weighting from market's?
    pub fn differential(&self) -> f64 {
        self.pod_weight - self.market_weight
    }

    /// Absolute magnitude of differential
    pub fn magnitude(&self) -> f64 {
        self.differential().abs()
    }

    /// Is this factor salient (differential exceeds threshold)?
    pub fn is_salient(&self, threshold: f64) -> bool {
        self.magnitude() >= threshold
    }

    /// Compute all differentials between pod and market factors
    pub fn compute_all(pod: &PodFactors, market: &MarketFactors) -> Vec<Self> {
        let mut differentials = Vec::new();

        // Build market factor lookup
        let market_lookup: HashMap<&str, &Factor> = market.factors.iter()
            .map(|f| (f.name.as_str(), f))
            .collect();

        // Compare each pod factor to market equivalent
        for pod_factor in &pod.factors {
            let market_weight = market_lookup
                .get(pod_factor.name.as_str())
                .map(|f| f.weight)
                .unwrap_or(0.0); // If market doesn't track this factor, weight = 0

            differentials.push(Self {
                factor: pod_factor.clone(),
                pod_weight: pod_factor.weight,
                market_weight,
            });
        }

        // Also check for factors market tracks that pod doesn't
        let pod_lookup: HashMap<&str, &Factor> = pod.factors.iter()
            .map(|f| (f.name.as_str(), f))
            .collect();

        for market_factor in &market.factors {
            if !pod_lookup.contains_key(market_factor.name.as_str()) {
                differentials.push(Self {
                    factor: market_factor.clone(),
                    pod_weight: 0.0, // Pod doesn't track this
                    market_weight: market_factor.weight,
                });
            }
        }

        differentials
    }
}

/// Collection of factors from pod analysis
#[derive(Debug, Clone)]
pub struct PodFactors {
    /// All factors identified by pod
    pub factors: Vec<Factor>,
}

impl PodFactors {
    pub fn new() -> Self {
        Self { factors: Vec::new() }
    }

    pub fn add(&mut self, factor: Factor) {
        self.factors.push(factor);
    }

    /// Total weight (should sum to ~1.0 if properly normalized)
    pub fn total_weight(&self) -> f64 {
        self.factors.iter().map(|f| f.weight).sum()
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.total_weight();
        if total > 0.0 {
            for factor in &mut self.factors {
                factor.weight /= total;
            }
        }
    }

    /// Get dominant factor (highest weight)
    pub fn dominant(&self) -> Option<&Factor> {
        self.factors.iter()
            .max_by(|a, b| {
                a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Filter to factors pushing in a specific direction
    pub fn by_direction(&self, direction: BetDirection) -> Vec<&Factor> {
        self.factors.iter()
            .filter(|f| f.direction == direction)
            .collect()
    }
}

impl Default for PodFactors {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of factors implied by market consensus
#[derive(Debug, Clone)]
pub struct MarketFactors {
    /// Factors inferred from market price and commentary
    pub factors: Vec<Factor>,
}

impl MarketFactors {
    pub fn new() -> Self {
        Self { factors: Vec::new() }
    }

    pub fn add(&mut self, factor: Factor) {
        self.factors.push(factor);
    }

    /// Create from market commentary analysis
    pub fn from_commentary(commentary: &[MarketCommentary]) -> Self {
        let mut factors = Vec::new();

        // Aggregate factor mentions across commentary
        let mut weight_sums: HashMap<String, (f64, usize, BetDirection)> = HashMap::new();

        for comment in commentary {
            for factor in &comment.mentioned_factors {
                let entry = weight_sums
                    .entry(factor.name.clone())
                    .or_insert((0.0, 0, factor.direction));
                entry.0 += factor.weight;
                entry.1 += 1;
            }
        }

        // Average weights
        for (name, (sum, count, direction)) in weight_sums {
            let avg_weight = sum / count as f64;
            factors.push(Factor::new(&name, avg_weight, direction, "Inferred from market commentary"));
        }

        Self { factors }
    }
}

impl Default for MarketFactors {
    fn default() -> Self {
        Self::new()
    }
}

/// Market commentary source for factor inference
#[derive(Debug, Clone)]
pub struct MarketCommentary {
    /// Source identifier
    pub source: String,
    /// Factors mentioned in commentary
    pub mentioned_factors: Vec<Factor>,
    /// Raw text (for reference)
    pub text: String,
}

/// Factor comparison result
#[derive(Debug, Clone)]
pub struct FactorComparison {
    /// All factor differentials
    pub differentials: Vec<FactorDifferential>,
    /// Salient factors (above threshold)
    pub salient: Vec<FactorDifferential>,
    /// Factors only pod tracks
    pub pod_unique: Vec<Factor>,
    /// Factors only market tracks
    pub market_unique: Vec<Factor>,
    /// Agreement factors (similar weights)
    pub agreed: Vec<FactorDifferential>,
}

impl FactorComparison {
    pub fn compute(pod: &PodFactors, market: &MarketFactors, threshold: f64) -> Self {
        let differentials = FactorDifferential::compute_all(pod, market);

        let salient: Vec<_> = differentials.iter()
            .filter(|d| d.is_salient(threshold))
            .cloned()
            .collect();

        let agreed: Vec<_> = differentials.iter()
            .filter(|d| !d.is_salient(threshold))
            .cloned()
            .collect();

        let pod_unique: Vec<_> = differentials.iter()
            .filter(|d| d.market_weight == 0.0 && d.pod_weight > 0.0)
            .map(|d| d.factor.clone())
            .collect();

        let market_unique: Vec<_> = differentials.iter()
            .filter(|d| d.pod_weight == 0.0 && d.market_weight > 0.0)
            .map(|d| d.factor.clone())
            .collect();

        Self {
            differentials,
            salient,
            pod_unique,
            market_unique,
            agreed,
        }
    }

    /// Total salience (sum of absolute differentials for salient factors)
    pub fn total_salience(&self) -> f64 {
        self.salient.iter().map(|d| d.magnitude()).sum()
    }

    /// Dominant differential (highest magnitude)
    pub fn dominant(&self) -> Option<&FactorDifferential> {
        self.differentials.iter()
            .max_by(|a, b| {
                a.magnitude().partial_cmp(&b.magnitude()).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Factor extractor — extracts factors from pod analysis output
pub struct FactorExtractor {
    /// Known factor categories
    #[allow(dead_code)] // Reserved for future category filtering
    categories: Vec<String>,
    /// Factor weight normalization mode
    normalize: bool,
}

impl FactorExtractor {
    pub fn new() -> Self {
        Self {
            categories: vec![
                "economic".to_string(),
                "political".to_string(),
                "technical".to_string(),
                "regulatory".to_string(),
                "geopolitical".to_string(),
                "behavioral".to_string(),
            ],
            normalize: true,
        }
    }

    /// Extract factors from structured pod output
    pub fn extract_from_pod(
        &self,
        alpha_factors: &[(String, f64, BetDirection, String)],
        beta_factors: &[(String, f64, BetDirection, String)],
        gamma_critique: &[(String, f64)], // Factor name + skepticism weight
        delta_synthesis: &[(String, f64, BetDirection, String)],
    ) -> PodFactors {
        let mut pod_factors = PodFactors::new();

        // Start with Delta synthesis (most authoritative)
        for (name, weight, direction, reasoning) in delta_synthesis {
            pod_factors.add(Factor::new(name, *weight, *direction, reasoning));
        }

        // If no delta synthesis, fall back to alpha/beta average
        if delta_synthesis.is_empty() {
            let mut combined: HashMap<String, (f64, usize, BetDirection, String)> = HashMap::new();

            for (name, weight, direction, reasoning) in alpha_factors {
                let entry = combined
                    .entry(name.clone())
                    .or_insert((0.0, 0, *direction, reasoning.clone()));
                entry.0 += weight;
                entry.1 += 1;
            }

            for (name, weight, direction, reasoning) in beta_factors {
                let entry = combined
                    .entry(name.clone())
                    .or_insert((0.0, 0, *direction, reasoning.clone()));
                entry.0 += weight;
                entry.1 += 1;
            }

            // Apply gamma skepticism discounts
            for (name, skepticism) in gamma_critique {
                if let Some(entry) = combined.get_mut(name) {
                    entry.0 *= 1.0 - skepticism; // Reduce weight by skepticism factor
                }
            }

            for (name, (sum, count, direction, reasoning)) in combined {
                let avg = sum / count as f64;
                pod_factors.add(Factor::new(&name, avg, direction, &reasoning));
            }
        }

        if self.normalize {
            pod_factors.normalize();
        }

        pod_factors
    }
}

impl Default for FactorExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_differential() {
        let pod = PodFactors {
            factors: vec![
                Factor::new("employment", 0.7, BetDirection::Yes, "Employment mandate"),
                Factor::new("inflation", 0.3, BetDirection::No, "Sticky inflation"),
            ],
        };

        let market = MarketFactors {
            factors: vec![
                Factor::new("employment", 0.3, BetDirection::Yes, ""),
                Factor::new("inflation", 0.5, BetDirection::No, ""),
                Factor::new("global_rates", 0.2, BetDirection::No, ""), // Market tracks, pod doesn't
            ],
        };

        let differentials = FactorDifferential::compute_all(&pod, &market);

        // Should have 3 differentials
        assert_eq!(differentials.len(), 3);

        // Employment: 0.7 - 0.3 = 0.4
        let employment = differentials.iter().find(|d| d.factor.name == "employment").unwrap();
        assert!((employment.differential() - 0.4).abs() < 0.001);

        // Inflation: 0.3 - 0.5 = -0.2
        let inflation = differentials.iter().find(|d| d.factor.name == "inflation").unwrap();
        assert!((inflation.differential() - (-0.2)).abs() < 0.001);

        // Global rates: 0.0 - 0.2 = -0.2 (pod doesn't track)
        let global = differentials.iter().find(|d| d.factor.name == "global_rates").unwrap();
        assert!((global.differential() - (-0.2)).abs() < 0.001);
    }

    #[test]
    fn test_factor_comparison() {
        let pod = PodFactors {
            factors: vec![
                Factor::new("key_factor", 0.8, BetDirection::Yes, "Very important"),
                Factor::new("minor_factor", 0.2, BetDirection::No, "Less important"),
            ],
        };

        let market = MarketFactors {
            factors: vec![
                Factor::new("key_factor", 0.3, BetDirection::Yes, ""),
                Factor::new("minor_factor", 0.25, BetDirection::No, ""),
            ],
        };

        let comparison = FactorComparison::compute(&pod, &market, 0.15);

        // key_factor should be salient (0.5 differential)
        assert!(!comparison.salient.is_empty());
        assert!(comparison.salient.iter().any(|d| d.factor.name == "key_factor"));

        // minor_factor should not be salient (0.05 differential)
        assert!(comparison.salient.iter().all(|d| d.factor.name != "minor_factor"));
    }
}
