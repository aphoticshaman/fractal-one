//! Fisher Information Under Projection — The General Theorem
//!
//! Links information geometry to classification under observation.
//!
//! # The Theorem (Informal)
//!
//! For linear observations of data, classification accuracy is bounded by
//! retained Fisher information. Structure in the kernel of the projection
//! contributes exactly zero to downstream discriminability.
//!
//! # The Theorem (Formal)
//!
//! Let X ∈ ℝⁿ be data from class distributions P₀, P₁ (Gaussian, means μ₀, μ₁, shared Σ).
//! Let π: ℝⁿ → ℝᵈ be a linear projection (d < n).
//!
//! Define:
//!   - Fisher discriminability: F(P₀, P₁) = (μ₁ - μ₀)ᵀ Σ⁻¹ (μ₁ - μ₀)
//!   - Projected Fisher: F_π = (πμ₁ - πμ₀)ᵀ (πΣπᵀ)⁻¹ (πμ₁ - πμ₀)
//!
//! Then:
//!   (1) F_π ≤ F                     (information never increases)
//!   (2) F_π = F  iff  (μ₁ - μ₀) ∈ row(π)   (alignment condition)
//!   (3) F_π = 0  iff  (μ₁ - μ₀) ∈ ker(π)   (orthogonality kills)
//!
//! Corollary (Classification bound):
//!   Bayes error under π satisfies: ε_π ≥ Φ(-√(F_π)/2)
//!   where Φ is the standard normal CDF.
//!
//! # Connection to Projection Bound Theorem
//!
//! This is the classification-theoretic instantiation of:
//! "Structure survives projection iff aligned or adapts projection."
//!
//! The "structure" here is class discriminability (μ₁ - μ₀).
//! The "projection" is the observation operator π.
//! Alignment means: discriminating direction lies in observed subspace.

use serde::{Deserialize, Serialize};

/// The Fisher-Projection Theorem components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherProjectionTheorem {
    /// Ambient dimension
    pub n: usize,
    /// Projected dimension
    pub d: usize,
    /// Full Fisher discriminability F(P₀, P₁)
    pub fisher_full: f64,
    /// Projected Fisher discriminability F_π
    pub fisher_projected: f64,
    /// Alignment coefficient: cos²(θ) where θ = angle between (μ₁-μ₀) and row(π)
    pub alignment: f64,
}

impl FisherProjectionTheorem {
    /// Compute theorem quantities for Gaussian classes under projection.
    ///
    /// - `mu_diff`: μ₁ - μ₀ (class mean difference)
    /// - `sigma_inv`: Σ⁻¹ (inverse covariance, assume identity for simplicity)
    /// - `projection`: rows of π (d × n matrix, flattened row-major)
    pub fn compute(mu_diff: &[f64], projection: &[f64], d: usize) -> Self {
        let n = mu_diff.len();
        assert_eq!(projection.len(), d * n, "projection must be d×n");

        // Full Fisher: F = ||μ₁ - μ₀||² (assuming Σ = I)
        let fisher_full: f64 = mu_diff.iter().map(|x| x * x).sum();

        // Project the mean difference: π(μ₁ - μ₀)
        let mut projected_diff = vec![0.0; d];
        for i in 0..d {
            for j in 0..n {
                projected_diff[i] += projection[i * n + j] * mu_diff[j];
            }
        }

        // Projected Fisher: F_π = ||π(μ₁ - μ₀)||² (assuming πΣπᵀ = I after projection)
        let fisher_projected: f64 = projected_diff.iter().map(|x| x * x).sum();

        // Alignment: F_π / F = cos²(θ)
        let alignment = if fisher_full > 1e-10 {
            fisher_projected / fisher_full
        } else {
            0.0
        };

        Self {
            n,
            d,
            fisher_full,
            fisher_projected,
            alignment,
        }
    }

    /// Information retention ratio
    pub fn retention(&self) -> f64 {
        self.alignment
    }

    /// Bayes error lower bound under projection
    pub fn bayes_error_bound(&self) -> f64 {
        // ε ≥ Φ(-√F/2) where Φ is standard normal CDF
        // Approximate: Φ(-x) ≈ exp(-x²/2) / (x√(2π)) for large x
        let x = self.fisher_projected.sqrt() / 2.0;
        if x < 0.01 {
            0.5 // No discriminability → random guessing
        } else if x > 4.0 {
            0.0 // Perfect discrimination
        } else {
            // Simple approximation of Φ(-x)
            0.5 * (1.0 - erf(x / std::f64::consts::SQRT_2))
        }
    }

    /// Is discriminability fully preserved?
    pub fn is_aligned(&self) -> bool {
        self.alignment > 0.99
    }

    /// Is discriminability destroyed?
    pub fn is_orthogonal(&self) -> bool {
        self.alignment < 0.01
    }
}

/// Error function approximation (Abramowitz & Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Demonstrate the theorem with concrete examples
pub fn demonstrate() {
    println!("═══════════════════════════════════════════════════════════════");
    println!(" FISHER-PROJECTION THEOREM: Classification Under Observation");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Example 1: Aligned projection (preserves discriminability)
    println!("Example 1: ALIGNED — discriminating direction in row(π)\n");
    let mu_diff = [1.0, 0.0, 0.0]; // Classes differ in x-direction
    let projection = [
        1.0, 0.0, 0.0, // π projects onto x-y plane
        0.0, 1.0, 0.0,
    ];
    let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);
    print_result(&result);

    // Example 2: Orthogonal projection (destroys discriminability)
    println!("\nExample 2: ORTHOGONAL — discriminating direction in ker(π)\n");
    let mu_diff = [0.0, 0.0, 1.0]; // Classes differ in z-direction
    let projection = [
        1.0, 0.0, 0.0, // π projects onto x-y plane (z discarded)
        0.0, 1.0, 0.0,
    ];
    let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);
    print_result(&result);

    // Example 3: Partial alignment
    println!("\nExample 3: PARTIAL — discriminating direction at 45° to row(π)\n");
    let s = 1.0 / 2.0_f64.sqrt();
    let mu_diff = [s, 0.0, s]; // 45° between x and z
    let projection = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);
    print_result(&result);

    // Sweep: alignment angle vs retention
    println!("\n───────────────────────────────────────────────────────────────");
    println!(" Alignment Sweep: θ = angle between (μ₁-μ₀) and row(π)");
    println!("───────────────────────────────────────────────────────────────\n");

    println!("┌─────────┬───────────┬───────────┬────────────┐");
    println!("│ θ (deg) │ F_full    │ F_proj    │ Retention  │");
    println!("├─────────┼───────────┼───────────┼────────────┤");

    for angle_deg in (0..=90).step_by(15) {
        let theta = (angle_deg as f64).to_radians();
        let mu_diff = [theta.cos(), 0.0, theta.sin()];
        let projection = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);
        println!(
            "│  {:>3}    │  {:.4}    │  {:.4}    │  {:.2}%     │",
            angle_deg,
            result.fisher_full,
            result.fisher_projected,
            result.retention() * 100.0
        );
    }
    println!("└─────────┴───────────┴───────────┴────────────┘");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!(" F_proj = F_full × cos²(θ)  ←  The whole theorem in one line");
    println!("═══════════════════════════════════════════════════════════════");
}

fn print_result(r: &FisherProjectionTheorem) {
    println!("  Dimensions: {} → {}", r.n, r.d);
    println!("  Fisher (full):      {:.4}", r.fisher_full);
    println!("  Fisher (projected): {:.4}", r.fisher_projected);
    println!("  Alignment (cos²θ):  {:.4}", r.alignment);
    println!("  Retention:          {:.1}%", r.retention() * 100.0);
    println!("  Bayes error bound:  {:.4}", r.bayes_error_bound());
    if r.is_aligned() {
        println!("  Status: ✓ ALIGNED — full discriminability preserved");
    } else if r.is_orthogonal() {
        println!("  Status: ✗ ORTHOGONAL — discriminability destroyed");
    } else {
        println!(
            "  Status: ~ PARTIAL — {:.0}% information lost",
            (1.0 - r.retention()) * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_preserves() {
        let mu_diff = [1.0, 0.0, 0.0];
        let projection = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);

        assert!(result.is_aligned());
        assert!((result.retention() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_orthogonal_destroys() {
        let mu_diff = [0.0, 0.0, 1.0];
        let projection = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);

        assert!(result.is_orthogonal());
        assert!(result.retention() < 0.01);
    }

    #[test]
    fn test_partial_alignment() {
        let s = 1.0 / 2.0_f64.sqrt();
        let mu_diff = [s, 0.0, s]; // 45°
        let projection = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);

        // cos²(45°) = 0.5
        assert!((result.retention() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_retention_equals_cos_squared() {
        for angle_deg in (0..=90).step_by(10) {
            let theta = (angle_deg as f64).to_radians();
            let mu_diff = [theta.cos(), 0.0, theta.sin()];
            let projection = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);

            let expected = theta.cos().powi(2);
            assert!(
                (result.retention() - expected).abs() < 0.001,
                "θ={}°: got {:.4}, expected {:.4}",
                angle_deg,
                result.retention(),
                expected
            );
        }
    }

    #[test]
    fn test_information_never_increases() {
        // Random directions
        for i in 0..10 {
            let t = i as f64 * 0.3;
            let mu_diff = [t.cos(), t.sin(), (t * 2.0).cos()];
            let projection = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            let result = FisherProjectionTheorem::compute(&mu_diff, &projection, 2);

            assert!(
                result.fisher_projected <= result.fisher_full + 1e-10,
                "Information increased: {} > {}",
                result.fisher_projected,
                result.fisher_full
            );
        }
    }
}
