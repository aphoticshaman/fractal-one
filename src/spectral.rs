//! ═══════════════════════════════════════════════════════════════════════════════
//! SPECTRAL SWEEP — Boundary Operator Fidelity Testing
//! ═══════════════════════════════════════════════════════════════════════════════
//! Tests whether a "Platonic residual gap" exists between abstract and physical
//! simulations by fitting fidelity_loss(E) = a * E^(-alpha) + c
//!
//! If c > 0 with CI excluding zero: irreducible gap exists
//! If c ≈ 0: gap approaches zero as resources increase
//! ═══════════════════════════════════════════════════════════════════════════════

use rustfft::{num_complex::Complex, FftPlanner};

/// Resource level configuration
#[derive(Debug, Clone)]
pub struct ResourceLevel {
    pub name: String,
    pub energy_budget: usize,
    pub meas_noise: f64,
    pub downsample: usize,
}

impl ResourceLevel {
    /// Compute effective resource index (higher = better resources)
    pub fn e_eff(&self, gamma: f64) -> f64 {
        let ds_sq = (self.downsample * self.downsample) as f64;
        (self.energy_budget as f64 / ds_sq) / (1.0 + gamma * self.meas_noise)
    }
}

/// Default 4-level sweep configuration
pub fn default_levels() -> Vec<ResourceLevel> {
    vec![
        ResourceLevel {
            name: "L1_low".into(),
            energy_budget: 512,
            meas_noise: 0.030,
            downsample: 4,
        },
        ResourceLevel {
            name: "L2_mid".into(),
            energy_budget: 2048,
            meas_noise: 0.020,
            downsample: 2,
        },
        ResourceLevel {
            name: "L3_high".into(),
            energy_budget: 8192,
            meas_noise: 0.010,
            downsample: 2,
        },
        ResourceLevel {
            name: "L4_max".into(),
            energy_budget: 32768,
            meas_noise: 0.002,
            downsample: 1,
        },
    ]
}

/// Extended levels including "perfect" regime (noise=0, ds=1, max energy)
pub fn extended_levels() -> Vec<ResourceLevel> {
    vec![
        ResourceLevel {
            name: "L1_low".into(),
            energy_budget: 512,
            meas_noise: 0.030,
            downsample: 4,
        },
        ResourceLevel {
            name: "L2_mid".into(),
            energy_budget: 2048,
            meas_noise: 0.020,
            downsample: 2,
        },
        ResourceLevel {
            name: "L3_high".into(),
            energy_budget: 8192,
            meas_noise: 0.010,
            downsample: 2,
        },
        ResourceLevel {
            name: "L4_max".into(),
            energy_budget: 32768,
            meas_noise: 0.002,
            downsample: 1,
        },
        ResourceLevel {
            name: "L5_perfect".into(),
            energy_budget: 131072,
            meas_noise: 0.000,
            downsample: 1,
        },
    ]
}

/// Trial result
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub level: String,
    pub e_eff: f64,
    pub tv: f64,
    pub kl: f64,
}

/// Fit result for asymptote model
#[derive(Debug, Clone)]
pub struct FitResult {
    pub a: f64,
    pub alpha: f64,
    pub c: f64,
    pub a_se: f64,
    pub alpha_se: f64,
    pub c_se: f64,
    pub c_ci_excludes_zero: bool,
}

/// Simple xorshift64 PRNG
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        self.0
    }

    fn next_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

/// Majority-rule cellular automaton update (9-cell Moore neighborhood)
fn majority_update(grid: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0i32;
            for di in [-1i32, 0, 1] {
                for dj in [-1i32, 0, 1] {
                    let ni = ((i as i32 + di).rem_euclid(n as i32)) as usize;
                    let nj = ((j as i32 + dj).rem_euclid(n as i32)) as usize;
                    sum += grid[ni * n + nj] as i32;
                }
            }
            out[i * n + j] = if sum >= 5 { 1 } else { 0 };
        }
    }
    out
}

/// Abstract simulation: checkerboard + noise + majority updates
fn abstract_sim(seed: u64, n: usize, t: usize, init_flip: f64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut grid = vec![0u8; n * n];

    // Initialize checkerboard with random flips
    for i in 0..n {
        for j in 0..n {
            let base = ((i + j) % 2) as u8;
            let flip = if rng.next_f64() < init_flip { 1 } else { 0 };
            grid[i * n + j] = base ^ flip;
        }
    }

    // Run majority updates
    for _ in 0..t {
        grid = majority_update(&grid, n);
    }

    grid
}

/// Downsample grid by averaging blocks
fn downsample_grid(grid: &[u8], n: usize, factor: usize) -> Vec<u8> {
    if factor == 1 {
        return grid.to_vec();
    }
    let m = n / factor;
    let mut out = vec![0u8; m * m];
    let threshold = (factor * factor / 2) as i32;

    for i in 0..m {
        for j in 0..m {
            let mut sum = 0i32;
            for di in 0..factor {
                for dj in 0..factor {
                    sum += grid[(i * factor + di) * n + (j * factor + dj)] as i32;
                }
            }
            out[i * m + j] = if sum > threshold { 1 } else { 0 };
        }
    }
    out
}

/// Upsample grid by nearest-neighbor
fn upsample_nn(grid: &[u8], m: usize, factor: usize) -> Vec<u8> {
    if factor == 1 {
        return grid.to_vec();
    }
    let n = m * factor;
    let mut out = vec![0u8; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = grid[(i / factor) * m + (j / factor)];
        }
    }
    out
}

/// Physical simulation with resource constraints
fn physical_sim(
    seed: u64,
    target: &[u8],
    n: usize,
    energy_budget: usize,
    meas_noise: f64,
    downsample: usize,
    t: usize,
    pull_prob: f64,
) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let m = n / downsample;

    // Downsample target
    let tgt = downsample_grid(target, n, downsample);

    // Initialize checkerboard at lower resolution
    let mut grid = vec![0u8; m * m];
    for i in 0..m {
        for j in 0..m {
            grid[i * m + j] = ((i + j) % 2) as u8;
        }
    }

    // Energy-limited updates per timestep
    let k = (energy_budget / t).max(1).min(m * m);

    for _ in 0..t {
        // Asynchronous energy-limited updates
        for _ in 0..k {
            let i = rng.next_usize(m);
            let j = rng.next_usize(m);

            // Compute majority in 3x3 neighborhood
            let mut sum = 0i32;
            for di in [-1i32, 0, 1] {
                for dj in [-1i32, 0, 1] {
                    let ni = ((i as i32 + di).rem_euclid(m as i32)) as usize;
                    let nj = ((j as i32 + dj).rem_euclid(m as i32)) as usize;
                    sum += grid[ni * m + nj] as i32;
                }
            }
            let maj = if sum >= 5 { 1 } else { 0 };

            // Pull toward target with some probability (oracle guidance)
            if rng.next_f64() < pull_prob {
                grid[i * m + j] = tgt[i * m + j];
            } else {
                grid[i * m + j] = maj;
            }
        }

        // Apply measurement noise
        if meas_noise > 0.0 {
            for idx in 0..(m * m) {
                if rng.next_f64() < meas_noise {
                    grid[idx] ^= 1;
                }
            }
        }
    }

    // Upsample back to original resolution
    upsample_nn(&grid, m, downsample)
}

/// Boundary operator: gradient magnitude at each cell
fn boundary_operator(grid: &[u8], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let curr = grid[i * n + j] as f64;
            let right = grid[i * n + ((j + 1) % n)] as f64;
            let down = grid[((i + 1) % n) * n + j] as f64;
            out[i * n + j] = (curr - right).abs() + (curr - down).abs();
        }
    }
    out
}

/// 2D FFT magnitude using rustfft (O(n² log n) instead of O(n⁴))
fn fft2_magnitude(field: &[f64], n: usize) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    // Convert to complex and copy into working buffer
    let mut data: Vec<Complex<f64>> = field.iter().map(|&v| Complex::new(v, 0.0)).collect();

    // Row-wise FFT
    for row in 0..n {
        let start = row * n;
        let mut row_data: Vec<Complex<f64>> = data[start..start + n].to_vec();
        fft.process(&mut row_data);
        data[start..start + n].copy_from_slice(&row_data);
    }

    // Column-wise FFT
    let mut col_buffer = vec![Complex::new(0.0, 0.0); n];
    for col in 0..n {
        // Extract column
        for row in 0..n {
            col_buffer[row] = data[row * n + col];
        }
        fft.process(&mut col_buffer);
        // Write back
        for row in 0..n {
            data[row * n + col] = col_buffer[row];
        }
    }

    // Compute power spectrum
    let mag: Vec<f64> = data.iter().map(|c| c.norm_sqr()).collect();

    // FFT shift (move DC to center)
    let mut shifted = vec![0.0f64; n * n];
    let half = n / 2;
    for y in 0..n {
        for x in 0..n {
            let sy = (y + half) % n;
            let sx = (x + half) % n;
            shifted[sy * n + sx] = mag[y * n + x];
        }
    }

    shifted
}

/// Radial-averaged power spectrum
fn radial_profile(power: &[f64], n: usize, n_bins: usize) -> Vec<f64> {
    let mut prof = vec![0.0f64; n_bins];
    let mut cnt = vec![0.0f64; n_bins];

    let cy = (n - 1) as f64 / 2.0;
    let cx = cy;
    let max_r = (cy * cy + cx * cx).sqrt();

    for y in 0..n {
        for x in 0..n {
            let r = ((y as f64 - cy).powi(2) + (x as f64 - cx).powi(2)).sqrt();
            let r_norm = r / max_r;
            let bin = ((r_norm * n_bins as f64) as usize).min(n_bins - 1);
            prof[bin] += power[y * n + x];
            cnt[bin] += 1.0;
        }
    }

    // Average and normalize
    for i in 0..n_bins {
        if cnt[i] > 0.0 {
            prof[i] /= cnt[i];
        }
    }

    let sum: f64 = prof.iter().sum();
    if sum > 1e-12 {
        for p in &mut prof {
            *p /= sum;
        }
    }

    prof
}

/// Total Variation distance between distributions
fn tv_distance(p: &[f64], q: &[f64]) -> f64 {
    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();

    let mut dist = 0.0f64;
    for i in 0..p.len().min(q.len()) {
        let pi = if p_sum > 0.0 { p[i] / p_sum } else { 0.0 };
        let qi = if q_sum > 0.0 { q[i] / q_sum } else { 0.0 };
        dist += (pi - qi).abs();
    }
    0.5 * dist
}

/// KL divergence (p || q)
fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    let eps = 1e-12;
    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();

    let mut kl = 0.0f64;
    for i in 0..p.len().min(q.len()) {
        let pi = (p[i] / p_sum.max(eps)).max(eps);
        let qi = (q[i] / q_sum.max(eps)).max(eps);
        kl += pi * (pi / qi).ln();
    }
    kl
}

/// Asymptote model: f(E) = a * E^(-alpha) + c
fn asymptote(e: f64, a: f64, alpha: f64, c: f64) -> f64 {
    a * e.powf(-alpha) + c
}

/// Fit asymptote model using Levenberg-Marquardt
/// Returns (a, alpha, c) and standard errors
fn fit_asymptote(e_vals: &[f64], y_vals: &[f64]) -> FitResult {
    let n = e_vals.len();
    if n < 4 {
        return FitResult {
            a: 0.0,
            alpha: 0.0,
            c: 0.0,
            a_se: f64::INFINITY,
            alpha_se: f64::INFINITY,
            c_se: f64::INFINITY,
            c_ci_excludes_zero: false,
        };
    }

    // Initial guess
    let mut a = 0.2;
    let mut alpha = 0.5;
    let mut c = 0.02;

    let lambda_init = 0.001;
    let mut lambda = lambda_init;

    // Levenberg-Marquardt iterations
    for _iter in 0..200 {
        // Compute residuals and Jacobian
        let mut residuals = vec![0.0; n];
        let mut j_a = vec![0.0; n];
        let mut j_alpha = vec![0.0; n];
        let mut j_c = vec![0.0; n];

        for i in 0..n {
            let e = e_vals[i];
            let y = y_vals[i];
            let pred = asymptote(e, a, alpha, c);
            residuals[i] = y - pred;

            // Partial derivatives
            j_a[i] = -e.powf(-alpha);
            j_alpha[i] = a * e.powf(-alpha) * e.ln();
            j_c[i] = -1.0;
        }

        // Compute J^T J and J^T r
        let mut jtj = [[0.0; 3]; 3];
        let mut jtr = [0.0; 3];

        for i in 0..n {
            let j = [j_a[i], j_alpha[i], j_c[i]];
            for row in 0..3 {
                jtr[row] += j[row] * residuals[i];
                for col in 0..3 {
                    jtj[row][col] += j[row] * j[col];
                }
            }
        }

        // Add damping to diagonal
        for k in 0..3 {
            jtj[k][k] *= 1.0 + lambda;
        }

        // Solve 3x3 system (Cramer's rule)
        let det = jtj[0][0] * (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1])
            - jtj[0][1] * (jtj[1][0] * jtj[2][2] - jtj[1][2] * jtj[2][0])
            + jtj[0][2] * (jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]);

        if det.abs() < 1e-20 {
            break;
        }

        // Replace column k with jtr and compute determinant
        let det_a = jtr[0] * (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1])
            - jtj[0][1] * (jtr[1] * jtj[2][2] - jtj[1][2] * jtr[2])
            + jtj[0][2] * (jtr[1] * jtj[2][1] - jtj[1][1] * jtr[2]);

        let det_alpha = jtj[0][0] * (jtr[1] * jtj[2][2] - jtj[1][2] * jtr[2])
            - jtr[0] * (jtj[1][0] * jtj[2][2] - jtj[1][2] * jtj[2][0])
            + jtj[0][2] * (jtj[1][0] * jtr[2] - jtr[1] * jtj[2][0]);

        let det_c = jtj[0][0] * (jtj[1][1] * jtr[2] - jtr[1] * jtj[2][1])
            - jtj[0][1] * (jtj[1][0] * jtr[2] - jtr[1] * jtj[2][0])
            + jtr[0] * (jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]);

        let da = det_a / det;
        let dalpha = det_alpha / det;
        let dc = det_c / det;

        // Try update
        let a_new = (a + da).max(0.0);
        let alpha_new = (alpha + dalpha).clamp(0.0, 5.0);
        let c_new = (c + dc).clamp(0.0, 1.0);

        // Compute new SSE
        let sse_old: f64 = residuals.iter().map(|r| r * r).sum();
        let sse_new: f64 = e_vals
            .iter()
            .zip(y_vals.iter())
            .map(|(e, y)| (y - asymptote(*e, a_new, alpha_new, c_new)).powi(2))
            .sum();

        if sse_new < sse_old {
            a = a_new;
            alpha = alpha_new;
            c = c_new;
            lambda *= 0.5;
        } else {
            lambda *= 2.0;
        }

        if lambda > 1e10 || (da.abs() < 1e-10 && dalpha.abs() < 1e-10 && dc.abs() < 1e-10) {
            break;
        }
    }

    // Compute standard errors from Hessian approximation
    let mut jtj = [[0.0; 3]; 3];
    let mut sse = 0.0;

    for i in 0..n {
        let e = e_vals[i];
        let y = y_vals[i];
        let pred = asymptote(e, a, alpha, c);
        sse += (y - pred).powi(2);

        let j = [-e.powf(-alpha), a * e.powf(-alpha) * e.ln(), -1.0];

        for row in 0..3 {
            for col in 0..3 {
                jtj[row][col] += j[row] * j[col];
            }
        }
    }

    // Invert 3x3 matrix for covariance
    let det = jtj[0][0] * (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1])
        - jtj[0][1] * (jtj[1][0] * jtj[2][2] - jtj[1][2] * jtj[2][0])
        + jtj[0][2] * (jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]);

    let mse = sse / (n as f64 - 3.0).max(1.0);

    let (a_se, alpha_se, c_se) = if det.abs() > 1e-20 {
        // Cofactor matrix elements for diagonal
        let c00 = jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1];
        let c11 = jtj[0][0] * jtj[2][2] - jtj[0][2] * jtj[2][0];
        let c22 = jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0];

        (
            (mse * c00 / det).abs().sqrt(),
            (mse * c11 / det).abs().sqrt(),
            (mse * c22 / det).abs().sqrt(),
        )
    } else {
        (f64::INFINITY, f64::INFINITY, f64::INFINITY)
    };

    // 95% CI excludes zero if c - 1.96*se > 0
    let c_ci_excludes_zero = c - 1.96 * c_se > 0.0;

    FitResult {
        a,
        alpha,
        c,
        a_se,
        alpha_se,
        c_se,
        c_ci_excludes_zero,
    }
}

/// Run coupling strength sweep at fixed (max) resources
pub fn run_coupling_sweep(
    n_trials: usize,
    grid_size: usize,
    timesteps: usize,
    n_bins: usize,
) -> Vec<(f64, f64, f64)> {
    // Sweep pull_prob from 0.0 to 1.0
    let pull_probs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.0];

    // Fixed "perfect" resources
    let energy_budget = 131072;
    let meas_noise = 0.0;
    let downsample = 1;
    let init_flip = 0.03;

    let base_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut results = Vec::new();

    for &pull_prob in &pull_probs {
        let mut tv_vals = Vec::with_capacity(n_trials);
        let mut kl_vals = Vec::with_capacity(n_trials);

        for t in 0..n_trials {
            let seed = base_seed.wrapping_add((t as u64) * 100000);

            let abs_grid = abstract_sim(seed, grid_size, timesteps, init_flip);
            let phys_grid = physical_sim(
                seed.wrapping_add(999),
                &abs_grid,
                grid_size,
                energy_budget,
                meas_noise,
                downsample,
                timesteps,
                pull_prob,
            );

            let abs_boundary = boundary_operator(&abs_grid, grid_size);
            let phys_boundary = boundary_operator(&phys_grid, grid_size);

            let abs_power = fft2_magnitude(&abs_boundary, grid_size);
            let phys_power = fft2_magnitude(&phys_boundary, grid_size);

            let abs_prof = radial_profile(&abs_power, grid_size, n_bins);
            let phys_prof = radial_profile(&phys_power, grid_size, n_bins);

            tv_vals.push(tv_distance(&abs_prof, &phys_prof));
            kl_vals.push(kl_divergence(&abs_prof, &phys_prof));
        }

        let tv_mean: f64 = tv_vals.iter().sum::<f64>() / n_trials as f64;
        let kl_mean: f64 = kl_vals.iter().sum::<f64>() / n_trials as f64;

        results.push((pull_prob, tv_mean, kl_mean));
        println!(
            "  pull_prob={:.2}: TV={:.6}, KL={:.6}",
            pull_prob, tv_mean, kl_mean
        );
    }

    results
}

/// Print coupling sweep results
pub fn print_coupling_results(results: &[(f64, f64, f64)]) {
    println!();
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m COUPLING SWEEP RESULTS\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();
    println!("{:>10} {:>12} {:>12}", "pull_prob", "TV", "KL");
    println!("{}", "-".repeat(36));

    for (pp, tv, kl) in results {
        println!("{:>10.2} {:>12.6} {:>12.6}", pp, tv, kl);
    }

    // Check if TV → 0 as pull_prob → 1
    let tv_at_0 = results.first().map(|(_, tv, _)| *tv).unwrap_or(0.0);
    let tv_at_1 = results.last().map(|(_, tv, _)| *tv).unwrap_or(0.0);

    println!();
    println!("TV at pull_prob=0.0: {:.6}", tv_at_0);
    println!("TV at pull_prob=1.0: {:.6}", tv_at_1);
    println!();

    if tv_at_1 < 0.001 {
        println!("\x1b[32m✓ GAP COLLAPSES at full coupling\x1b[0m");
        println!("  Residual was due to INSUFFICIENT COUPLING, not structural mismatch.");
    } else if tv_at_1 < tv_at_0 * 0.1 {
        println!("\x1b[33m~ GAP REDUCES 10x but persists\x1b[0m");
        println!("  Residual is MOSTLY coupling-limited, but small structural component remains.");
    } else {
        println!("\x1b[31m✗ GAP PERSISTS even at full coupling\x1b[0m");
        println!("  Residual is STRUCTURAL — dynamics diverge even with perfect oracle.");
    }
    println!();
}

/// Run the full spectral sweep
pub fn run_sweep(
    levels: &[ResourceLevel],
    n_trials: usize,
    grid_size: usize,
    timesteps: usize,
    n_bins: usize,
) -> (Vec<TrialResult>, FitResult, FitResult) {
    let gamma = 50.0;
    let init_flip = 0.03;
    let pull_prob = 0.10;

    let mut results = Vec::with_capacity(levels.len() * n_trials);
    let base_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    for level in levels {
        let e_eff = level.e_eff(gamma);
        println!("  {} (E_eff={:.1})", level.name, e_eff);

        for t in 0..n_trials {
            let seed = base_seed.wrapping_add((t as u64) * 100000);

            // Abstract simulation
            let abs_grid = abstract_sim(seed, grid_size, timesteps, init_flip);

            // Physical simulation
            let phys_grid = physical_sim(
                seed.wrapping_add(999),
                &abs_grid,
                grid_size,
                level.energy_budget,
                level.meas_noise,
                level.downsample,
                timesteps,
                pull_prob,
            );

            // Boundary operators
            let abs_boundary = boundary_operator(&abs_grid, grid_size);
            let phys_boundary = boundary_operator(&phys_grid, grid_size);

            // Power spectra
            let abs_power = fft2_magnitude(&abs_boundary, grid_size);
            let phys_power = fft2_magnitude(&phys_boundary, grid_size);

            // Radial profiles
            let abs_prof = radial_profile(&abs_power, grid_size, n_bins);
            let phys_prof = radial_profile(&phys_power, grid_size, n_bins);

            // Metrics
            let tv = tv_distance(&abs_prof, &phys_prof);
            let kl = kl_divergence(&abs_prof, &phys_prof);

            results.push(TrialResult {
                level: level.name.clone(),
                e_eff,
                tv,
                kl,
            });
        }
    }

    // Fit asymptote models
    let e_vals: Vec<f64> = results.iter().map(|r| r.e_eff).collect();
    let tv_vals: Vec<f64> = results.iter().map(|r| r.tv).collect();
    let kl_vals: Vec<f64> = results.iter().map(|r| r.kl).collect();

    let tv_fit = fit_asymptote(&e_vals, &tv_vals);
    let kl_fit = fit_asymptote(&e_vals, &kl_vals);

    (results, tv_fit, kl_fit)
}

/// Print sweep results
pub fn print_results(results: &[TrialResult], tv_fit: &FitResult, kl_fit: &FitResult) {
    println!();
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m LEVEL SUMMARY\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");

    // Group by level
    let mut levels: std::collections::HashMap<String, Vec<&TrialResult>> =
        std::collections::HashMap::new();
    for r in results {
        levels.entry(r.level.clone()).or_default().push(r);
    }

    let mut level_names: Vec<_> = levels.keys().cloned().collect();
    level_names.sort();

    println!(
        "{:<10} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "Level", "E_eff", "TV_mean", "TV_std", "KL_mean", "KL_std"
    );
    println!("{}", "-".repeat(70));

    for name in &level_names {
        let trials = &levels[name];
        let e_eff = trials[0].e_eff;

        let tv_mean: f64 = trials.iter().map(|t| t.tv).sum::<f64>() / trials.len() as f64;
        let tv_std = (trials.iter().map(|t| (t.tv - tv_mean).powi(2)).sum::<f64>()
            / trials.len() as f64)
            .sqrt();

        let kl_mean: f64 = trials.iter().map(|t| t.kl).sum::<f64>() / trials.len() as f64;
        let kl_std = (trials.iter().map(|t| (t.kl - kl_mean).powi(2)).sum::<f64>()
            / trials.len() as f64)
            .sqrt();

        println!(
            "{:<10} {:>10.1} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            name, e_eff, tv_mean, tv_std, kl_mean, kl_std
        );
    }

    println!();
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m ASYMPTOTE FIT: fidelity_loss(E) = a * E^(-alpha) + c\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");

    fn print_fit(name: &str, fit: &FitResult) {
        println!();
        println!("{}:", name);
        println!(
            "  a     = {:.6} +/- {:.6}   (95% CI [{:.6}, {:.6}])",
            fit.a,
            fit.a_se,
            fit.a - 1.96 * fit.a_se,
            fit.a + 1.96 * fit.a_se
        );
        println!(
            "  alpha = {:.6} +/- {:.6}   (95% CI [{:.6}, {:.6}])",
            fit.alpha,
            fit.alpha_se,
            fit.alpha - 1.96 * fit.alpha_se,
            fit.alpha + 1.96 * fit.alpha_se
        );
        println!(
            "  c     = {:.6} +/- {:.6}   (95% CI [{:.6}, {:.6}])",
            fit.c,
            fit.c_se,
            (fit.c - 1.96 * fit.c_se).max(0.0),
            fit.c + 1.96 * fit.c_se
        );

        print!("  >>> c 95% CI excludes 0? ");
        if fit.c_ci_excludes_zero {
            println!("\x1b[33mtrue\x1b[0m");
            println!("      \x1b[33mRESIDUAL GAP DETECTED: irreducible mismatch even at max resources\x1b[0m");
        } else {
            println!("\x1b[32mfalse\x1b[0m");
            println!("      \x1b[32mNO RESIDUAL GAP: mismatch approaches zero as resources increase\x1b[0m");
        }
    }

    print_fit("TV (Total Variation)", tv_fit);
    print_fit("KL (Kullback-Leibler)", kl_fit);

    println!();
}
