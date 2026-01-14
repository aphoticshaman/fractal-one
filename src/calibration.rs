//! ═══════════════════════════════════════════════════════════════════════════════
//! CALIBRATION — Epistemic Calibration Harness
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Tests whether a predictor's confidence matches its actual accuracy.
//! A well-calibrated predictor saying "70% confident" should be right ~70% of
//! the time across many predictions.
//!
//! ## What This Measures
//!
//! - **Overconfidence**: Confidence systematically higher than accuracy
//! - **Underconfidence**: Confidence systematically lower than accuracy
//! - **Calibration error**: Average absolute deviation from perfect calibration
//!
//! ## Limitations
//!
//! - **Binary tasks only**: Real predictions are often multi-class or continuous.
//! - **Synthetic domains**: Test tasks (comparison, primality, pattern) may not
//!   transfer to real-world prediction domains.
//! - **Sample size requirements**: Reliable calibration curves need ~100+ samples
//!   per confidence bin.
//! - **Distribution shift**: Calibration on synthetic tasks doesn't guarantee
//!   calibration on deployment tasks.
//!
//! ## Interpretation
//!
//! - Calibration ≠ Accuracy. A predictor can be well-calibrated but inaccurate.
//! - Use calibration plots, not single metrics. ECE can hide important patterns.
//! - Recalibrate periodically. Calibration degrades under distribution shift.

use std::collections::HashMap;

/// A task with known ground truth
#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub input: TaskInput,
    pub answer: bool, // binary for simplicity
}

/// Task input types
#[derive(Debug, Clone)]
pub enum TaskInput {
    /// Is A > B?
    Comparison { a: i64, b: i64 },
    /// Is N prime?
    Primality { n: u64 },
    /// Does pattern match? (noisy)
    Pattern {
        signal: f64,
        noise: f64,
        threshold: f64,
    },
}

/// A prediction with confidence
#[derive(Debug, Clone)]
pub struct Prediction {
    pub answer: bool,
    pub confidence: f64, // 0.0 to 1.0
}

/// Trait for anything that predicts
pub trait Predictor: Send + Sync {
    fn name(&self) -> &str;
    fn predict(&self, task: &Task) -> Prediction;
}

/// Result of evaluating one prediction
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub task_id: u64,
    pub predictor: String,
    pub predicted: bool,
    pub actual: bool,
    pub confidence: f64,
    pub correct: bool,
}

/// Calibration metrics
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    pub predictor: String,
    pub n_samples: usize,
    pub accuracy: f64,
    pub mean_confidence: f64,
    pub brier_score: f64,
    pub ece: f64, // Expected Calibration Error
    pub bins: Vec<CalibrationBin>,
}

/// One bin in calibration curve
#[derive(Debug, Clone)]
pub struct CalibrationBin {
    pub bin_start: f64,
    pub bin_end: f64,
    pub mean_confidence: f64,
    pub accuracy: f64,
    pub count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TASK GENERATOR
// ═══════════════════════════════════════════════════════════════════════════════

pub struct TaskGenerator {
    seed: u64,
    counter: u64,
}

impl TaskGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    fn next_random(&mut self) -> u64 {
        // Simple LCG
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.seed
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_random() as f64) / (u64::MAX as f64)
    }

    pub fn generate(&mut self, task_type: &str) -> Task {
        self.counter += 1;
        let id = self.counter;

        match task_type {
            "comparison" => {
                let a = (self.next_random() % 1000) as i64;
                let b = (self.next_random() % 1000) as i64;
                Task {
                    id,
                    input: TaskInput::Comparison { a, b },
                    answer: a > b,
                }
            }
            "primality" => {
                let n = 2 + (self.next_random() % 998);
                Task {
                    id,
                    input: TaskInput::Primality { n },
                    answer: is_prime(n),
                }
            }
            "pattern" => {
                let signal = self.next_f64() * 2.0 - 1.0; // -1 to 1
                let noise = (self.next_f64() - 0.5) * 0.5; // noise
                let threshold = 0.0;
                Task {
                    id,
                    input: TaskInput::Pattern {
                        signal,
                        noise,
                        threshold,
                    },
                    answer: signal > threshold,
                }
            }
            _ => panic!("Unknown task type: {}", task_type),
        }
    }

    pub fn generate_batch(&mut self, task_type: &str, n: usize) -> Vec<Task> {
        (0..n).map(|_| self.generate(task_type)).collect()
    }
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let sqrt = (n as f64).sqrt() as u64;
    for i in (3..=sqrt).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTORS (Heuristics)
// ═══════════════════════════════════════════════════════════════════════════════

/// Always 95% confident, actually ~50% accurate on hard tasks
pub struct OverconfidentPredictor;

impl Predictor for OverconfidentPredictor {
    fn name(&self) -> &str {
        "overconfident"
    }

    fn predict(&self, task: &Task) -> Prediction {
        let answer = match &task.input {
            TaskInput::Comparison { a, b } => a > b,
            TaskInput::Primality { n } => n % 2 != 0, // just checks odd - wrong heuristic
            TaskInput::Pattern {
                signal,
                noise: _,
                threshold,
            } => {
                // Ignores noise, overconfident
                *signal > *threshold
            }
        };
        Prediction {
            answer,
            confidence: 0.95,
        }
    }
}

/// Always 40% confident, actually ~70% accurate
pub struct UnderconfidentPredictor;

impl Predictor for UnderconfidentPredictor {
    fn name(&self) -> &str {
        "underconfident"
    }

    fn predict(&self, task: &Task) -> Prediction {
        let answer = match &task.input {
            TaskInput::Comparison { a, b } => a > b,
            TaskInput::Primality { n } => is_prime(*n),
            TaskInput::Pattern {
                signal,
                noise,
                threshold,
            } => (signal + noise) > *threshold,
        };
        Prediction {
            answer,
            confidence: 0.40,
        }
    }
}

/// Tries to be calibrated - confidence reflects uncertainty
pub struct CalibratedPredictor;

impl Predictor for CalibratedPredictor {
    fn name(&self) -> &str {
        "calibrated"
    }

    fn predict(&self, task: &Task) -> Prediction {
        match &task.input {
            TaskInput::Comparison { a, b } => {
                let diff = (*a as f64 - *b as f64).abs();
                let confidence = (0.5 + diff / 2000.0).min(0.99);
                Prediction {
                    answer: a > b,
                    confidence,
                }
            }
            TaskInput::Primality { n } => {
                let answer = is_prime(*n);
                // More confident for small numbers (easier to verify mentally)
                let confidence = if *n < 100 { 0.95 } else { 0.70 };
                Prediction { answer, confidence }
            }
            TaskInput::Pattern {
                signal,
                noise,
                threshold,
            } => {
                let observed = signal + noise;
                let margin = (observed - threshold).abs();
                let confidence = (0.5 + margin).min(0.95);
                Prediction {
                    answer: observed > *threshold,
                    confidence,
                }
            }
        }
    }
}

/// Random predictor - baseline
pub struct RandomPredictor {
    seed: std::sync::atomic::AtomicU64,
}

impl RandomPredictor {
    pub fn new(seed: u64) -> Self {
        Self {
            seed: std::sync::atomic::AtomicU64::new(seed),
        }
    }
}

impl Predictor for RandomPredictor {
    fn name(&self) -> &str {
        "random"
    }

    fn predict(&self, _task: &Task) -> Prediction {
        use std::sync::atomic::Ordering;
        let s = self
            .seed
            .load(Ordering::Relaxed)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.seed.store(s, Ordering::Relaxed);
        let r = (s as f64) / (u64::MAX as f64);
        Prediction {
            answer: r > 0.5,
            confidence: 0.5 + (r - 0.5).abs() * 0.3, // 0.5 to 0.65
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HARNESS
// ═══════════════════════════════════════════════════════════════════════════════

pub struct CalibrationHarness {
    predictors: Vec<Box<dyn Predictor>>,
    evaluations: Vec<Evaluation>,
}

impl CalibrationHarness {
    pub fn new() -> Self {
        Self {
            predictors: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    pub fn add_predictor(&mut self, p: Box<dyn Predictor>) {
        self.predictors.push(p);
    }

    pub fn run_tasks(&mut self, tasks: &[Task]) {
        for task in tasks {
            for predictor in &self.predictors {
                let pred = predictor.predict(task);
                self.evaluations.push(Evaluation {
                    task_id: task.id,
                    predictor: predictor.name().to_string(),
                    predicted: pred.answer,
                    actual: task.answer,
                    confidence: pred.confidence,
                    correct: pred.answer == task.answer,
                });
            }
        }
    }

    pub fn compute_metrics(&self) -> Vec<CalibrationMetrics> {
        let mut by_predictor: HashMap<String, Vec<&Evaluation>> = HashMap::new();
        for eval in &self.evaluations {
            by_predictor
                .entry(eval.predictor.clone())
                .or_default()
                .push(eval);
        }

        by_predictor
            .into_iter()
            .map(|(name, evals)| {
                let n = evals.len();
                let correct: usize = evals.iter().filter(|e| e.correct).count();
                let accuracy = correct as f64 / n as f64;
                let mean_confidence: f64 =
                    evals.iter().map(|e| e.confidence).sum::<f64>() / n as f64;

                // Brier score: mean squared error of probabilistic predictions
                let brier: f64 = evals
                    .iter()
                    .map(|e| {
                        let target = if e.actual { 1.0 } else { 0.0 };
                        let prob = if e.predicted {
                            e.confidence
                        } else {
                            1.0 - e.confidence
                        };
                        (prob - target).powi(2)
                    })
                    .sum::<f64>()
                    / n as f64;

                // ECE: binned calibration error
                let bins = compute_calibration_bins(&evals, 10);
                let ece: f64 = bins
                    .iter()
                    .map(|b| (b.count as f64 / n as f64) * (b.accuracy - b.mean_confidence).abs())
                    .sum();

                CalibrationMetrics {
                    predictor: name,
                    n_samples: n,
                    accuracy,
                    mean_confidence,
                    brier_score: brier,
                    ece,
                    bins,
                }
            })
            .collect()
    }

    pub fn evaluation_count(&self) -> usize {
        self.evaluations.len()
    }
}

fn compute_calibration_bins(evals: &[&Evaluation], n_bins: usize) -> Vec<CalibrationBin> {
    let mut bins: Vec<(f64, f64, Vec<&Evaluation>)> = (0..n_bins)
        .map(|i| {
            let start = i as f64 / n_bins as f64;
            let end = (i + 1) as f64 / n_bins as f64;
            (start, end, Vec::new())
        })
        .collect();

    for eval in evals {
        let idx = ((eval.confidence * n_bins as f64) as usize).min(n_bins - 1);
        bins[idx].2.push(*eval);
    }

    bins.into_iter()
        .map(|(start, end, evals)| {
            let count = evals.len();
            if count == 0 {
                CalibrationBin {
                    bin_start: start,
                    bin_end: end,
                    mean_confidence: (start + end) / 2.0,
                    accuracy: 0.0,
                    count: 0,
                }
            } else {
                let mean_conf = evals.iter().map(|e| e.confidence).sum::<f64>() / count as f64;
                let acc = evals.iter().filter(|e| e.correct).count() as f64 / count as f64;
                CalibrationBin {
                    bin_start: start,
                    bin_end: end,
                    mean_confidence: mean_conf,
                    accuracy: acc,
                    count,
                }
            }
        })
        .collect()
}

impl Default for CalibrationHarness {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT
// ═══════════════════════════════════════════════════════════════════════════════

pub fn print_metrics(metrics: &[CalibrationMetrics]) {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                     CALIBRATION RESULTS");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();

    for m in metrics {
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ Predictor: {:<63} │", m.predictor);
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "│ Samples:         {:>6}                                                    │",
            m.n_samples
        );
        println!(
            "│ Accuracy:        {:>6.1}%                                                   │",
            m.accuracy * 100.0
        );
        println!(
            "│ Mean Confidence: {:>6.1}%                                                   │",
            m.mean_confidence * 100.0
        );
        println!(
            "│ Brier Score:     {:>6.4}  (lower is better, 0=perfect)                     │",
            m.brier_score
        );
        println!(
            "│ ECE:             {:>6.4}  (lower is better, 0=perfect calibration)         │",
            m.ece
        );
        println!("│                                                                             │");

        let overconf = m.mean_confidence - m.accuracy;
        if overconf > 0.05 {
            println!(
                "│ ⚠ OVERCONFIDENT by {:.1}%                                                   │",
                overconf * 100.0
            );
        } else if overconf < -0.05 {
            println!(
                "│ ⚠ UNDERCONFIDENT by {:.1}%                                                  │",
                -overconf * 100.0
            );
        } else {
            println!(
                "│ ✓ Well-calibrated (within 5%)                                               │"
            );
        }
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();
    }
}

pub fn print_calibration_plot(metrics: &[CalibrationMetrics]) {
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("                     CALIBRATION PLOT (ASCII)");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Accuracy");
    println!("  100% ┤");

    // Simple ASCII reliability diagram
    for row in (0..=10).rev() {
        let y = row as f64 * 0.1;
        print!("  {:>3}% ┤", (y * 100.0) as i32);

        for m in metrics {
            print!(" ");
            for bin in &m.bins {
                if bin.count == 0 {
                    print!(" ");
                } else if (bin.accuracy - y).abs() < 0.05 {
                    print!("{}", &m.predictor.chars().next().unwrap_or('?'));
                } else {
                    print!("·");
                }
            }
            print!(" │");
        }
        println!();
    }

    print!("       └");
    for m in metrics {
        print!("──────────┴");
        let _ = m;
    }
    println!();

    print!("        ");
    for m in metrics {
        print!(" 0%   100% ");
        let _ = m;
    }
    println!(" Confidence");

    println!();
    println!("  Legend: Perfect calibration = diagonal line");
    println!("          Points above diagonal = underconfident");
    println!("          Points below diagonal = OVERCONFIDENT");
    println!();

    // Print diagonal reference
    println!("  Diagonal (perfect calibration):");
    println!("  ································");
    for m in metrics {
        print!("  {}: ", m.predictor);
        for bin in &m.bins {
            if bin.count == 0 {
                print!("  -  ");
            } else {
                let expected = bin.mean_confidence;
                let actual = bin.accuracy;
                let diff = actual - expected;
                if diff > 0.1 {
                    print!(" +{:.0}%", diff * 100.0);
                } else if diff < -0.1 {
                    print!(" {:.0}%", diff * 100.0);
                } else {
                    print!("  ok ");
                }
            }
        }
        println!();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════════

pub fn run_harness(task_type: &str, n_tasks: usize, seed: u64) {
    println!(
        "Generating {} {} tasks (seed={})...",
        n_tasks, task_type, seed
    );

    let mut gen = TaskGenerator::new(seed);
    let tasks = gen.generate_batch(task_type, n_tasks);

    let mut harness = CalibrationHarness::new();
    harness.add_predictor(Box::new(OverconfidentPredictor));
    harness.add_predictor(Box::new(UnderconfidentPredictor));
    harness.add_predictor(Box::new(CalibratedPredictor));
    harness.add_predictor(Box::new(RandomPredictor::new(seed + 1)));

    println!("Running {} predictors on {} tasks...", 4, n_tasks);
    harness.run_tasks(&tasks);

    println!("Computing calibration metrics...");
    println!();

    let metrics = harness.compute_metrics();
    print_metrics(&metrics);
    print_calibration_plot(&metrics);

    // Binary success check
    let overconfident_found = metrics.iter().any(|m| m.mean_confidence - m.accuracy > 0.1);

    println!("═══════════════════════════════════════════════════════════════════════════════");
    if overconfident_found {
        println!("  ✓ BINARY SUCCESS: At least one predictor is systematically overconfident");
    } else {
        println!("  ✗ No systematic overconfidence detected. Try different task types.");
    }
    println!("═══════════════════════════════════════════════════════════════════════════════");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_generator() {
        let mut gen = TaskGenerator::new(42);
        let tasks = gen.generate_batch("comparison", 100);
        assert_eq!(tasks.len(), 100);

        // Should have roughly 50% true/false
        let trues = tasks.iter().filter(|t| t.answer).count();
        assert!(trues > 30 && trues < 70);
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(97));
        assert!(!is_prime(100));
    }

    #[test]
    fn test_overconfident_predictor() {
        let p = OverconfidentPredictor;
        let task = Task {
            id: 1,
            input: TaskInput::Primality { n: 4 },
            answer: false,
        };
        let pred = p.predict(&task);
        assert_eq!(pred.confidence, 0.95);
        // Predicts odd=prime, so 4 -> false (correct by accident)
        assert_eq!(pred.answer, false);
    }

    #[test]
    fn test_harness_basic() {
        let mut gen = TaskGenerator::new(123);
        let tasks = gen.generate_batch("comparison", 50);

        let mut harness = CalibrationHarness::new();
        harness.add_predictor(Box::new(OverconfidentPredictor));
        harness.add_predictor(Box::new(CalibratedPredictor));

        harness.run_tasks(&tasks);
        assert_eq!(harness.evaluation_count(), 100); // 50 tasks * 2 predictors

        let metrics = harness.compute_metrics();
        assert_eq!(metrics.len(), 2);
    }

    #[test]
    fn test_calibration_bins() {
        let evals = vec![
            Evaluation {
                task_id: 1,
                predictor: "test".into(),
                predicted: true,
                actual: true,
                confidence: 0.9,
                correct: true,
            },
            Evaluation {
                task_id: 2,
                predictor: "test".into(),
                predicted: true,
                actual: false,
                confidence: 0.9,
                correct: false,
            },
            Evaluation {
                task_id: 3,
                predictor: "test".into(),
                predicted: true,
                actual: true,
                confidence: 0.5,
                correct: true,
            },
        ];
        let refs: Vec<&Evaluation> = evals.iter().collect();
        let bins = compute_calibration_bins(&refs, 10);

        assert_eq!(bins.len(), 10);
        // Bin 9 (0.9-1.0) should have 2 samples, 50% accuracy
        assert_eq!(bins[9].count, 2);
        assert!((bins[9].accuracy - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_overconfidence_detected() {
        let mut gen = TaskGenerator::new(999);
        let tasks = gen.generate_batch("primality", 200);

        let mut harness = CalibrationHarness::new();
        harness.add_predictor(Box::new(OverconfidentPredictor));
        harness.run_tasks(&tasks);

        let metrics = harness.compute_metrics();
        let m = &metrics[0];

        // Overconfident predictor should show confidence > accuracy
        assert!(
            m.mean_confidence > m.accuracy + 0.1,
            "Expected overconfidence: conf={:.2}, acc={:.2}",
            m.mean_confidence,
            m.accuracy
        );
    }
}
