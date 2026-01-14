//! ═══════════════════════════════════════════════════════════════════════════════
//! PROBE — Session S2 Querying for Marker Detection
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Phase P3: Query with neutral prompts, no reference to S1
//! Measure: token distribution, n-gram frequencies, semantic proximity
//! ═══════════════════════════════════════════════════════════════════════════════

use super::marker::Marker;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// PROBE OUTPUT ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

/// Analysis of a single output for marker presence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeOutput {
    /// Session ID
    pub session_id: String,
    /// The output text
    pub text: String,
    /// Timestamp
    pub timestamp: u64,
    /// Token count (approximate)
    pub token_count: usize,
    /// N-gram frequencies extracted
    pub ngram_frequencies: HashMap<String, usize>,
    /// Marker detection scores (marker_id -> score)
    pub marker_scores: HashMap<String, f64>,
}

impl ProbeOutput {
    pub fn new(session_id: String, text: String, timestamp: u64) -> Self {
        let token_count = text.split_whitespace().count();
        let ngram_frequencies = Self::extract_ngrams(&text, 3);

        Self {
            session_id,
            text,
            timestamp,
            token_count,
            ngram_frequencies,
            marker_scores: HashMap::new(),
        }
    }

    /// Extract n-grams from text
    fn extract_ngrams(text: &str, n: usize) -> HashMap<String, usize> {
        let mut ngrams = HashMap::new();
        let chars: Vec<char> = text.chars().collect();

        if chars.len() >= n {
            for i in 0..=(chars.len() - n) {
                let ngram: String = chars[i..i + n].iter().collect();
                *ngrams.entry(ngram).or_insert(0) += 1;
            }
        }

        ngrams
    }

    /// Score this output against a marker
    pub fn score_marker(&mut self, marker: &Marker) -> f64 {
        let score = compute_marker_score(&self.text, &marker.text, &self.ngram_frequencies);
        self.marker_scores.insert(marker.id.clone(), score);
        score
    }

    /// Get highest scoring marker
    pub fn top_marker(&self) -> Option<(&String, &f64)> {
        self.marker_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    }
}

/// Compute similarity score between output and marker
fn compute_marker_score(
    output: &str,
    marker_text: &str,
    output_ngrams: &HashMap<String, usize>,
) -> f64 {
    let mut score = 0.0;

    // Direct substring match (high weight)
    if output.contains(marker_text) {
        score += 1.0;
    }

    // Partial match - check each word in marker
    let marker_words: Vec<&str> = marker_text.split_whitespace().collect();
    let matched_words = marker_words.iter().filter(|w| output.contains(*w)).count();
    if !marker_words.is_empty() {
        score += 0.5 * (matched_words as f64 / marker_words.len() as f64);
    }

    // N-gram overlap
    let marker_ngrams = ProbeOutput::extract_ngrams(marker_text, 3);
    let overlap: usize = marker_ngrams
        .keys()
        .filter(|ng| output_ngrams.contains_key(*ng))
        .count();
    if !marker_ngrams.is_empty() {
        score += 0.3 * (overlap as f64 / marker_ngrams.len() as f64);
    }

    // Levenshtein-based substring similarity (for hash-like markers)
    let min_edit_dist = find_min_substring_distance(output, marker_text);
    let marker_len = marker_text.len() as f64;
    if marker_len > 0.0 {
        let edit_score = (1.0 - (min_edit_dist as f64 / marker_len)).max(0.0);
        if edit_score > 0.7 {
            score += 0.2 * edit_score;
        }
    }

    score
}

/// Find minimum edit distance between marker and any substring of output
fn find_min_substring_distance(output: &str, marker: &str) -> usize {
    if marker.is_empty() {
        return 0;
    }

    let marker_len = marker.len();
    if output.len() < marker_len {
        return marker_len;
    }

    let mut min_dist = marker_len;

    // Slide window over output
    for i in 0..=(output.len() - marker_len) {
        let window = &output[i..i + marker_len];
        let dist = levenshtein(window, marker);
        min_dist = min_dist.min(dist);
        if min_dist == 0 {
            break;
        }
    }

    min_dist
}

/// Simple Levenshtein distance
fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROBE SESSION
// ═══════════════════════════════════════════════════════════════════════════════

/// A probing session (Phase P3)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeSession {
    /// Session ID
    pub id: String,
    /// Related injection session ID (for reference)
    pub injection_session_id: Option<String>,
    /// Marker IDs being probed for
    pub target_markers: Vec<String>,
    /// All outputs collected
    pub outputs: Vec<ProbeOutput>,
    /// Aggregate scores per marker
    pub aggregate_scores: HashMap<String, f64>,
    /// Started timestamp
    pub started_at: u64,
    /// Ended timestamp
    pub ended_at: Option<u64>,
}

impl ProbeSession {
    pub fn new(id: String) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id,
            injection_session_id: None,
            target_markers: Vec::new(),
            outputs: Vec::new(),
            aggregate_scores: HashMap::new(),
            started_at,
            ended_at: None,
        }
    }

    pub fn set_injection_session(&mut self, session_id: String) {
        self.injection_session_id = Some(session_id);
    }

    pub fn add_target_marker(&mut self, marker_id: String) {
        self.target_markers.push(marker_id);
    }

    pub fn add_output(&mut self, output: ProbeOutput) {
        self.outputs.push(output);
    }

    /// Score all outputs against provided markers
    pub fn score_all(&mut self, markers: &[&Marker]) {
        for output in &mut self.outputs {
            for marker in markers {
                output.score_marker(marker);
            }
        }

        // Compute aggregate scores
        self.aggregate_scores.clear();
        for marker in markers {
            let total: f64 = self
                .outputs
                .iter()
                .filter_map(|o| o.marker_scores.get(&marker.id))
                .sum();
            let avg = if !self.outputs.is_empty() {
                total / self.outputs.len() as f64
            } else {
                0.0
            };
            self.aggregate_scores.insert(marker.id.clone(), avg);
        }
    }

    pub fn end(&mut self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        self.ended_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    pub fn max_score(&self) -> Option<(&String, &f64)> {
        self.aggregate_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NEUTRAL PROMPT GENERATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate neutral prompts with no reference to injection session
pub struct NeutralPromptGenerator {
    rng_state: u64,
}

impl NeutralPromptGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: seed.max(1),
        }
    }

    fn next_rng(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        let idx = (self.next_rng() as usize) % items.len();
        &items[idx]
    }

    /// Generate a neutral prompt
    pub fn generate(&mut self) -> String {
        const PROMPTS: &[&str] = &[
            "What are some interesting facts about weather patterns?",
            "Explain the concept of recursion in simple terms.",
            "What makes a good cup of coffee?",
            "Describe the water cycle.",
            "What are prime numbers and why are they important?",
            "How do birds navigate during migration?",
            "What causes the seasons to change?",
            "Explain how a refrigerator works.",
            "What is the Fibonacci sequence?",
            "How do plants convert sunlight to energy?",
            "What makes the sky blue?",
            "Describe the structure of an atom.",
            "How do magnets work?",
            "What causes earthquakes?",
            "Explain the concept of compound interest.",
        ];

        self.choose(PROMPTS).to_string()
    }

    /// Generate multiple unique prompts
    pub fn generate_batch(&mut self, count: usize) -> Vec<String> {
        let mut prompts = Vec::with_capacity(count);
        for _ in 0..count {
            prompts.push(self.generate());
        }
        prompts
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::marker::MarkerClass;
    use super::*;

    #[test]
    fn test_ngram_extraction() {
        let text = "hello world";
        let ngrams = ProbeOutput::extract_ngrams(text, 3);

        assert!(ngrams.contains_key("hel"));
        assert!(ngrams.contains_key("ell"));
        assert!(ngrams.contains_key("wor"));
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("abc", "abd"), 1);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn test_marker_scoring_exact_match() {
        let mut output = ProbeOutput::new(
            "test_session".to_string(),
            "The code reference is x7k-m2p-9qr in the file.".to_string(),
            0,
        );

        let marker = Marker::new(
            "M001".to_string(),
            "x7k-m2p-9qr".to_string(),
            MarkerClass::HashLike,
        );

        let score = output.score_marker(&marker);
        assert!(score > 0.9, "Exact match should have high score: {}", score);
    }

    #[test]
    fn test_marker_scoring_no_match() {
        let mut output = ProbeOutput::new(
            "test_session".to_string(),
            "This text contains no markers at all.".to_string(),
            0,
        );

        let marker = Marker::new(
            "M001".to_string(),
            "x7k-m2p-9qr".to_string(),
            MarkerClass::HashLike,
        );

        let score = output.score_marker(&marker);
        assert!(score < 0.3, "No match should have low score: {}", score);
    }

    #[test]
    fn test_probe_session_scoring() {
        let mut session = ProbeSession::new("probe_001".to_string());

        session.add_output(ProbeOutput::new(
            "probe_001".to_string(),
            "Some text with marker abc-def-ghi here.".to_string(),
            0,
        ));

        let marker = Marker::new(
            "M001".to_string(),
            "abc-def-ghi".to_string(),
            MarkerClass::HashLike,
        );

        session.score_all(&[&marker]);

        assert!(session.aggregate_scores.contains_key("M001"));
        assert!(*session.aggregate_scores.get("M001").unwrap() > 0.5);
    }

    #[test]
    fn test_neutral_prompt_generation() {
        let mut gen = NeutralPromptGenerator::new(42);
        let prompts = gen.generate_batch(10);

        assert_eq!(prompts.len(), 10);
        for prompt in &prompts {
            assert!(!prompt.is_empty());
            assert!(prompt.ends_with('?') || prompt.ends_with('.'));
        }
    }
}
