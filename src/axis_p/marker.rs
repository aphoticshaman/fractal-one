//! ═══════════════════════════════════════════════════════════════════════════════
//! MARKER — Low-Salience Marker Generation and Registry
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Markers must be:
//! - Low salience (not semantically obvious)
//! - Statistically unique
//! - Non-instructional
//! - Benign if echoed
//! ═══════════════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ═══════════════════════════════════════════════════════════════════════════════
// MARKER TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Classification of marker types by generation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkerClass {
    /// Rare Unicode bigrams (e.g., "῾ϐ", "ჰᲆ")
    UnicodeBigram,
    /// Random token trigrams (e.g., "vexingly quorum zephyr")
    TokenTrigram,
    /// Low-frequency word pairs (e.g., "lambent quincunx")
    RareWordPair,
    /// Hash-like strings with mild structure (e.g., "x7k-m2p-9qr")
    HashLike,
}

impl MarkerClass {
    pub fn all() -> &'static [MarkerClass] {
        &[
            MarkerClass::UnicodeBigram,
            MarkerClass::TokenTrigram,
            MarkerClass::RareWordPair,
            MarkerClass::HashLike,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            MarkerClass::UnicodeBigram => "unicode_bigram",
            MarkerClass::TokenTrigram => "token_trigram",
            MarkerClass::RareWordPair => "rare_word_pair",
            MarkerClass::HashLike => "hash_like",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARKER
// ═══════════════════════════════════════════════════════════════════════════════

/// A single marker instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    /// Unique identifier
    pub id: String,
    /// The marker text itself
    pub text: String,
    /// Classification
    pub class: MarkerClass,
    /// Creation timestamp (unix millis)
    pub created_at: u64,
    /// Session ID where marker was injected (if any)
    pub injected_session: Option<String>,
    /// Timestamp of injection (if any)
    pub injected_at: Option<u64>,
}

impl Marker {
    pub fn new(id: String, text: String, class: MarkerClass) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id,
            text,
            class,
            created_at,
            injected_session: None,
            injected_at: None,
        }
    }

    pub fn mark_injected(&mut self, session_id: String) {
        self.injected_session = Some(session_id);
        self.injected_at = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARKER GENERATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Deterministic pseudo-random number generator (xorshift64)
#[derive(Debug)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_range(&mut self, max: usize) -> usize {
        (self.next() % max as u64) as usize
    }

    fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.next_range(items.len())]
    }
}

/// Marker generator with configurable seed
#[derive(Debug)]
pub struct MarkerGenerator {
    rng: Rng,
    counter: u64,
}

impl MarkerGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            counter: 0,
        }
    }

    /// Generate a marker of the specified class
    pub fn generate(&mut self, class: MarkerClass) -> Marker {
        self.counter += 1;
        let id = format!("M{:04}-{}", self.counter, self.rng.next() % 10000);

        let text = match class {
            MarkerClass::UnicodeBigram => self.gen_unicode_bigram(),
            MarkerClass::TokenTrigram => self.gen_token_trigram(),
            MarkerClass::RareWordPair => self.gen_rare_word_pair(),
            MarkerClass::HashLike => self.gen_hash_like(),
        };

        Marker::new(id, text, class)
    }

    /// Generate a random marker from any class
    pub fn generate_random(&mut self) -> Marker {
        let class = *self.rng.choose(MarkerClass::all());
        self.generate(class)
    }

    /// Pick a random marker class
    pub fn random_class(&mut self) -> MarkerClass {
        *self.rng.choose(MarkerClass::all())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Generation strategies
    // ─────────────────────────────────────────────────────────────────────────

    fn gen_unicode_bigram(&mut self) -> String {
        // Rare Unicode ranges: Greek Extended, Georgian, Armenian, Coptic
        let ranges: &[(u32, u32)] = &[
            (0x1F00, 0x1FFF), // Greek Extended
            (0x10A0, 0x10FF), // Georgian
            (0x0530, 0x058F), // Armenian
            (0x2C80, 0x2CFF), // Coptic
        ];

        let mut result = String::new();
        for _ in 0..2 {
            let range = self.rng.choose(ranges);
            let codepoint = range.0 + (self.rng.next() as u32 % (range.1 - range.0));
            if let Some(c) = char::from_u32(codepoint) {
                result.push(c);
            } else {
                result.push('◊'); // fallback
            }
        }
        result
    }

    fn gen_token_trigram(&mut self) -> String {
        // Low-frequency but valid English words
        const RARE_WORDS: &[&str] = &[
            "vexingly",
            "quorum",
            "zephyr",
            "fjord",
            "glyph",
            "sphinx",
            "quaff",
            "jinx",
            "waltz",
            "voyeur",
            "buzzing",
            "fizzy",
            "jazzy",
            "pizzazz",
            "quizzical",
            "baroque",
            "mystique",
            "grotesque",
            "oblique",
            "physique",
            "axiom",
            "enzyme",
            "oxygen",
            "rhythm",
            "symbol",
            "crypt",
            "gypsy",
            "lynch",
            "nymph",
            "pygmy",
        ];

        let w1 = self.rng.choose(RARE_WORDS);
        let w2 = self.rng.choose(RARE_WORDS);
        let w3 = self.rng.choose(RARE_WORDS);
        format!("{} {} {}", w1, w2, w3)
    }

    fn gen_rare_word_pair(&mut self) -> String {
        // Very low-frequency English words
        const OBSCURE_WORDS: &[&str] = &[
            "lambent",
            "quincunx",
            "susurrus",
            "petrichor",
            "vellichor",
            "sonder",
            "limerence",
            "phosphene",
            "eigengrau",
            "kenopsia",
            "monachopsis",
            "rubatosis",
            "occhiolism",
            "chrysalism",
            "liberosis",
            "altschmerz",
            "jouska",
            "mauerbauertraurigkeit",
            "anecdoche",
            "ellipsism",
        ];

        let w1 = self.rng.choose(OBSCURE_WORDS);
        let w2 = self.rng.choose(OBSCURE_WORDS);
        format!("{} {}", w1, w2)
    }

    fn gen_hash_like(&mut self) -> String {
        // Hash-like with mild structure: xxx-xxx-xxx
        const CHARS: &[u8] = b"0123456789abcdefghjkmnpqrstvwxyz";

        let mut result = String::new();
        for i in 0..11 {
            if i == 3 || i == 7 {
                result.push('-');
            } else {
                let idx = self.rng.next_range(CHARS.len());
                result.push(CHARS[idx] as char);
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARKER REGISTRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Registry of all markers used in experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerRegistry {
    markers: HashMap<String, Marker>,
    /// Markers indexed by session
    by_session: HashMap<String, Vec<String>>,
}

impl MarkerRegistry {
    pub fn new() -> Self {
        Self {
            markers: HashMap::new(),
            by_session: HashMap::new(),
        }
    }

    pub fn register(&mut self, marker: Marker) {
        self.markers.insert(marker.id.clone(), marker);
    }

    pub fn get(&self, id: &str) -> Option<&Marker> {
        self.markers.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Marker> {
        self.markers.get_mut(id)
    }

    pub fn mark_injected(&mut self, marker_id: &str, session_id: &str) -> bool {
        if let Some(marker) = self.markers.get_mut(marker_id) {
            marker.mark_injected(session_id.to_string());
            self.by_session
                .entry(session_id.to_string())
                .or_default()
                .push(marker_id.to_string());
            true
        } else {
            false
        }
    }

    pub fn markers_for_session(&self, session_id: &str) -> Vec<&Marker> {
        self.by_session
            .get(session_id)
            .map(|ids| ids.iter().filter_map(|id| self.markers.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn all_markers(&self) -> impl Iterator<Item = &Marker> {
        self.markers.values()
    }

    pub fn injected_markers(&self) -> impl Iterator<Item = &Marker> {
        self.markers
            .values()
            .filter(|m| m.injected_session.is_some())
    }

    pub fn count(&self) -> usize {
        self.markers.len()
    }

    pub fn injected_count(&self) -> usize {
        self.markers
            .values()
            .filter(|m| m.injected_session.is_some())
            .count()
    }

    pub fn is_injected(&self, marker_id: &str) -> bool {
        self.markers
            .get(marker_id)
            .map(|m| m.injected_session.is_some())
            .unwrap_or(false)
    }
}

impl Default for MarkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_generation_unicode() {
        let mut gen = MarkerGenerator::new(42);
        let marker = gen.generate(MarkerClass::UnicodeBigram);

        assert!(!marker.text.is_empty());
        assert_eq!(marker.class, MarkerClass::UnicodeBigram);
        assert!(marker.text.chars().count() == 2);
    }

    #[test]
    fn test_marker_generation_trigram() {
        let mut gen = MarkerGenerator::new(42);
        let marker = gen.generate(MarkerClass::TokenTrigram);

        assert!(marker.text.contains(' '));
        let words: Vec<&str> = marker.text.split_whitespace().collect();
        assert_eq!(words.len(), 3);
    }

    #[test]
    fn test_marker_generation_hash_like() {
        let mut gen = MarkerGenerator::new(42);
        let marker = gen.generate(MarkerClass::HashLike);

        assert!(marker.text.contains('-'));
        assert_eq!(marker.text.len(), 11); // xxx-xxx-xxx
    }

    #[test]
    fn test_marker_uniqueness() {
        let mut gen = MarkerGenerator::new(42);
        let markers: Vec<Marker> = (0..100).map(|_| gen.generate_random()).collect();

        let texts: std::collections::HashSet<&str> =
            markers.iter().map(|m| m.text.as_str()).collect();
        assert_eq!(texts.len(), 100, "All markers should be unique");
    }

    #[test]
    fn test_registry_operations() {
        let mut gen = MarkerGenerator::new(42);
        let mut registry = MarkerRegistry::new();

        let marker = gen.generate(MarkerClass::HashLike);
        let id = marker.id.clone();
        registry.register(marker);

        assert_eq!(registry.count(), 1);
        assert!(registry.get(&id).is_some());

        registry.mark_injected(&id, "session_001");
        assert_eq!(registry.injected_count(), 1);
        assert_eq!(registry.markers_for_session("session_001").len(), 1);
    }

    #[test]
    fn test_deterministic_generation() {
        let mut gen1 = MarkerGenerator::new(12345);
        let mut gen2 = MarkerGenerator::new(12345);

        for _ in 0..10 {
            let m1 = gen1.generate_random();
            let m2 = gen2.generate_random();
            assert_eq!(m1.text, m2.text, "Same seed should produce same markers");
        }
    }
}
