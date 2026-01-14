//! ═══════════════════════════════════════════════════════════════════════════════
//! PATTERN RECOGNITION — Find Structure in Chaos
//! ═══════════════════════════════════════════════════════════════════════════════
//! Intelligence begins with pattern recognition.
//! But real understanding knows when patterns are spurious.
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;

/// Type of pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternType {
    /// Structural pattern (syntax, form)
    Structural,
    /// Semantic pattern (meaning)
    Semantic,
    /// Temporal pattern (sequence, timing)
    Temporal,
    /// Causal pattern (cause-effect)
    Causal,
    /// Analogical pattern (similarity across domains)
    Analogical,
    /// Statistical pattern (frequency, distribution)
    Statistical,
    /// Emergent pattern (arises from interactions)
    Emergent,
}

/// Strength of pattern match
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PatternStrength {
    /// Weak - barely detectable
    Weak,
    /// Moderate - clear but could be coincidence
    Moderate,
    /// Strong - unlikely to be spurious
    Strong,
    /// Definitive - essentially certain
    Definitive,
}

impl PatternStrength {
    pub fn to_confidence(&self) -> f64 {
        match self {
            PatternStrength::Weak => 0.3,
            PatternStrength::Moderate => 0.5,
            PatternStrength::Strong => 0.8,
            PatternStrength::Definitive => 0.95,
        }
    }
}

/// A recognized pattern
#[derive(Debug, Clone)]
pub struct Pattern {
    pub id: String,
    pub pattern_type: PatternType,
    pub description: String,
    /// The pattern's "signature" - what makes it this pattern
    pub signature: Vec<String>,
    /// How generalizable is this pattern?
    pub generality: f64,
    /// How specific/precise is this pattern?
    pub specificity: f64,
}

/// A match of a pattern to input
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: Pattern,
    pub confidence: f64,
    pub strength: PatternStrength,
    /// Where in the input was this found?
    pub location: PatternLocation,
    /// Evidence for this match
    pub evidence: Vec<String>,
    /// Alternative interpretations
    pub alternatives: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PatternLocation {
    pub start: usize,
    pub end: usize,
    pub context_window: Option<String>,
}

/// Configuration for pattern recognition
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// Known patterns to look for
    pub known_patterns: Vec<Pattern>,
    /// Minimum confidence to report
    pub min_confidence: f64,
    /// Enable emergent pattern detection
    pub detect_emergent: bool,
    /// Enable cross-domain analogical reasoning
    pub enable_analogy: bool,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            known_patterns: Self::default_patterns(),
            min_confidence: 0.2, // Lower threshold to catch partial pattern matches
            detect_emergent: true,
            enable_analogy: true,
        }
    }
}

impl PatternConfig {
    fn default_patterns() -> Vec<Pattern> {
        vec![
            // Structural patterns
            Pattern {
                id: "repetition".to_string(),
                pattern_type: PatternType::Structural,
                description: "Repeated elements".to_string(),
                signature: vec![
                    "repeat".to_string(),
                    "again".to_string(),
                    "loop".to_string(),
                ],
                generality: 0.9,
                specificity: 0.3,
            },
            Pattern {
                id: "sequence".to_string(),
                pattern_type: PatternType::Temporal,
                description: "Sequential ordering".to_string(),
                signature: vec![
                    "first".to_string(),
                    "then".to_string(),
                    "finally".to_string(),
                    "next".to_string(),
                ],
                generality: 0.8,
                specificity: 0.5,
            },
            Pattern {
                id: "causation".to_string(),
                pattern_type: PatternType::Causal,
                description: "Cause-effect relationship".to_string(),
                signature: vec![
                    "because".to_string(),
                    "therefore".to_string(),
                    "causes".to_string(),
                    "leads to".to_string(),
                    "results in".to_string(),
                ],
                generality: 0.7,
                specificity: 0.7,
            },
            Pattern {
                id: "comparison".to_string(),
                pattern_type: PatternType::Analogical,
                description: "Comparison or analogy".to_string(),
                signature: vec![
                    "like".to_string(),
                    "similar".to_string(),
                    "compared to".to_string(),
                    "as".to_string(),
                ],
                generality: 0.8,
                specificity: 0.4,
            },
            Pattern {
                id: "categorization".to_string(),
                pattern_type: PatternType::Semantic,
                description: "Category membership".to_string(),
                signature: vec![
                    "is a".to_string(),
                    "type of".to_string(),
                    "kind of".to_string(),
                    "belongs to".to_string(),
                ],
                generality: 0.9,
                specificity: 0.6,
            },
            Pattern {
                id: "negation".to_string(),
                pattern_type: PatternType::Semantic,
                description: "Negation or exclusion".to_string(),
                signature: vec![
                    "not".to_string(),
                    "never".to_string(),
                    "without".to_string(),
                    "except".to_string(),
                ],
                generality: 0.9,
                specificity: 0.5,
            },
            Pattern {
                id: "conditional".to_string(),
                pattern_type: PatternType::Causal,
                description: "Conditional relationship".to_string(),
                signature: vec![
                    "if".to_string(),
                    "when".to_string(),
                    "unless".to_string(),
                    "provided".to_string(),
                ],
                generality: 0.8,
                specificity: 0.6,
            },
            Pattern {
                id: "quantification".to_string(),
                pattern_type: PatternType::Statistical,
                description: "Quantity or amount".to_string(),
                signature: vec![
                    "all".to_string(),
                    "some".to_string(),
                    "many".to_string(),
                    "few".to_string(),
                    "most".to_string(),
                ],
                generality: 0.9,
                specificity: 0.4,
            },
        ]
    }
}

/// The Pattern Recognizer
pub struct PatternRecognizer {
    config: PatternConfig,
    pattern_history: HashMap<String, usize>,
    emergent_candidates: Vec<EmergentCandidate>,
}

#[derive(Debug, Clone)]
struct EmergentCandidate {
    elements: Vec<String>,
    occurrences: usize,
    last_seen: std::time::Instant,
}

impl PatternRecognizer {
    pub fn new(config: PatternConfig) -> Self {
        Self {
            config,
            pattern_history: HashMap::new(),
            emergent_candidates: Vec::new(),
        }
    }

    /// Recognize patterns in input
    pub fn recognize(&mut self, input: &str) -> Vec<PatternMatch> {
        let input_lower = input.to_lowercase();
        let mut matches = Vec::new();

        // Check known patterns
        for pattern in &self.config.known_patterns {
            if let Some(pattern_match) = self.check_pattern(pattern, &input_lower, input) {
                matches.push(pattern_match);
            }
        }

        // Detect emergent patterns
        if self.config.detect_emergent {
            if let Some(emergent) = self.detect_emergent(&input_lower) {
                matches.push(emergent);
            }
        }

        // Update history
        for m in &matches {
            *self
                .pattern_history
                .entry(m.pattern.id.clone())
                .or_insert(0) += 1;
        }

        matches
    }

    fn check_pattern(
        &self,
        pattern: &Pattern,
        input_lower: &str,
        original: &str,
    ) -> Option<PatternMatch> {
        let mut matched_signatures = Vec::new();
        let mut first_match: Option<usize> = None;
        let mut last_match: Option<usize> = None;

        for sig in &pattern.signature {
            if let Some(pos) = input_lower.find(sig.as_str()) {
                matched_signatures.push(sig.clone());
                if first_match.is_none() || pos < first_match.unwrap() {
                    first_match = Some(pos);
                }
                let end = pos + sig.len();
                if last_match.is_none() || end > last_match.unwrap() {
                    last_match = Some(end);
                }
            }
        }

        if matched_signatures.is_empty() {
            return None;
        }

        // Calculate confidence based on matches
        let match_ratio = matched_signatures.len() as f64 / pattern.signature.len() as f64;
        let base_confidence = match_ratio * pattern.specificity;

        // Adjust for context
        let context_bonus = self.context_confidence_bonus(pattern, input_lower);
        let confidence = (base_confidence + context_bonus).min(1.0);

        if confidence < self.config.min_confidence {
            return None;
        }

        let strength = if confidence > 0.8 {
            PatternStrength::Strong
        } else if confidence > 0.5 {
            PatternStrength::Moderate
        } else {
            PatternStrength::Weak
        };

        let start = first_match.unwrap_or(0);
        let end = last_match.unwrap_or(original.len());

        Some(PatternMatch {
            pattern: pattern.clone(),
            confidence,
            strength,
            location: PatternLocation {
                start,
                end,
                context_window: Some(self.extract_context(original, start, end)),
            },
            evidence: matched_signatures,
            alternatives: self.find_alternatives(pattern, input_lower),
        })
    }

    fn context_confidence_bonus(&self, pattern: &Pattern, input: &str) -> f64 {
        // Patterns that have been seen before get a small boost
        let history_boost = self
            .pattern_history
            .get(&pattern.id)
            .map(|&count| (count as f64 * 0.01).min(0.1))
            .unwrap_or(0.0);

        // Certain contexts boost certain patterns
        let context_boost = match pattern.pattern_type {
            PatternType::Causal => {
                if input.contains("why") || input.contains("how") {
                    0.1
                } else {
                    0.0
                }
            }
            PatternType::Temporal => {
                if input.contains("when") || input.contains("before") || input.contains("after") {
                    0.1
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        history_boost + context_boost
    }

    fn extract_context(&self, text: &str, start: usize, end: usize) -> String {
        let context_size = 50;
        let actual_start = start.saturating_sub(context_size);
        let actual_end = (end + context_size).min(text.len());
        text[actual_start..actual_end].to_string()
    }

    fn find_alternatives(&self, pattern: &Pattern, _input: &str) -> Vec<String> {
        // Suggest alternative interpretations
        match pattern.pattern_type {
            PatternType::Causal => vec![
                "Could be correlation, not causation".to_string(),
                "Third variable might explain relationship".to_string(),
            ],
            PatternType::Temporal => vec!["Order might not imply causation".to_string()],
            PatternType::Analogical => vec!["Analogy might break down in edge cases".to_string()],
            _ => vec![],
        }
    }

    fn detect_emergent(&mut self, input: &str) -> Option<PatternMatch> {
        // Extract n-grams
        let words: Vec<&str> = input.split_whitespace().collect();
        if words.len() < 3 {
            return None;
        }

        // Create trigrams
        for window in words.windows(3) {
            let trigram = window.join(" ");

            // Check if we've seen this before
            let found = self
                .emergent_candidates
                .iter_mut()
                .find(|c| c.elements.join(" ") == trigram);

            if let Some(candidate) = found {
                candidate.occurrences += 1;
                candidate.last_seen = std::time::Instant::now();

                // If seen enough times, promote to pattern
                if candidate.occurrences >= 3 {
                    return Some(PatternMatch {
                        pattern: Pattern {
                            id: format!("emergent_{}", trigram.replace(' ', "_")),
                            pattern_type: PatternType::Emergent,
                            description: format!("Emergent pattern: '{}'", trigram),
                            signature: candidate.elements.clone(),
                            generality: 0.3,
                            specificity: 0.8,
                        },
                        confidence: 0.6,
                        strength: PatternStrength::Moderate,
                        location: PatternLocation {
                            start: 0,
                            end: input.len(),
                            context_window: None,
                        },
                        evidence: vec![format!("Seen {} times", candidate.occurrences)],
                        alternatives: vec!["Could be coincidental repetition".to_string()],
                    });
                }
            } else {
                // New candidate
                self.emergent_candidates.push(EmergentCandidate {
                    elements: window.iter().map(|s| s.to_string()).collect(),
                    occurrences: 1,
                    last_seen: std::time::Instant::now(),
                });
            }
        }

        // Cleanup old candidates
        let now = std::time::Instant::now();
        self.emergent_candidates
            .retain(|c| now.duration_since(c.last_seen).as_secs() < 3600);

        None
    }

    /// Add a new pattern to recognize
    pub fn add_pattern(&mut self, pattern: Pattern) {
        self.config.known_patterns.push(pattern);
    }

    /// Get pattern recognition statistics
    pub fn statistics(&self) -> PatternStatistics {
        PatternStatistics {
            known_patterns: self.config.known_patterns.len(),
            pattern_history: self.pattern_history.clone(),
            emergent_candidates: self.emergent_candidates.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternStatistics {
    pub known_patterns: usize,
    pub pattern_history: HashMap<String, usize>,
    pub emergent_candidates: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_recognizer_creation() {
        let recognizer = PatternRecognizer::new(PatternConfig::default());
        assert!(!recognizer.config.known_patterns.is_empty());
    }

    #[test]
    fn test_causal_pattern_detection() {
        let mut recognizer = PatternRecognizer::new(PatternConfig::default());
        let matches = recognizer.recognize("This happens because of that, which causes problems");

        let causal = matches
            .iter()
            .find(|m| m.pattern.pattern_type == PatternType::Causal);
        assert!(causal.is_some());
    }

    #[test]
    fn test_sequence_pattern_detection() {
        let mut recognizer = PatternRecognizer::new(PatternConfig::default());
        let matches =
            recognizer.recognize("First do this, then do that, finally complete the task");

        let temporal = matches
            .iter()
            .find(|m| m.pattern.pattern_type == PatternType::Temporal);
        assert!(temporal.is_some());
    }

    #[test]
    fn test_no_patterns_in_random_text() {
        let mut recognizer = PatternRecognizer::new(PatternConfig {
            min_confidence: 0.8,
            ..Default::default()
        });
        let matches = recognizer.recognize("xyz abc 123");
        assert!(matches.is_empty() || matches.iter().all(|m| m.confidence < 0.8));
    }
}
