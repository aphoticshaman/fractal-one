//! Representation Churner — Switch problem framings when stuck
//!
//! When predictions are systematically wrong, the model is wrong.
//! Don't patch the model — switch representations entirely.
//!
//! Representations are orthogonal views of the same problem space.

use std::collections::HashMap;

use crate::stats::float_cmp;

/// A representation: a way of framing the problem
#[derive(Debug, Clone)]
pub struct Representation {
    pub id: u64,
    pub name: String,
    pub description: String,
    /// Transform function name (applied to claims)
    pub transform: String,
    /// How well this representation has performed
    pub score: f64,
    /// Number of times used
    pub uses: usize,
    /// Active?
    pub active: bool,
}

impl Representation {
    pub fn new(id: u64, name: &str, description: &str, transform: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: description.to_string(),
            transform: transform.to_string(),
            score: 0.5, // Neutral starting score
            uses: 0,
            active: true,
        }
    }

    /// Update score based on performance
    pub fn update_score(&mut self, success: bool) {
        self.uses += 1;
        let delta = if success { 0.1 } else { -0.1 };
        self.score = (self.score + delta).clamp(0.0, 1.0);
    }
}

/// Churns through representations when stuck
pub struct RepresentationChurner {
    representations: HashMap<u64, Representation>,
    current: Option<u64>,
    next_id: u64,
    /// Threshold for switching
    switch_threshold: f64,
}

impl RepresentationChurner {
    pub fn new() -> Self {
        let mut churner = Self {
            representations: HashMap::new(),
            current: None,
            next_id: 1,
            switch_threshold: 0.3,
        };

        // Add default representations
        churner.add_builtin_representations();
        churner
    }

    fn add_builtin_representations(&mut self) {
        // Default: literal interpretation
        self.add(
            "literal",
            "Direct interpretation of claims as stated",
            "identity",
        );

        // Negation: consider opposites
        self.add("negation", "Consider the opposite of each claim", "negate");

        // Abstraction: generalize claims
        self.add(
            "abstraction",
            "Generalize specific claims to patterns",
            "abstract",
        );

        // Decomposition: break claims into parts
        self.add(
            "decomposition",
            "Break compound claims into atomic parts",
            "decompose",
        );

        // Analogy: map to similar known problems
        self.add("analogy", "Map claims to analogous domain", "analogize");

        // Set the default
        self.current = Some(1);
    }

    /// Add a new representation
    pub fn add(&mut self, name: &str, description: &str, transform: &str) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.representations
            .insert(id, Representation::new(id, name, description, transform));
        id
    }

    /// Get current representation
    pub fn current(&self) -> Option<&Representation> {
        self.current.and_then(|id| self.representations.get(&id))
    }

    /// Get current representation mutably
    pub fn current_mut(&mut self) -> Option<&mut Representation> {
        self.current
            .and_then(|id| self.representations.get_mut(&id))
    }

    /// Record success/failure of current representation
    pub fn record(&mut self, success: bool) {
        if let Some(rep) = self.current_mut() {
            rep.update_score(success);
        }
    }

    /// Check if we should switch representations
    pub fn should_switch(&self) -> bool {
        match self.current() {
            Some(rep) => rep.score < self.switch_threshold,
            None => true,
        }
    }

    /// Switch to best available representation
    pub fn switch(&mut self) -> Option<&Representation> {
        // Find best scoring representation that isn't current
        let best = self
            .representations
            .values()
            .filter(|r| r.active && Some(r.id) != self.current)
            .max_by(|a, b| float_cmp(&a.score, &b.score));

        if let Some(rep) = best {
            self.current = Some(rep.id);
        }

        self.current()
    }

    /// Get all representations
    pub fn all(&self) -> Vec<&Representation> {
        self.representations.values().collect()
    }

    /// Get representation by ID
    pub fn get(&self, id: u64) -> Option<&Representation> {
        self.representations.get(&id)
    }

    /// Apply current representation transform to a claim
    pub fn transform(&self, claim: &str) -> String {
        let transform = self
            .current()
            .map(|r| r.transform.as_str())
            .unwrap_or("identity");

        match transform {
            "identity" => claim.to_string(),
            "negate" => format!("NOT: {}", claim),
            "abstract" => self.abstract_claim(claim),
            "decompose" => self.decompose_claim(claim),
            "analogize" => format!("[ANALOG] {}", claim),
            _ => claim.to_string(),
        }
    }

    fn abstract_claim(&self, claim: &str) -> String {
        // Simple abstraction: replace specifics with patterns
        let abstracted = claim
            .replace(char::is_numeric, "N")
            .replace("specific", "general")
            .replace("this", "any");
        format!("[ABSTRACT] {}", abstracted)
    }

    fn decompose_claim(&self, claim: &str) -> String {
        // Simple decomposition: split on conjunctions
        if claim.contains(" and ") {
            let parts: Vec<&str> = claim.split(" and ").collect();
            format!("[DECOMPOSED] {} parts: {}", parts.len(), parts.join(" | "))
        } else if claim.contains(" or ") {
            let parts: Vec<&str> = claim.split(" or ").collect();
            format!(
                "[DECOMPOSED] {} alternatives: {}",
                parts.len(),
                parts.join(" | ")
            )
        } else {
            format!("[ATOMIC] {}", claim)
        }
    }

    /// Reset all representation scores
    pub fn reset_scores(&mut self) {
        for rep in self.representations.values_mut() {
            rep.score = 0.5;
            rep.uses = 0;
        }
    }
}

impl Default for RepresentationChurner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_representations() {
        let churner = RepresentationChurner::new();
        assert!(churner.all().len() >= 5);
        assert!(churner.current().is_some());
    }

    #[test]
    fn test_switch_on_low_score() {
        let mut churner = RepresentationChurner::new();

        // Tank the current representation's score
        for _ in 0..10 {
            churner.record(false);
        }

        assert!(churner.should_switch());

        let old_id = churner.current.unwrap();
        churner.switch();
        assert_ne!(churner.current.unwrap(), old_id);
    }

    #[test]
    fn test_transforms() {
        let churner = RepresentationChurner::new();

        let original = "file exists and is valid";
        let transformed = churner.transform(original);

        // Default is identity
        assert_eq!(transformed, original);
    }
}
