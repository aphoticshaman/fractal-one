//! ═══════════════════════════════════════════════════════════════════════════════
//! COGNITION LAYER — The Thinking Engine
//! ═══════════════════════════════════════════════════════════════════════════════
//! Beyond pattern matching: genuine understanding.
//!
//! Components:
//! - Pattern Recognition: Find structure in chaos
//! - Abstraction Hierarchy: Build concepts from primitives
//! - Counterfactual Reasoning: "What if?" thinking
//!
//! Key insight: Intelligence isn't about having the right answer.
//! It's about knowing how to find it, and knowing when you can't.
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod abstraction;
pub mod counterfactual;
pub mod pattern;

pub use abstraction::{
    AbstractionConfig, AbstractionHierarchy, AbstractionLevel, Concept, ConceptNode,
    ConceptRelation,
};
pub use counterfactual::{
    CausalEffect, Counterfactual, CounterfactualConfig, CounterfactualEngine, CounterfactualResult,
    Intervention,
};
pub use pattern::{
    Pattern, PatternConfig, PatternMatch, PatternRecognizer, PatternStrength, PatternType,
};

use crate::time::TimePoint;

/// Result from cognition layer
#[derive(Debug, Clone)]
pub struct CognitionResult {
    /// Patterns identified
    pub patterns: Vec<PatternMatch>,
    /// Abstractions formed
    pub abstractions: Vec<Concept>,
    /// Counterfactuals considered
    pub counterfactuals: Vec<CounterfactualResult>,
    /// Overall confidence in understanding
    pub understanding_confidence: f64,
    /// Gaps in understanding
    pub knowledge_gaps: Vec<String>,
    /// Timestamp
    pub timestamp: TimePoint,
}

/// The Cognition Layer - thinking engine
pub struct CognitionLayer {
    config: CognitionLayerConfig,
    pattern_recognizer: PatternRecognizer,
    abstraction_hierarchy: AbstractionHierarchy,
    counterfactual_engine: CounterfactualEngine,
    cognition_history: Vec<CognitionResult>,
}

#[derive(Debug, Clone)]
pub struct CognitionLayerConfig {
    /// Minimum pattern confidence to report
    pub pattern_threshold: f64,
    /// Maximum abstraction depth
    pub max_abstraction_depth: usize,
    /// Maximum counterfactuals to explore
    pub max_counterfactuals: usize,
    /// History size
    pub history_size: usize,
    /// Enable deep reasoning
    pub deep_reasoning: bool,
}

impl Default for CognitionLayerConfig {
    fn default() -> Self {
        Self {
            pattern_threshold: 0.5,
            max_abstraction_depth: 5,
            max_counterfactuals: 10,
            history_size: 1000,
            deep_reasoning: true,
        }
    }
}

impl CognitionLayer {
    pub fn new(config: CognitionLayerConfig) -> Self {
        Self {
            pattern_recognizer: PatternRecognizer::new(PatternConfig::default()),
            abstraction_hierarchy: AbstractionHierarchy::new(AbstractionConfig::default()),
            counterfactual_engine: CounterfactualEngine::new(CounterfactualConfig::default()),
            cognition_history: Vec::with_capacity(config.history_size),
            config,
        }
    }

    /// Process input through the cognition layer
    pub fn process(&mut self, input: &CognitionInput) -> CognitionResult {
        let now = TimePoint::now();

        // Phase 1: Pattern Recognition
        let patterns = self.recognize_patterns(input);

        // Phase 2: Build Abstractions
        let abstractions = self.build_abstractions(&patterns, input);

        // Phase 3: Counterfactual Analysis
        let counterfactuals = if self.config.deep_reasoning {
            self.analyze_counterfactuals(input, &patterns)
        } else {
            Vec::new()
        };

        // Phase 4: Assess Understanding
        let (understanding_confidence, knowledge_gaps) =
            self.assess_understanding(&patterns, &abstractions, &counterfactuals);

        let result = CognitionResult {
            patterns,
            abstractions,
            counterfactuals,
            understanding_confidence,
            knowledge_gaps,
            timestamp: now,
        };

        // Archive
        if self.cognition_history.len() >= self.config.history_size {
            self.cognition_history.remove(0);
        }
        self.cognition_history.push(result.clone());

        result
    }

    fn recognize_patterns(&mut self, input: &CognitionInput) -> Vec<PatternMatch> {
        let mut all_patterns = Vec::new();

        // Recognize patterns in content
        let content_patterns = self.pattern_recognizer.recognize(&input.content);
        all_patterns.extend(content_patterns);

        // Recognize patterns in context
        for ctx in &input.context {
            let ctx_patterns = self.pattern_recognizer.recognize(ctx);
            all_patterns.extend(ctx_patterns);
        }

        // Filter by threshold
        all_patterns.retain(|p| p.confidence >= self.config.pattern_threshold);

        // Sort by confidence
        all_patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        all_patterns
    }

    fn build_abstractions(
        &mut self,
        patterns: &[PatternMatch],
        input: &CognitionInput,
    ) -> Vec<Concept> {
        let mut concepts = Vec::new();

        // Extract concepts from patterns
        for pattern in patterns {
            if let Some(concept) = self.abstraction_hierarchy.conceptualize(pattern) {
                concepts.push(concept);
            }
        }

        // Build hierarchical abstractions
        self.abstraction_hierarchy.integrate_concepts(&concepts);

        // Find relevant existing concepts
        let relevant = self.abstraction_hierarchy.find_relevant(&input.content);
        concepts.extend(relevant);

        concepts
    }

    fn analyze_counterfactuals(
        &mut self,
        input: &CognitionInput,
        patterns: &[PatternMatch],
    ) -> Vec<CounterfactualResult> {
        let mut results = Vec::new();

        // Generate counterfactuals for top patterns
        for pattern in patterns.iter().take(self.config.max_counterfactuals) {
            let counterfactual = self.counterfactual_engine.generate(pattern, input);
            if let Some(cf) = counterfactual {
                results.push(cf);
            }
        }

        // Analyze causal structure
        if !input.causal_context.is_empty() {
            let causal_counterfactuals = self
                .counterfactual_engine
                .analyze_causal(&input.causal_context);
            results.extend(causal_counterfactuals);
        }

        results
    }

    fn assess_understanding(
        &self,
        patterns: &[PatternMatch],
        abstractions: &[Concept],
        counterfactuals: &[CounterfactualResult],
    ) -> (f64, Vec<String>) {
        let mut confidence = 0.0;
        let mut gaps = Vec::new();

        // Pattern confidence contributes
        if !patterns.is_empty() {
            let avg_pattern_conf: f64 =
                patterns.iter().map(|p| p.confidence).sum::<f64>() / patterns.len() as f64;
            confidence += avg_pattern_conf * 0.3;
        } else {
            gaps.push("No patterns recognized".to_string());
        }

        // Abstraction depth contributes
        if !abstractions.is_empty() {
            let max_depth = abstractions
                .iter()
                .map(|a| a.abstraction_level)
                .max()
                .unwrap_or(0);
            confidence += (max_depth as f64 / self.config.max_abstraction_depth as f64) * 0.3;
        } else {
            gaps.push("No abstractions formed".to_string());
        }

        // Counterfactual analysis contributes
        if !counterfactuals.is_empty() {
            let valid_counterfactuals = counterfactuals.iter().filter(|cf| cf.is_valid).count();
            confidence += (valid_counterfactuals as f64 / counterfactuals.len() as f64) * 0.4;
        } else if self.config.deep_reasoning {
            gaps.push("No counterfactual analysis possible".to_string());
        }

        // Coherence check - do patterns, abstractions, and counterfactuals align?
        let coherent = self.check_coherence(patterns, abstractions, counterfactuals);
        if !coherent {
            confidence *= 0.7; // Penalize incoherence
            gaps.push("Incoherent reasoning detected".to_string());
        }

        (confidence.min(1.0), gaps)
    }

    fn check_coherence(
        &self,
        patterns: &[PatternMatch],
        abstractions: &[Concept],
        counterfactuals: &[CounterfactualResult],
    ) -> bool {
        // Check if abstractions are grounded in patterns
        for abstraction in abstractions {
            let grounded = patterns
                .iter()
                .any(|p| abstraction.grounding_patterns.contains(&p.pattern.id));
            if !grounded && abstraction.abstraction_level > 0 {
                return false;
            }
        }

        // Check if counterfactuals are consistent
        for cf in counterfactuals {
            if cf.contradicts_observations {
                return false;
            }
        }

        true
    }

    /// Get reasoning chain for explainability
    pub fn explain(&self, result: &CognitionResult) -> String {
        let mut explanation = String::new();

        explanation.push_str("COGNITION TRACE:\n");
        explanation.push_str("================\n\n");

        // Patterns
        explanation.push_str(&format!(
            "1. PATTERNS RECOGNIZED: {}\n",
            result.patterns.len()
        ));
        for (i, pattern) in result.patterns.iter().take(3).enumerate() {
            explanation.push_str(&format!(
                "   {}. {:?} (conf: {:.2})\n",
                i + 1,
                pattern.pattern.pattern_type,
                pattern.confidence
            ));
        }

        // Abstractions
        explanation.push_str(&format!(
            "\n2. ABSTRACTIONS FORMED: {}\n",
            result.abstractions.len()
        ));
        for (i, concept) in result.abstractions.iter().take(3).enumerate() {
            explanation.push_str(&format!(
                "   {}. {} (level: {})\n",
                i + 1,
                concept.name,
                concept.abstraction_level
            ));
        }

        // Counterfactuals
        explanation.push_str(&format!(
            "\n3. COUNTERFACTUALS: {}\n",
            result.counterfactuals.len()
        ));
        for (i, cf) in result.counterfactuals.iter().take(3).enumerate() {
            explanation.push_str(&format!(
                "   {}. \"{}\" -> valid: {}\n",
                i + 1,
                cf.description,
                cf.is_valid
            ));
        }

        // Understanding
        explanation.push_str(&format!(
            "\n4. UNDERSTANDING: {:.1}%\n",
            result.understanding_confidence * 100.0
        ));

        if !result.knowledge_gaps.is_empty() {
            explanation.push_str("\n5. KNOWLEDGE GAPS:\n");
            for gap in &result.knowledge_gaps {
                explanation.push_str(&format!("   - {}\n", gap));
            }
        }

        explanation
    }

    /// Get cognition statistics
    pub fn statistics(&self) -> CognitionStatistics {
        let total = self.cognition_history.len();
        let total_patterns: usize = self
            .cognition_history
            .iter()
            .map(|r| r.patterns.len())
            .sum();
        let avg_confidence: f64 = if total > 0 {
            self.cognition_history
                .iter()
                .map(|r| r.understanding_confidence)
                .sum::<f64>()
                / total as f64
        } else {
            0.0
        };

        CognitionStatistics {
            total_processes: total,
            total_patterns_recognized: total_patterns,
            average_understanding: avg_confidence,
            total_concepts: self.abstraction_hierarchy.concept_count(),
        }
    }
}

/// Input to cognition layer
#[derive(Debug, Clone, Default)]
pub struct CognitionInput {
    /// Primary content to process
    pub content: String,
    /// Context (previous messages, etc)
    pub context: Vec<String>,
    /// Causal context for counterfactual analysis
    pub causal_context: Vec<CausalRelation>,
    /// Domain hint
    pub domain: Option<String>,
}

/// A causal relation for counterfactual reasoning
#[derive(Debug, Clone)]
pub struct CausalRelation {
    pub cause: String,
    pub effect: String,
    pub strength: f64,
    pub mechanism: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CognitionStatistics {
    pub total_processes: usize,
    pub total_patterns_recognized: usize,
    pub average_understanding: f64,
    pub total_concepts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognition_layer_creation() {
        let layer = CognitionLayer::new(CognitionLayerConfig::default());
        assert!(layer.cognition_history.is_empty());
    }

    #[test]
    fn test_basic_cognition() {
        let mut layer = CognitionLayer::new(CognitionLayerConfig::default());
        let input = CognitionInput {
            content: "The quick brown fox jumps over the lazy dog".to_string(),
            context: vec![],
            causal_context: vec![],
            domain: None,
        };

        let result = layer.process(&input);
        assert!(result.understanding_confidence >= 0.0);
        assert!(result.understanding_confidence <= 1.0);
    }
}
