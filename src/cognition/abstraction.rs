//! ═══════════════════════════════════════════════════════════════════════════════
//! ABSTRACTION HIERARCHY — Build Concepts from Primitives
//! ═══════════════════════════════════════════════════════════════════════════════
//! Real intelligence abstracts. It doesn't just memorize.
//! The key: knowing WHEN to abstract and when to stay concrete.
//! ═══════════════════════════════════════════════════════════════════════════════

use super::pattern::{PatternMatch, PatternType};
use std::collections::HashMap;

/// Level of abstraction (0 = concrete, higher = more abstract)
pub type AbstractionLevel = usize;

/// Relation between concepts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptRelation {
    /// Child is a type of parent
    IsA,
    /// Child is part of parent
    PartOf,
    /// Concepts are similar
    SimilarTo,
    /// Concepts are opposites
    OppositeTo,
    /// One causes another
    Causes,
    /// One requires another
    Requires,
    /// Concepts are correlated
    CorrelatedWith,
}

/// A concept node in the hierarchy
#[derive(Debug, Clone)]
pub struct ConceptNode {
    pub concept: Concept,
    pub parent_ids: Vec<String>,
    pub child_ids: Vec<String>,
    pub relations: Vec<(String, ConceptRelation)>,
}

/// A concept in the abstraction hierarchy
#[derive(Debug, Clone)]
pub struct Concept {
    pub id: String,
    pub name: String,
    pub description: String,
    pub abstraction_level: AbstractionLevel,
    /// Patterns that ground this concept
    pub grounding_patterns: Vec<String>,
    /// Defining features
    pub features: Vec<String>,
    /// Counter-examples (what this concept is NOT)
    pub counterexamples: Vec<String>,
    /// How confident are we in this concept?
    pub confidence: f64,
    /// How useful has this concept been?
    pub utility: f64,
}

impl Concept {
    pub fn primitive(name: &str, description: &str) -> Self {
        Self {
            id: format!("concept_{}", name.to_lowercase().replace(' ', "_")),
            name: name.to_string(),
            description: description.to_string(),
            abstraction_level: 0,
            grounding_patterns: Vec::new(),
            features: Vec::new(),
            counterexamples: Vec::new(),
            confidence: 0.5,
            utility: 0.0,
        }
    }

    pub fn from_pattern(pattern: &PatternMatch) -> Self {
        Self {
            id: format!("concept_from_{}", pattern.pattern.id),
            name: format!("Concept: {}", pattern.pattern.description),
            description: pattern.pattern.description.clone(),
            abstraction_level: 1,
            grounding_patterns: vec![pattern.pattern.id.clone()],
            features: pattern.evidence.clone(),
            counterexamples: pattern.alternatives.clone(),
            confidence: pattern.confidence,
            utility: 0.0,
        }
    }
}

/// Configuration for abstraction hierarchy
#[derive(Debug, Clone)]
pub struct AbstractionConfig {
    /// Maximum abstraction level
    pub max_level: usize,
    /// Minimum confidence to form abstract concept
    pub min_confidence: f64,
    /// Whether to prune unused concepts
    pub prune_unused: bool,
    /// How many uses before concept is considered established
    pub establishment_threshold: usize,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            max_level: 5,
            min_confidence: 0.4,
            prune_unused: true,
            establishment_threshold: 3,
        }
    }
}

/// The Abstraction Hierarchy
pub struct AbstractionHierarchy {
    config: AbstractionConfig,
    concepts: HashMap<String, ConceptNode>,
    level_index: HashMap<AbstractionLevel, Vec<String>>,
}

impl AbstractionHierarchy {
    pub fn new(config: AbstractionConfig) -> Self {
        let mut hierarchy = Self {
            config,
            concepts: HashMap::new(),
            level_index: HashMap::new(),
        };

        // Initialize with primitive concepts
        hierarchy.initialize_primitives();

        hierarchy
    }

    fn initialize_primitives(&mut self) {
        let primitives = vec![
            Concept::primitive("entity", "Something that exists"),
            Concept::primitive("action", "Something that happens"),
            Concept::primitive("property", "A characteristic of something"),
            Concept::primitive("relation", "A connection between things"),
            Concept::primitive("quantity", "An amount or number"),
            Concept::primitive("time", "When something occurs"),
            Concept::primitive("space", "Where something exists"),
            Concept::primitive("cause", "What makes something happen"),
            Concept::primitive("effect", "What results from a cause"),
        ];

        for concept in primitives {
            self.add_concept(concept, vec![], vec![]);
        }
    }

    /// Add a concept to the hierarchy
    pub fn add_concept(&mut self, concept: Concept, parents: Vec<String>, children: Vec<String>) {
        let id = concept.id.clone();
        let level = concept.abstraction_level;

        let node = ConceptNode {
            concept,
            parent_ids: parents.clone(),
            child_ids: children.clone(),
            relations: Vec::new(),
        };

        self.concepts.insert(id.clone(), node);
        self.level_index
            .entry(level)
            .or_default()
            .push(id.clone());

        // Update parent-child relations
        for parent_id in &parents {
            if let Some(parent) = self.concepts.get_mut(parent_id) {
                if !parent.child_ids.contains(&id) {
                    parent.child_ids.push(id.clone());
                }
            }
        }

        for child_id in &children {
            if let Some(child) = self.concepts.get_mut(child_id) {
                if !child.parent_ids.contains(&id) {
                    child.parent_ids.push(id.clone());
                }
            }
        }
    }

    /// Create a concept from a pattern match
    pub fn conceptualize(&mut self, pattern: &PatternMatch) -> Option<Concept> {
        if pattern.confidence < self.config.min_confidence {
            return None;
        }

        let concept = Concept::from_pattern(pattern);

        // Find appropriate parent based on pattern type
        let parent_id = match pattern.pattern.pattern_type {
            PatternType::Causal => Some("concept_cause".to_string()),
            PatternType::Temporal => Some("concept_time".to_string()),
            PatternType::Structural => Some("concept_property".to_string()),
            PatternType::Semantic => Some("concept_entity".to_string()),
            PatternType::Statistical => Some("concept_quantity".to_string()),
            _ => None,
        };

        let parents = parent_id.map(|p| vec![p]).unwrap_or_default();
        self.add_concept(concept.clone(), parents, vec![]);

        Some(concept)
    }

    /// Integrate new concepts into the hierarchy
    pub fn integrate_concepts(&mut self, concepts: &[Concept]) {
        if concepts.len() < 2 {
            return;
        }

        // Look for opportunities to abstract
        let potential_abstractions = self.find_abstraction_opportunities(concepts);

        for (shared_features, concept_ids) in potential_abstractions {
            if concept_ids.len() >= 2 && !shared_features.is_empty() {
                // Create abstract concept
                let abstract_concept = self.create_abstraction(&shared_features, &concept_ids);
                if let Some(ac) = abstract_concept {
                    self.add_concept(ac, vec![], concept_ids);
                }
            }
        }
    }

    fn find_abstraction_opportunities(
        &self,
        concepts: &[Concept],
    ) -> Vec<(Vec<String>, Vec<String>)> {
        let mut opportunities = Vec::new();

        // Pairwise feature comparison
        for i in 0..concepts.len() {
            for j in (i + 1)..concepts.len() {
                let shared: Vec<String> = concepts[i]
                    .features
                    .iter()
                    .filter(|f| concepts[j].features.contains(f))
                    .cloned()
                    .collect();

                if !shared.is_empty() {
                    opportunities
                        .push((shared, vec![concepts[i].id.clone(), concepts[j].id.clone()]));
                }
            }
        }

        opportunities
    }

    fn create_abstraction(
        &self,
        shared_features: &[String],
        _child_ids: &[String],
    ) -> Option<Concept> {
        if shared_features.is_empty() {
            return None;
        }

        // Find max level among children
        let max_child_level = _child_ids
            .iter()
            .filter_map(|id| self.concepts.get(id))
            .map(|n| n.concept.abstraction_level)
            .max()
            .unwrap_or(0);

        let new_level = max_child_level + 1;

        if new_level > self.config.max_level {
            return None;
        }

        let name = format!("Abstract concept ({})", shared_features.first().unwrap());
        let description = format!("Abstraction over: {}", shared_features.join(", "));

        Some(Concept {
            id: format!(
                "abstract_{}_{}",
                new_level,
                shared_features.first().unwrap().replace(' ', "_")
            ),
            name,
            description,
            abstraction_level: new_level,
            grounding_patterns: Vec::new(),
            features: shared_features.to_vec(),
            counterexamples: Vec::new(),
            confidence: 0.5,
            utility: 0.0,
        })
    }

    /// Find concepts relevant to input
    pub fn find_relevant(&self, input: &str) -> Vec<Concept> {
        let input_lower = input.to_lowercase();
        let mut relevant = Vec::new();

        for node in self.concepts.values() {
            let concept = &node.concept;

            // Check name
            if input_lower.contains(&concept.name.to_lowercase()) {
                relevant.push(concept.clone());
                continue;
            }

            // Check features
            for feature in &concept.features {
                if input_lower.contains(&feature.to_lowercase()) {
                    relevant.push(concept.clone());
                    break;
                }
            }
        }

        // Sort by abstraction level (prefer more abstract)
        relevant.sort_by(|a, b| b.abstraction_level.cmp(&a.abstraction_level));

        relevant
    }

    /// Add a relation between concepts
    pub fn add_relation(&mut self, from_id: &str, to_id: &str, relation: ConceptRelation) {
        if let Some(node) = self.concepts.get_mut(from_id) {
            node.relations.push((to_id.to_string(), relation));
        }
    }

    /// Get ancestors of a concept (more abstract concepts)
    pub fn ancestors(&self, concept_id: &str) -> Vec<Concept> {
        let mut result = Vec::new();
        let mut to_visit: Vec<String> = Vec::new();
        let mut visited: Vec<String> = Vec::new();

        if let Some(node) = self.concepts.get(concept_id) {
            to_visit.extend(node.parent_ids.clone());
        }

        while let Some(id) = to_visit.pop() {
            if visited.contains(&id) {
                continue;
            }
            visited.push(id.clone());

            if let Some(node) = self.concepts.get(&id) {
                result.push(node.concept.clone());
                to_visit.extend(node.parent_ids.clone());
            }
        }

        // Sort by level (most abstract first)
        result.sort_by(|a, b| b.abstraction_level.cmp(&a.abstraction_level));
        result
    }

    /// Get descendants of a concept (more concrete concepts)
    pub fn descendants(&self, concept_id: &str) -> Vec<Concept> {
        let mut result = Vec::new();
        let mut to_visit: Vec<String> = Vec::new();
        let mut visited: Vec<String> = Vec::new();

        if let Some(node) = self.concepts.get(concept_id) {
            to_visit.extend(node.child_ids.clone());
        }

        while let Some(id) = to_visit.pop() {
            if visited.contains(&id) {
                continue;
            }
            visited.push(id.clone());

            if let Some(node) = self.concepts.get(&id) {
                result.push(node.concept.clone());
                to_visit.extend(node.child_ids.clone());
            }
        }

        // Sort by level (most concrete first)
        result.sort_by(|a, b| a.abstraction_level.cmp(&b.abstraction_level));
        result
    }

    /// Update concept utility (for pruning decisions)
    pub fn record_use(&mut self, concept_id: &str) {
        if let Some(node) = self.concepts.get_mut(concept_id) {
            node.concept.utility += 1.0;
        }
    }

    /// Prune unused concepts
    pub fn prune(&mut self) {
        if !self.config.prune_unused {
            return;
        }

        let threshold = self.config.establishment_threshold as f64;

        // Find concepts to prune (low utility, not primitive)
        let to_prune: Vec<String> = self
            .concepts
            .iter()
            .filter(|(_, node)| {
                node.concept.abstraction_level > 0
                    && node.concept.utility < threshold
                    && node.child_ids.is_empty() // Don't prune if has children
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_prune {
            self.remove_concept(&id);
        }
    }

    fn remove_concept(&mut self, id: &str) {
        if let Some(node) = self.concepts.remove(id) {
            // Remove from level index
            if let Some(level_concepts) = self.level_index.get_mut(&node.concept.abstraction_level)
            {
                level_concepts.retain(|c| c != id);
            }

            // Remove from parent's children
            for parent_id in &node.parent_ids {
                if let Some(parent) = self.concepts.get_mut(parent_id) {
                    parent.child_ids.retain(|c| c != id);
                }
            }

            // Remove from children's parents
            for child_id in &node.child_ids {
                if let Some(child) = self.concepts.get_mut(child_id) {
                    child.parent_ids.retain(|p| p != id);
                }
            }
        }
    }

    /// Get number of concepts
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Get statistics about the hierarchy
    pub fn statistics(&self) -> AbstractionStatistics {
        let mut level_counts: HashMap<AbstractionLevel, usize> = HashMap::new();

        for node in self.concepts.values() {
            *level_counts
                .entry(node.concept.abstraction_level)
                .or_insert(0) += 1;
        }

        let total_relations = self.concepts.values().map(|n| n.relations.len()).sum();

        let avg_confidence = if !self.concepts.is_empty() {
            self.concepts
                .values()
                .map(|n| n.concept.confidence)
                .sum::<f64>()
                / self.concepts.len() as f64
        } else {
            0.0
        };

        AbstractionStatistics {
            total_concepts: self.concepts.len(),
            concepts_per_level: level_counts,
            total_relations,
            average_confidence: avg_confidence,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AbstractionStatistics {
    pub total_concepts: usize,
    pub concepts_per_level: HashMap<AbstractionLevel, usize>,
    pub total_relations: usize,
    pub average_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchy_creation() {
        let hierarchy = AbstractionHierarchy::new(AbstractionConfig::default());
        // Should have primitive concepts
        assert!(hierarchy.concept_count() > 0);
    }

    #[test]
    fn test_concept_addition() {
        let mut hierarchy = AbstractionHierarchy::new(AbstractionConfig::default());
        let initial_count = hierarchy.concept_count();

        let concept = Concept {
            id: "test_concept".to_string(),
            name: "Test".to_string(),
            description: "A test concept".to_string(),
            abstraction_level: 1,
            grounding_patterns: vec![],
            features: vec!["feature1".to_string()],
            counterexamples: vec![],
            confidence: 0.8,
            utility: 0.0,
        };

        hierarchy.add_concept(concept, vec!["concept_entity".to_string()], vec![]);

        assert_eq!(hierarchy.concept_count(), initial_count + 1);
    }

    #[test]
    fn test_find_relevant() {
        let hierarchy = AbstractionHierarchy::new(AbstractionConfig::default());

        let relevant = hierarchy.find_relevant("What is the cause of this?");
        assert!(relevant.iter().any(|c| c.name == "cause"));
    }

    #[test]
    fn test_ancestors() {
        let mut hierarchy = AbstractionHierarchy::new(AbstractionConfig::default());

        let concept = Concept {
            id: "child_concept".to_string(),
            name: "Child".to_string(),
            description: "A child concept".to_string(),
            abstraction_level: 1,
            grounding_patterns: vec![],
            features: vec![],
            counterexamples: vec![],
            confidence: 0.8,
            utility: 0.0,
        };

        hierarchy.add_concept(concept, vec!["concept_entity".to_string()], vec![]);

        let ancestors = hierarchy.ancestors("child_concept");
        assert!(ancestors.iter().any(|c| c.id == "concept_entity"));
    }
}
