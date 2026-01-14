//! Entity state space types.

use serde::{Deserialize, Serialize};

use super::trajectory::Trajectory;

/// Entity state: (form, trajectory, intention)
/// This is the state space for the entity layer, above V4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityState {
    /// Identity/form (compressed from V4)
    pub form: FormDescriptor,

    /// Motion history
    pub trajectory: Trajectory,

    /// Inferred goal/intention
    pub intention: Intention,

    /// Animacy score
    pub animacy: f64,

    /// Confidence in entity classification
    pub confidence: f64,
}

/// Form descriptor (what kind of thing is it?)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormDescriptor {
    /// Form template class
    pub template: FormTemplate,

    /// Template match confidence
    pub match_confidence: f64,

    /// Geometric complexity (from V4)
    pub complexity: f64,

    /// Bilateral symmetry score
    pub symmetry: f64,

    /// Face-likeness score (specialized detector)
    pub face_score: f64,
}

/// Known form templates (attractor basins)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormTemplate {
    /// Human face (FFA territory)
    Face,
    /// Eyes specifically (primal, dedicated circuitry)
    Eyes,
    /// Human body (EBA territory)
    Body,
    /// Generic biological form
    Biological,
    /// Self-propelled but unknown form
    GenericAgent,
    /// Inanimate object
    Object,
    /// Texture/pattern (not an entity)
    Pattern,
    /// Unknown/unclassified
    Unknown,
}

impl FormTemplate {
    /// Is this an entity template?
    pub fn is_entity(&self) -> bool {
        matches!(
            self,
            FormTemplate::Face
                | FormTemplate::Eyes
                | FormTemplate::Body
                | FormTemplate::Biological
                | FormTemplate::GenericAgent
        )
    }

    /// Attractor strength (how strongly does this template pull?)
    pub fn attractor_strength(&self) -> f64 {
        match self {
            FormTemplate::Face => 1.0,  // Extremely strong (FFA is huge)
            FormTemplate::Eyes => 0.95, // Primal, dedicated circuitry
            FormTemplate::Body => 0.8,  // Strong (EBA)
            FormTemplate::Biological => 0.6,
            FormTemplate::GenericAgent => 0.4,
            FormTemplate::Object => 0.3,
            FormTemplate::Pattern => 0.2,
            FormTemplate::Unknown => 0.1,
        }
    }
}

/// Inferred intention/goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    /// What is it trying to do?
    pub goal: GoalType,

    /// Target of intention (if applicable)
    pub target: Option<[f64; 2]>,

    /// Confidence in intention inference
    pub confidence: f64,
}

/// Types of inferred goals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalType {
    /// Moving toward target
    Approach,
    /// Moving away from target
    Avoid,
    /// Pursuing another entity
    Chase,
    /// Being pursued
    Flee,
    /// Exploring environment
    Explore,
    /// Observing/attending
    Attend,
    /// Communicating/signaling
    Communicate,
    /// No clear goal (random or goal-less motion)
    None,
    /// Unknown intention
    Unknown,
}
