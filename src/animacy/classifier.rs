//! Entity attractor dynamics and classification.

use serde::{Deserialize, Serialize};

use crate::stats::float_cmp;

use super::detection::AnimacyScore;
use super::entity::{FormDescriptor, FormTemplate};

/// Entity attractor basin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttractorConfig {
    /// Threshold for entity classification
    pub entity_threshold: f64,

    /// Threshold for face detection
    pub face_threshold: f64,

    /// Psychedelic gain multiplier (μ_E in the model)
    /// Higher values lower thresholds → more entity detection
    pub gain: f64,

    /// Winner-take-all competition strength
    pub competition_strength: f64,
}

impl Default for AttractorConfig {
    fn default() -> Self {
        Self {
            entity_threshold: 0.5,
            face_threshold: 0.7,
            gain: 1.0, // Normal perception
            competition_strength: 0.8,
        }
    }
}

impl AttractorConfig {
    /// Low-threshold configuration (psychedelic-like)
    pub fn low_threshold() -> Self {
        Self {
            entity_threshold: 0.2,
            face_threshold: 0.3,
            gain: 2.5,
            competition_strength: 0.5,
        }
    }

    /// High-threshold configuration (conservative)
    pub fn high_threshold() -> Self {
        Self {
            entity_threshold: 0.7,
            face_threshold: 0.85,
            gain: 0.7,
            competition_strength: 0.9,
        }
    }
}

/// Entity attractor classifier
#[derive(Debug, Clone)]
pub struct EntityClassifier {
    config: AttractorConfig,
}

impl EntityClassifier {
    pub fn new(config: AttractorConfig) -> Self {
        Self { config }
    }

    /// Classify an input into an entity attractor basin
    pub fn classify(&self, form: &FormDescriptor, animacy: &AnimacyScore) -> ClassificationResult {
        // Effective thresholds (modified by gain)
        let eff_entity_threshold = self.config.entity_threshold / self.config.gain;
        let eff_face_threshold = self.config.face_threshold / self.config.gain;

        // Compute activation for each attractor
        let mut activations = Vec::new();

        // Face attractor
        let face_activation = form.face_score * FormTemplate::Face.attractor_strength();
        activations.push((FormTemplate::Face, face_activation));

        // Eyes attractor (subset of face features)
        let eyes_activation = (form.face_score * 0.8) * FormTemplate::Eyes.attractor_strength();
        activations.push((FormTemplate::Eyes, eyes_activation));

        // Body attractor
        let body_activation = if form.symmetry > 0.6 && form.complexity > 0.3 {
            form.symmetry * FormTemplate::Body.attractor_strength()
        } else {
            0.0
        };
        activations.push((FormTemplate::Body, body_activation));

        // Biological motion (from animacy)
        let bio_activation =
            animacy.biological_motion * FormTemplate::Biological.attractor_strength();
        activations.push((FormTemplate::Biological, bio_activation));

        // Generic agent (animate but unknown form)
        let agent_activation = if animacy.is_animate && form.match_confidence < 0.5 {
            animacy.confidence * FormTemplate::GenericAgent.attractor_strength()
        } else {
            0.0
        };
        activations.push((FormTemplate::GenericAgent, agent_activation));

        // Object (inanimate, structured)
        let object_activation = if !animacy.is_animate && form.complexity > 0.2 {
            (1.0 - animacy.confidence) * FormTemplate::Object.attractor_strength()
        } else {
            0.0
        };
        activations.push((FormTemplate::Object, object_activation));

        // Pattern (texture, no structure)
        let pattern_activation = if form.complexity < 0.2 {
            (1.0 - form.complexity) * FormTemplate::Pattern.attractor_strength()
        } else {
            0.0
        };
        activations.push((FormTemplate::Pattern, pattern_activation));

        // Apply gain
        for (_, act) in &mut activations {
            *act *= self.config.gain;
        }

        // Winner-take-all competition
        let total_activation: f64 = activations.iter().map(|(_, a)| *a).sum();
        if total_activation > 0.01 {
            for (_, act) in &mut activations {
                let normalized = *act / total_activation;
                *act = normalized.powf(self.config.competition_strength);
            }
        }

        // Find winner
        let (winner_template, winner_activation) = activations
            .iter()
            .max_by(|(_, a1), (_, a2)| float_cmp(a1, a2))
            .map(|(t, a)| (*t, *a))
            .unwrap_or((FormTemplate::Unknown, 0.0));

        // Check if winner exceeds threshold
        let is_entity = winner_template.is_entity() && winner_activation > eff_entity_threshold;
        let is_face = matches!(winner_template, FormTemplate::Face | FormTemplate::Eyes)
            && winner_activation > eff_face_threshold;

        ClassificationResult {
            template: winner_template,
            activation: winner_activation,
            is_entity,
            is_face,
            all_activations: activations,
            effective_threshold: eff_entity_threshold,
        }
    }
}

/// Result of entity classification
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Winning template
    pub template: FormTemplate,

    /// Winning activation level
    pub activation: f64,

    /// Did it cross entity threshold?
    pub is_entity: bool,

    /// Is it specifically a face?
    pub is_face: bool,

    /// All attractor activations
    pub all_activations: Vec<(FormTemplate, f64)>,

    /// Effective threshold used
    pub effective_threshold: f64,
}

impl ClassificationResult {
    /// Is this a "machine elf" case? (Entity attractor without template match)
    pub fn is_untemplated_entity(&self) -> bool {
        self.is_entity
            && matches!(
                self.template,
                FormTemplate::GenericAgent | FormTemplate::Unknown
            )
    }

    /// Interpretation
    pub fn interpretation(&self) -> String {
        if self.is_untemplated_entity() {
            "ENTITY WITHOUT TEMPLATE: Agent-ness without familiar form (machine elf territory)"
                .to_string()
        } else if self.is_face {
            "FACE DETECTED: Strong FFA activation".to_string()
        } else if self.is_entity {
            format!("ENTITY DETECTED: {:?} template", self.template)
        } else {
            format!("NON-ENTITY: {:?}", self.template)
        }
    }
}
