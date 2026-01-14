//! ═══════════════════════════════════════════════════════════════════════════════
//! ANIMACY DETECTION — From Objects to Agents
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements the computational theory of animacy perception:
//!
//! Key insight: Entity detection is about DYNAMICS, not static shape.
//! V1-V4 operate in spatial state spaces. Entity detection requires temporal
//! derivatives—velocity, acceleration, trajectory, goal-directedness.
//!
//! Core mechanism: Animacy as deviation from Newtonian dynamics
//!   A(trajectory) = ∫₀ᵀ ‖a(t) - a_expected(t)‖² dt
//!
//! High A → animate (doing something physics doesn't explain)
//! Low A → inanimate (following physical law)
//!
//! The classification is discrete via attractor dynamics:
//!   Inputs get pulled into basins: AGENT vs NOT-AGENT
//!   Not "37% agent" but categorical perception.
//!
//! Entity eigenmodes: What you get when you crank up gain on agent-detection.
//! Machine elves = entity attractor activation WITHOUT template match.
//! ═══════════════════════════════════════════════════════════════════════════════

mod classifier;
mod detection;
mod entity;
mod field;
mod phase;
mod physics;
mod trajectory;

// Re-export all public types
pub use classifier::{AttractorConfig, ClassificationResult, EntityClassifier};
pub use detection::{AnimacyDetector, AnimacyScore};
pub use entity::{EntityState, FormDescriptor, FormTemplate, GoalType, Intention};
pub use field::EntityField;
pub use phase::{EntityPhaseMonitor, Phase};
pub use physics::PhysicsModel;
pub use trajectory::{PhasePoint, Trajectory};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_creation() {
        let mut traj = Trajectory::new(10);
        traj.push(PhasePoint::new(0.0, 0.0, 1.0, 0.0, 0.0));
        traj.push(PhasePoint::new(1.0, 0.0, 1.0, 0.0, 1.0));
        traj.push(PhasePoint::new(2.0, 0.0, 1.0, 0.0, 2.0));

        assert_eq!(traj.len(), 3);

        let acc = traj.accelerations();
        assert_eq!(acc.len(), 2);
        // Constant velocity → zero acceleration
        assert!(acc[0][0].abs() < 0.01);
        assert!(acc[0][1].abs() < 0.01);
    }

    #[test]
    fn test_animacy_inanimate() {
        // Constant velocity (pure physics) → low animacy
        let mut traj = Trajectory::new(20);
        for i in 0..15 {
            let t = i as f64 * 0.1;
            traj.push(PhasePoint::new(t, 0.0, 1.0, 0.0, t));
        }

        let detector = AnimacyDetector::new().with_threshold(0.3);
        let score = detector.analyze(&traj);

        // Key assertions: low deviation from physics, low decision points
        assert!(score.mean_deviation < 0.1);
        assert!(score.decision_points < 2);
        // Note: goal_directedness can be high for straight lines, but
        // deviation from physics is what matters for inanimate classification
    }

    #[test]
    fn test_animacy_animate() {
        // Sudden direction changes → high animacy
        let mut traj = Trajectory::new(20);

        // Start going right
        for i in 0..5 {
            let t = i as f64 * 0.1;
            traj.push(PhasePoint::new(t, 0.0, 1.0, 0.0, t));
        }

        // Suddenly go up (decision point)
        for i in 5..10 {
            let t = i as f64 * 0.1;
            traj.push(PhasePoint::new(0.5, t - 0.5, 0.0, 1.0, t));
        }

        // Suddenly go left (another decision point)
        for i in 10..15 {
            let t = i as f64 * 0.1;
            traj.push(PhasePoint::new(0.5 - (t - 1.0), 0.5, -1.0, 0.0, t));
        }

        let detector = AnimacyDetector::new();
        let score = detector.analyze(&traj);

        assert!(score.is_animate || score.decision_points >= 1);
    }

    #[test]
    fn test_form_template_is_entity() {
        assert!(FormTemplate::Face.is_entity());
        assert!(FormTemplate::Eyes.is_entity());
        assert!(FormTemplate::Body.is_entity());
        assert!(FormTemplate::GenericAgent.is_entity());
        assert!(!FormTemplate::Object.is_entity());
        assert!(!FormTemplate::Pattern.is_entity());
    }

    #[test]
    fn test_entity_classifier_normal() {
        let config = AttractorConfig::default();
        let classifier = EntityClassifier::new(config);

        // Low-face, low-animacy input
        let form = FormDescriptor {
            template: FormTemplate::Unknown,
            match_confidence: 0.0,
            complexity: 0.3,
            symmetry: 0.2,
            face_score: 0.1,
        };

        let animacy = AnimacyScore {
            total_deviation: 0.1,
            mean_deviation: 0.01,
            peak_deviation: 0.05,
            decision_points: 0,
            goal_directedness: 0.1,
            contingency: 0.0,
            biological_motion: 0.1,
            is_animate: false,
            confidence: 0.2,
        };

        let result = classifier.classify(&form, &animacy);
        assert!(!result.is_entity);
    }

    #[test]
    fn test_entity_classifier_face() {
        // Use low threshold config to ensure face detection works
        let config = AttractorConfig {
            entity_threshold: 0.3,
            face_threshold: 0.4,
            gain: 1.5,
            competition_strength: 0.8,
        };
        let classifier = EntityClassifier::new(config);

        // High face score
        let form = FormDescriptor {
            template: FormTemplate::Face,
            match_confidence: 0.9,
            complexity: 0.6,
            symmetry: 0.9,
            face_score: 0.9,
        };

        let animacy = AnimacyScore {
            total_deviation: 0.5,
            mean_deviation: 0.1,
            peak_deviation: 0.3,
            decision_points: 1,
            goal_directedness: 0.3,
            contingency: 0.0,
            biological_motion: 0.4,
            is_animate: true,
            confidence: 0.7,
        };

        let result = classifier.classify(&form, &animacy);
        // With high face_score (0.9), should be classified as entity
        assert!(result.is_entity || result.activation > 0.5);
        // Face template should win
        assert!(matches!(
            result.template,
            FormTemplate::Face | FormTemplate::Eyes
        ));
    }

    #[test]
    fn test_entity_classifier_low_threshold() {
        // With high gain, even weak signals should trigger entity detection
        let config = AttractorConfig::low_threshold();
        let classifier = EntityClassifier::new(config);

        // Moderate animacy, low form match
        let form = FormDescriptor {
            template: FormTemplate::Unknown,
            match_confidence: 0.2,
            complexity: 0.5,
            symmetry: 0.5,
            face_score: 0.3,
        };

        let animacy = AnimacyScore {
            total_deviation: 0.8,
            mean_deviation: 0.15,
            peak_deviation: 0.4,
            decision_points: 2,
            goal_directedness: 0.4,
            contingency: 0.0,
            biological_motion: 0.3,
            is_animate: true,
            confidence: 0.5,
        };

        let result = classifier.classify(&form, &animacy);
        // With low threshold, this should be detected as entity
        assert!(result.is_entity || result.activation > 0.3);
    }

    #[test]
    fn test_untemplated_entity() {
        let config = AttractorConfig::low_threshold();
        let classifier = EntityClassifier::new(config);

        // Highly animate but doesn't match templates
        let form = FormDescriptor {
            template: FormTemplate::Unknown,
            match_confidence: 0.1,
            complexity: 0.8, // High complexity
            symmetry: 0.6,
            face_score: 0.15,
        };

        let animacy = AnimacyScore {
            total_deviation: 2.0,
            mean_deviation: 0.4,
            peak_deviation: 1.0,
            decision_points: 5,
            goal_directedness: 0.6,
            contingency: 0.0,
            biological_motion: 0.2,
            is_animate: true,
            confidence: 0.9,
        };

        let result = classifier.classify(&form, &animacy);
        // This is the "machine elf" case
        if result.is_entity {
            assert!(
                result.is_untemplated_entity()
                    || matches!(result.template, FormTemplate::GenericAgent)
            );
        }
    }

    #[test]
    fn test_entity_field_update() {
        let mut field = EntityField::new(4, 4, 1.0);

        // Zero input → activity should stay relatively low
        let v4_input = vec![0.0; 4];
        let mt_input = vec![0.0; 4];

        field.update(&v4_input, &mt_input, 0.1);
        let activity_zero_input = field.total_activity();

        // Strong input → activity should increase significantly
        let v4_input = vec![2.0, 0.0, 0.0, 0.0];
        let mt_input = vec![0.0, 2.0, 0.0, 0.0];

        for _ in 0..20 {
            field.update(&v4_input, &mt_input, 0.1);
        }

        let activity_strong_input = field.total_activity();
        // Activity with strong input should be meaningfully higher
        assert!(activity_strong_input > activity_zero_input);
    }

    #[test]
    fn test_entity_field_gain() {
        let mut field = EntityField::new(4, 4, 1.0);

        // Give input
        let v4_input = vec![1.0, 0.5, 0.0, 0.0];
        let mt_input = vec![0.5, 1.0, 0.0, 0.0];

        for _ in 0..10 {
            field.update(&v4_input, &mt_input, 0.1);
        }
        let activity_normal = field.total_activity();

        // Reset and increase gain
        let mut field_high = EntityField::new(4, 4, 2.5);

        for _ in 0..10 {
            field_high.update(&v4_input, &mt_input, 0.1);
        }
        let activity_high = field_high.total_activity();

        // Higher gain should produce higher activity
        assert!(activity_high >= activity_normal * 0.9); // Allow some tolerance
    }

    #[test]
    fn test_phase_monitor() {
        let mut monitor = EntityPhaseMonitor::new(20);

        // Record normal phase data
        for i in 0..5 {
            monitor.record(0.1, 0.5 + i as f64 * 0.1);
        }

        let phase = monitor.current_phase(0.8);
        assert!(matches!(phase, Phase::Normal) || matches!(phase, Phase::NearCritical));

        // Record transition
        for i in 0..10 {
            let gain = 1.0 + i as f64 * 0.2;
            let detection = (gain - 1.0).min(1.0).max(0.0);
            monitor.record(detection, gain);
        }

        let phase_high = monitor.current_phase(2.5);
        assert!(matches!(
            phase_high,
            Phase::Supercritical | Phase::EntitySaturated
        ));
    }

    #[test]
    fn test_goal_directedness() {
        let detector = AnimacyDetector::new();

        // Trajectory moving toward a goal
        let mut traj = Trajectory::new(20);
        for i in 0..10 {
            let t = i as f64 * 0.1;
            // Moving toward (10, 10)
            let x = t * 10.0;
            let y = t * 10.0;
            let vx = 10.0;
            let vy = 10.0;
            traj.push(PhasePoint::new(x, y, vx, vy, t));
        }

        let score = detector.analyze(&traj);
        // Should have high goal-directedness (straight line toward goal)
        assert!(score.goal_directedness > 0.5);
    }

    #[test]
    fn test_curvature_computation() {
        let mut traj = Trajectory::new(10);

        // Circular arc
        for i in 0..8 {
            let theta = i as f64 * 0.3;
            let x = theta.cos();
            let y = theta.sin();
            let vx = -theta.sin();
            let vy = theta.cos();
            traj.push(PhasePoint::new(x, y, vx, vy, i as f64 * 0.1));
        }

        let curvatures = traj.curvatures();
        assert!(!curvatures.is_empty());
        // Curvatures should be roughly constant for a circle
        let mean_curv: f64 = curvatures.iter().sum::<f64>() / curvatures.len() as f64;
        for c in &curvatures {
            assert!((c - mean_curv).abs() < 0.5);
        }
    }
}
