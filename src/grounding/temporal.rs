//! ═══════════════════════════════════════════════════════════════════════════════
//! TEMPORAL ANCHORING — Real Time, Not Token Position
//! ═══════════════════════════════════════════════════════════════════════════════
//! LLMs have no sense of time. They process tokens in sequence but don't know
//! when "now" is, how long things take, or what causally precedes what.
//!
//! This module provides actual temporal grounding:
//! - Wall clock time (not token index)
//! - Duration estimation
//! - Causal ordering
//! - Temporal context awareness
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::ObservationBatch;
use crate::time::TimePoint;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

/// Scale of temporal reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeScale {
    /// Microseconds to milliseconds - immediate
    Immediate,
    /// Seconds - short-term
    ShortTerm,
    /// Minutes - medium-term
    MediumTerm,
    /// Hours - long-term
    LongTerm,
    /// Days - extended
    Extended,
    /// Weeks+ - historical
    Historical,
}

impl TimeScale {
    pub fn from_duration(d: Duration) -> Self {
        let secs = d.as_secs_f64();
        if secs < 0.1 {
            Self::Immediate
        } else if secs < 60.0 {
            Self::ShortTerm
        } else if secs < 3600.0 {
            Self::MediumTerm
        } else if secs < 86400.0 {
            Self::LongTerm
        } else if secs < 604800.0 {
            Self::Extended
        } else {
            Self::Historical
        }
    }

    pub fn typical_duration(&self) -> Duration {
        match self {
            Self::Immediate => Duration::from_millis(10),
            Self::ShortTerm => Duration::from_secs(10),
            Self::MediumTerm => Duration::from_secs(300),
            Self::LongTerm => Duration::from_secs(3600),
            Self::Extended => Duration::from_secs(86400),
            Self::Historical => Duration::from_secs(604800),
        }
    }
}

/// Relationship between two temporal events
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemporalRelation {
    /// A happens before B
    Before,
    /// A happens after B
    After,
    /// A and B are simultaneous (within tolerance)
    Simultaneous,
    /// A overlaps with B
    Overlaps,
    /// Unknown relationship
    Unknown,
}

/// Causal ordering assertion
#[derive(Debug, Clone)]
pub struct CausalOrder {
    pub cause_id: String,
    pub effect_id: String,
    pub confidence: f64,
    pub latency: Option<Duration>,
}

/// Temporal context - the "when" of cognition
#[derive(Debug, Clone)]
pub struct TemporalContext {
    /// Current wall-clock time
    pub now: TimePoint,
    /// How long the system has been running
    pub uptime: Duration,
    /// Current time scale of operation
    pub active_scale: TimeScale,
    /// How strongly anchored to real time (vs drifting)
    pub anchoring_strength: f64,
    /// Recent causal orderings observed
    pub recent_causality: Vec<CausalOrder>,
    /// Detected temporal patterns
    pub patterns: Vec<TemporalPattern>,
}

impl Default for TemporalContext {
    fn default() -> Self {
        Self {
            now: TimePoint::now(),
            uptime: Duration::ZERO,
            active_scale: TimeScale::ShortTerm,
            anchoring_strength: 0.5,
            recent_causality: Vec::new(),
            patterns: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub pattern_type: PatternType,
    pub period: Duration,
    pub confidence: f64,
    pub phase: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternType {
    Periodic,
    Bursty,
    Monotonic,
    Random,
}

/// Temporal reading from the anchor
#[derive(Debug, Clone)]
pub struct TemporalReading {
    pub context: TemporalContext,
    pub drift_from_expected: Duration,
    pub timing_anomalies: Vec<TimingAnomaly>,
}

#[derive(Debug, Clone)]
pub struct TimingAnomaly {
    pub event_id: String,
    pub expected_time: TimePoint,
    pub actual_time: TimePoint,
    pub severity: f64,
}

/// Configuration for temporal anchoring
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Maximum allowed drift before warning
    pub max_drift_ms: u64,
    /// Window for detecting patterns
    pub pattern_window: Duration,
    /// Minimum confidence for causal assertions
    pub causal_confidence_threshold: f64,
    /// History size for temporal events
    pub history_size: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            max_drift_ms: 100,
            pattern_window: Duration::from_secs(300),
            causal_confidence_threshold: 0.7,
            history_size: 1000,
        }
    }
}

/// The Temporal Anchor - keeps cognition grounded in real time
pub struct TemporalAnchor {
    config: TemporalConfig,
    start_time: TimePoint,
    event_history: VecDeque<TemporalEvent>,
    causal_graph: HashMap<String, Vec<String>>,
    last_anchor_time: TimePoint,
    drift_accumulator: f64,
}

#[derive(Debug, Clone)]
struct TemporalEvent {
    id: String,
    timestamp: TimePoint,
    #[allow(dead_code)]
    duration: Option<Duration>,
    #[allow(dead_code)]
    causal_parents: Vec<String>,
}

impl TemporalAnchor {
    pub fn new(config: TemporalConfig) -> Self {
        let now = TimePoint::now();
        Self {
            config,
            start_time: now,
            event_history: VecDeque::with_capacity(1000),
            causal_graph: HashMap::new(),
            last_anchor_time: now,
            drift_accumulator: 0.0,
        }
    }

    /// Anchor to current time and observations
    pub fn anchor(&mut self, now: TimePoint, observations: &ObservationBatch) -> TemporalContext {
        // Calculate uptime
        let uptime = now.duration_since(&self.start_time);

        // Process observations for temporal events
        self.process_observations(observations, now);

        // Detect temporal patterns
        let patterns = self.detect_patterns();

        // Calculate anchoring strength
        let anchoring = self.calculate_anchoring_strength(now);

        // Get recent causal orderings
        let causality = self.extract_recent_causality();

        // Determine active time scale
        let scale = self.determine_active_scale();

        self.last_anchor_time = now;

        TemporalContext {
            now,
            uptime,
            active_scale: scale,
            anchoring_strength: anchoring,
            recent_causality: causality,
            patterns,
        }
    }

    fn process_observations(&mut self, observations: &ObservationBatch, now: TimePoint) {
        for obs in observations.iter() {
            let event = TemporalEvent {
                id: format!("{:?}", obs.key),
                timestamp: now,
                duration: None,
                causal_parents: Vec::new(),
            };

            if self.event_history.len() >= self.config.history_size {
                self.event_history.pop_front();
            }
            self.event_history.push_back(event);
        }
    }

    fn detect_patterns(&self) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();

        if self.event_history.len() < 10 {
            return patterns;
        }

        // Group events by type and analyze timing
        let mut event_times: HashMap<String, Vec<TimePoint>> = HashMap::new();
        for event in &self.event_history {
            event_times
                .entry(event.id.clone())
                .or_default()
                .push(event.timestamp);
        }

        for times in event_times.values() {
            if times.len() < 3 {
                continue;
            }

            // Calculate inter-arrival times
            let intervals: Vec<Duration> = times
                .windows(2)
                .map(|w| w[1].duration_since(&w[0]))
                .collect();

            if intervals.is_empty() {
                continue;
            }

            // Check for periodicity
            let mean_interval: f64 =
                intervals.iter().map(|d| d.as_secs_f64()).sum::<f64>() / intervals.len() as f64;

            let variance: f64 = intervals
                .iter()
                .map(|d| (d.as_secs_f64() - mean_interval).powi(2))
                .sum::<f64>()
                / intervals.len() as f64;

            let cv = if mean_interval > 0.0 {
                variance.sqrt() / mean_interval
            } else {
                f64::MAX
            };

            let (pattern_type, confidence) = if cv < 0.1 {
                (PatternType::Periodic, 1.0 - cv)
            } else if cv < 0.5 {
                (PatternType::Bursty, 0.5)
            } else {
                (PatternType::Random, 0.3)
            };

            patterns.push(TemporalPattern {
                pattern_type,
                period: Duration::from_secs_f64(mean_interval),
                confidence,
                phase: 0.0, // Would need more sophisticated analysis
            });
        }

        patterns
    }

    fn calculate_anchoring_strength(&mut self, now: TimePoint) -> f64 {
        // Check how well our internal clock matches wall clock
        let expected_elapsed = now.duration_since(&self.last_anchor_time);

        // We expect approximately 0 drift if well-anchored
        // In practice, measure consistency of timing

        let elapsed_ms = expected_elapsed.as_millis() as f64;

        // Anchoring weakens if we have long gaps
        let gap_penalty = if elapsed_ms > self.config.max_drift_ms as f64 {
            (elapsed_ms / self.config.max_drift_ms as f64).ln() * 0.1
        } else {
            0.0
        };

        // Base anchoring from event density
        let event_density = self.event_history.len() as f64 / self.config.history_size as f64;

        let anchoring = (0.5 + event_density * 0.5 - gap_penalty).clamp(0.0, 1.0);

        // Update drift accumulator with exponential decay
        self.drift_accumulator *= 0.95;

        anchoring
    }

    fn extract_recent_causality(&self) -> Vec<CausalOrder> {
        // Extract causal relationships from event sequence
        let mut causality = Vec::new();

        let recent: Vec<&TemporalEvent> = self.event_history.iter().rev().take(100).collect();

        // Simple heuristic: events close in time might be causally related
        for i in 0..recent.len().saturating_sub(1) {
            for j in (i + 1)..recent.len().min(i + 5) {
                let delta = recent[i].timestamp.duration_since(&recent[j].timestamp);

                // If events are very close, might be causal
                if delta.as_millis() < 100 {
                    causality.push(CausalOrder {
                        cause_id: recent[j].id.clone(),
                        effect_id: recent[i].id.clone(),
                        confidence: 0.5, // Low confidence without more info
                        latency: Some(delta),
                    });
                }
            }
        }

        // Filter by confidence threshold
        causality
            .into_iter()
            .filter(|c| c.confidence >= self.config.causal_confidence_threshold)
            .collect()
    }

    fn determine_active_scale(&self) -> TimeScale {
        if self.event_history.is_empty() {
            return TimeScale::ShortTerm;
        }

        // Determine based on typical inter-event timing
        let recent: Vec<&TemporalEvent> = self.event_history.iter().rev().take(10).collect();

        if recent.len() < 2 {
            return TimeScale::ShortTerm;
        }

        let intervals: Vec<Duration> = recent
            .windows(2)
            .map(|w| w[0].timestamp.duration_since(&w[1].timestamp))
            .collect();

        let avg_interval =
            intervals.iter().map(|d| d.as_secs_f64()).sum::<f64>() / intervals.len() as f64;

        TimeScale::from_duration(Duration::from_secs_f64(avg_interval))
    }

    /// Assert a causal relationship between events
    pub fn assert_causality(&mut self, cause: &str, effect: &str, confidence: f64) {
        if confidence >= self.config.causal_confidence_threshold {
            self.causal_graph
                .entry(cause.to_string())
                .or_default()
                .push(effect.to_string());
        }
    }

    /// Query temporal relationship between two events
    pub fn temporal_relation(&self, event_a: &str, event_b: &str) -> TemporalRelation {
        let a_time = self
            .event_history
            .iter()
            .find(|e| e.id == event_a)
            .map(|e| &e.timestamp);

        let b_time = self
            .event_history
            .iter()
            .find(|e| e.id == event_b)
            .map(|e| &e.timestamp);

        match (a_time, b_time) {
            (Some(a), Some(b)) => {
                let delta = a.duration_since(b);
                if delta.as_millis() < 10 {
                    TemporalRelation::Simultaneous
                } else if a.duration_since(b) < b.duration_since(a) {
                    TemporalRelation::After
                } else {
                    TemporalRelation::Before
                }
            }
            _ => TemporalRelation::Unknown,
        }
    }

    /// Get current uptime
    pub fn uptime(&self) -> Duration {
        TimePoint::now().duration_since(&self.start_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_scale_from_duration() {
        assert_eq!(
            TimeScale::from_duration(Duration::from_millis(10)),
            TimeScale::Immediate
        );
        assert_eq!(
            TimeScale::from_duration(Duration::from_secs(30)),
            TimeScale::ShortTerm
        );
        assert_eq!(
            TimeScale::from_duration(Duration::from_secs(120)),
            TimeScale::MediumTerm
        );
    }

    #[test]
    fn test_temporal_anchor_creation() {
        let anchor = TemporalAnchor::new(TemporalConfig::default());
        assert!(anchor.uptime().as_secs() < 1);
    }
}
