//! ═══════════════════════════════════════════════════════════════════════════════
//! OPERATOR STATE — Combined Human Presence & Engagement Inference
//! ═══════════════════════════════════════════════════════════════════════════════
//! Fuses auditory and visual qualia to answer:
//! - Is the operator present?
//! - Is the operator engaged?
//! - Should the system interrupt?
//! ═══════════════════════════════════════════════════════════════════════════════

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Use types from parent module (handles feature gating)
use super::{Auditory, Visual};

/// Operator state - inferred from combined qualia
#[derive(Debug, Clone, Default)]
pub struct OperatorState {
    /// Is the operator physically present? (high confidence)
    pub present: bool,

    /// Is the operator actively engaged? (interacting vs passive)
    pub engaged: bool,

    /// Attention level (0.0 = AFK, 1.0 = locked in)
    pub attention_level: f64,

    /// Estimated fatigue (0.0 = fresh, 1.0 = exhausted) - accumulates over session
    pub fatigue_estimate: f64,

    /// Seconds since session started (presence first detected)
    pub session_duration: f64,

    /// Seconds since last engagement signal
    pub idle_duration: f64,

    /// Should system initiate interaction? (gating flag)
    pub interruptible: bool,

    /// Confidence in presence detection (0.0-1.0)
    pub presence_confidence: f64,

    /// Timestamp of last update
    pub timestamp: f64,
}

/// Configuration for operator state inference
#[derive(Debug, Clone)]
pub struct OperatorConfig {
    /// Presence threshold (audio silence OR visual presence)
    pub presence_audio_silence_max: f64, // seconds
    pub presence_visual_threshold: f64, // 0.0-1.0

    /// Engagement detection
    pub engagement_voice_required: bool,
    pub engagement_motion_threshold: f64,

    /// Fatigue accumulation
    pub fatigue_rate_per_hour: f64, // fatigue gained per hour
    pub fatigue_recovery_per_minute_afk: f64, // fatigue lost per minute AFK

    /// Interruptibility
    pub interrupt_min_attention: f64, // min attention to allow interrupt
    pub interrupt_min_presence_seconds: f64, // must be present this long

    /// Idle threshold
    pub idle_threshold_seconds: f64, // seconds without engagement = idle
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            presence_audio_silence_max: 120.0, // 2 minutes silence = maybe AFK
            presence_visual_threshold: 0.3,
            engagement_voice_required: false,
            engagement_motion_threshold: 0.05,
            fatigue_rate_per_hour: 0.05, // ~20 hours to full fatigue
            fatigue_recovery_per_minute_afk: 0.01, // 100 min AFK = full recovery
            interrupt_min_attention: 0.3,
            interrupt_min_presence_seconds: 10.0,
            idle_threshold_seconds: 30.0,
        }
    }
}

/// Operator state tracker - runs inference loop
pub struct OperatorTracker {
    config: OperatorConfig,
    state: Arc<RwLock<OperatorState>>,

    // External qualia handles
    auditory: Option<Arc<RwLock<Auditory>>>,
    visual: Option<Arc<RwLock<Visual>>>,

    // Internal tracking
    session_start: Option<Instant>,
    last_engagement: Instant,
    accumulated_fatigue: f64,
}

impl OperatorTracker {
    /// Create new operator tracker
    pub fn new(config: OperatorConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(OperatorState::default())),
            auditory: None,
            visual: None,
            session_start: None,
            last_engagement: Instant::now(),
            accumulated_fatigue: 0.0,
        }
    }

    /// Get shared state handle
    pub fn state_handle(&self) -> Arc<RwLock<OperatorState>> {
        Arc::clone(&self.state)
    }

    /// Connect auditory qualia source
    pub fn connect_auditory(&mut self, handle: Arc<RwLock<Auditory>>) {
        self.auditory = Some(handle);
    }

    /// Connect visual qualia source
    pub fn connect_visual(&mut self, handle: Arc<RwLock<Visual>>) {
        self.visual = Some(handle);
    }

    /// Start inference loop
    pub fn start(&self) -> Result<(), TrackerError> {
        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let auditory = self.auditory.clone();
        let visual = self.visual.clone();

        std::thread::Builder::new()
            .name("operator-tracker".into())
            .spawn(move || {
                Self::inference_loop(state, config, auditory, visual);
            })
            .map_err(|e| TrackerError::ThreadError(e.to_string()))?;

        Ok(())
    }

    /// Main inference loop
    fn inference_loop(
        state: Arc<RwLock<OperatorState>>,
        config: OperatorConfig,
        auditory: Option<Arc<RwLock<Auditory>>>,
        visual: Option<Arc<RwLock<Visual>>>,
    ) {
        let mut session_start: Option<Instant> = None;
        let mut last_engagement = Instant::now();
        let mut accumulated_fatigue = 0.0_f64;
        let mut last_update = Instant::now();

        loop {
            std::thread::sleep(Duration::from_millis(100)); // 10 Hz update

            let dt = last_update.elapsed().as_secs_f64();
            last_update = Instant::now();

            // ══════════════════════════════════════════════════════════════
            // READ QUALIA
            // ══════════════════════════════════════════════════════════════
            let audio = auditory.as_ref().map(|a| a.read().clone());
            let video = visual.as_ref().map(|v| v.read().clone());

            // ══════════════════════════════════════════════════════════════
            // PRESENCE DETECTION
            // ══════════════════════════════════════════════════════════════
            let audio_suggests_present = audio
                .as_ref()
                .map(|a| a.silence_duration < config.presence_audio_silence_max)
                .unwrap_or(false);

            let visual_suggests_present = video
                .as_ref()
                .map(|v| v.presence_confidence > config.presence_visual_threshold)
                .unwrap_or(false);

            // Either modality can confirm presence
            let present = audio_suggests_present || visual_suggests_present;

            // Combined confidence
            let presence_confidence = {
                let audio_conf = audio
                    .as_ref()
                    .map(|a| {
                        1.0 - (a.silence_duration / config.presence_audio_silence_max).min(1.0)
                    })
                    .unwrap_or(0.0);
                let visual_conf = video.as_ref().map(|v| v.presence_confidence).unwrap_or(0.0);
                // Max of both (either can confirm)
                audio_conf.max(visual_conf)
            };

            // ══════════════════════════════════════════════════════════════
            // SESSION TRACKING
            // ══════════════════════════════════════════════════════════════
            if present && session_start.is_none() {
                session_start = Some(Instant::now());
            } else if !present {
                // Only reset session after extended absence
                if session_start.is_some() {
                    let absence_duration =
                        audio.as_ref().map(|a| a.silence_duration).unwrap_or(0.0);
                    if absence_duration > 300.0 {
                        // 5 min absence = session end
                        session_start = None;
                        accumulated_fatigue = 0.0; // Reset fatigue on session end
                    }
                }
            }

            let session_duration = session_start
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);

            // ══════════════════════════════════════════════════════════════
            // ENGAGEMENT DETECTION
            // ══════════════════════════════════════════════════════════════
            let voice_engaged = audio.as_ref().map(|a| a.voice_detected).unwrap_or(false);

            let motion_engaged = video
                .as_ref()
                .map(|v| v.motion_magnitude > config.engagement_motion_threshold)
                .unwrap_or(false);

            let engaged = voice_engaged || motion_engaged;

            if engaged {
                last_engagement = Instant::now();
            }

            let idle_duration = last_engagement.elapsed().as_secs_f64();

            // ══════════════════════════════════════════════════════════════
            // ATTENTION LEVEL
            // ══════════════════════════════════════════════════════════════
            let attention_level = if !present {
                0.0
            } else {
                // Base attention from presence
                let base = presence_confidence * 0.5;

                // Boost from recent engagement
                let engagement_boost = if idle_duration < 5.0 {
                    0.5
                } else if idle_duration < 30.0 {
                    0.3 * (1.0 - idle_duration / 30.0)
                } else {
                    0.0
                };

                // Voice activity is strong signal
                let voice_boost = if voice_engaged { 0.2 } else { 0.0 };

                (base + engagement_boost + voice_boost).min(1.0)
            };

            // ══════════════════════════════════════════════════════════════
            // FATIGUE ESTIMATION
            // ══════════════════════════════════════════════════════════════
            if present && engaged {
                // Accumulate fatigue while working
                accumulated_fatigue += config.fatigue_rate_per_hour * dt / 3600.0;
            } else if !present {
                // Recover fatigue while AFK
                accumulated_fatigue -= config.fatigue_recovery_per_minute_afk * dt / 60.0;
            }
            accumulated_fatigue = accumulated_fatigue.clamp(0.0, 1.0);

            // ══════════════════════════════════════════════════════════════
            // INTERRUPTIBILITY
            // ══════════════════════════════════════════════════════════════
            let interruptible = present
                && attention_level >= config.interrupt_min_attention
                && session_duration >= config.interrupt_min_presence_seconds
                && idle_duration < config.idle_threshold_seconds;

            // ══════════════════════════════════════════════════════════════
            // UPDATE STATE
            // ══════════════════════════════════════════════════════════════
            {
                let mut state = state.write();
                state.present = present;
                state.engaged = engaged;
                state.attention_level = attention_level;
                state.fatigue_estimate = accumulated_fatigue;
                state.session_duration = session_duration;
                state.idle_duration = idle_duration;
                state.interruptible = interruptible;
                state.presence_confidence = presence_confidence;
                state.timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
            }
        }
    }

    /// Get current state snapshot
    pub fn read(&self) -> OperatorState {
        self.state.read().clone()
    }

    /// Get session start time (if session is active)
    pub fn session_start(&self) -> Option<Instant> {
        self.session_start
    }

    /// Get time of last engagement
    pub fn last_engagement(&self) -> Instant {
        self.last_engagement
    }

    /// Get accumulated fatigue (local tracker copy, may differ from state)
    pub fn accumulated_fatigue(&self) -> f64 {
        self.accumulated_fatigue
    }

    /// Get current session duration
    pub fn session_duration(&self) -> std::time::Duration {
        self.session_start.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// Get time since last engagement
    pub fn idle_duration(&self) -> std::time::Duration {
        self.last_engagement.elapsed()
    }

    /// Check if tracker has auditory source connected
    pub fn has_auditory(&self) -> bool {
        self.auditory.is_some()
    }

    /// Check if tracker has visual source connected
    pub fn has_visual(&self) -> bool {
        self.visual.is_some()
    }

    /// Get diagnostic info
    pub fn diagnostic_info(&self) -> String {
        format!(
            "OperatorTracker: session={}, idle={:.1}s, fatigue={:.2}, auditory={}, visual={}",
            self.session_start.map(|_| "active").unwrap_or("inactive"),
            self.last_engagement.elapsed().as_secs_f64(),
            self.accumulated_fatigue,
            self.auditory.is_some(),
            self.visual.is_some()
        )
    }
}

/// Tracker errors
#[derive(Debug)]
pub enum TrackerError {
    ThreadError(String),
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThreadError(e) => write!(f, "Tracker thread error: {}", e),
        }
    }
}

impl std::error::Error for TrackerError {}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Quick check: should the system bother the operator right now?
pub fn should_interrupt(state: &OperatorState) -> bool {
    state.interruptible && state.attention_level > 0.5
}

/// Quick check: is the operator likely taking a break?
pub fn is_on_break(state: &OperatorState) -> bool {
    !state.present || state.idle_duration > 60.0
}

/// Quick check: should the system suggest a break?
pub fn suggest_break(state: &OperatorState) -> bool {
    state.present
        && state.session_duration > 3600.0 // 1+ hour session
        && state.fatigue_estimate > 0.5 // moderate fatigue
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_state_default() {
        let s = OperatorState::default();
        assert!(!s.present);
        assert!(!s.engaged);
        assert_eq!(s.attention_level, 0.0);
    }

    #[test]
    fn test_should_interrupt() {
        let mut s = OperatorState::default();
        assert!(!should_interrupt(&s));

        s.interruptible = true;
        s.attention_level = 0.8;
        assert!(should_interrupt(&s));
    }

    #[test]
    fn test_suggest_break() {
        let mut s = OperatorState::default();
        s.present = true;
        s.session_duration = 7200.0; // 2 hours
        s.fatigue_estimate = 0.7;
        assert!(suggest_break(&s));
    }
}
