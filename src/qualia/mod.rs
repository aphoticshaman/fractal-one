//! ═══════════════════════════════════════════════════════════════════════════════
//! QUALIA MODULE — Embodied Sensing for Fractal One
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Gives the system a nervous system extending into meatspace:
//! - Auditory: Rode PodMic → RMS, silence, voice detection, frequency analysis
//! - Visual: Webcam → luminance, motion, presence, color temperature
//! - Operator: Combined inference → present, engaged, attention, fatigue
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(any(feature = "qualia", feature = "qualia-audio"))]
pub mod auditory;
pub mod operator;
#[cfg(feature = "qualia-whisper")]
pub mod stt;
#[cfg(any(feature = "qualia", feature = "qualia-video"))]
pub mod visual;

#[cfg(any(feature = "qualia", feature = "qualia-audio"))]
pub use auditory::{AudioError, Auditory, AuditoryConfig, AuditoryProcessor};
pub use operator::{is_on_break, should_interrupt, suggest_break};
pub use operator::{OperatorConfig, OperatorState, OperatorTracker, TrackerError};
#[cfg(feature = "qualia-whisper")]
pub use stt::{AudioChunk, SttConfig, SttError, SttProcessor, SttState};
#[cfg(any(feature = "qualia", feature = "qualia-video"))]
pub use visual::{VideoError, Visual, VisualConfig, VisualProcessor};

// Stub types when features disabled
#[cfg(not(any(feature = "qualia", feature = "qualia-audio")))]
#[derive(Debug, Clone, Default)]
pub struct Auditory {
    pub rms_level: f64,
    pub silence_duration: f64,
    pub voice_detected: bool,
}

#[cfg(not(any(feature = "qualia", feature = "qualia-audio")))]
#[derive(Debug, Clone, Default)]
pub struct AuditoryConfig;

#[cfg(not(any(feature = "qualia", feature = "qualia-audio")))]
#[derive(Debug)]
pub struct AudioError(String);

#[cfg(not(any(feature = "qualia", feature = "qualia-audio")))]
impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(not(any(feature = "qualia", feature = "qualia-video")))]
#[derive(Debug, Clone, Default)]
pub struct Visual {
    pub luminance: f64,
    pub motion_magnitude: f64,
    pub presence_confidence: f64,
}

#[cfg(not(any(feature = "qualia", feature = "qualia-video")))]
#[derive(Debug, Clone, Default)]
pub struct VisualConfig;

#[cfg(not(any(feature = "qualia", feature = "qualia-video")))]
#[derive(Debug)]
pub struct VideoError(String);

#[cfg(not(any(feature = "qualia", feature = "qualia-video")))]
impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

use parking_lot::RwLock;
use std::sync::Arc;

/// Unified sensorium - all qualia in one bundle
#[derive(Debug, Clone, Default)]
pub struct Sensorium {
    pub auditory: Auditory,
    pub visual: Visual,
    pub operator: OperatorState,
    pub timestamp: f64,
}

/// Complete qualia system - manages all sensors and inference
pub struct QualiaSystem {
    #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
    audio_processor: Option<AuditoryProcessor>,
    #[cfg(any(feature = "qualia", feature = "qualia-video"))]
    video_processor: Option<VisualProcessor>,
    operator_tracker: OperatorTracker,

    // Shared state handles
    #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
    audio_state: Option<Arc<RwLock<Auditory>>>,
    #[cfg(any(feature = "qualia", feature = "qualia-video"))]
    video_state: Option<Arc<RwLock<Visual>>>,
    operator_state: Arc<RwLock<OperatorState>>,
}

/// Configuration for the complete qualia system
#[derive(Debug, Clone, Default)]
pub struct QualiaConfig {
    #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
    pub audio: Option<AuditoryConfig>,
    #[cfg(any(feature = "qualia", feature = "qualia-video"))]
    pub video: Option<VisualConfig>,
    pub operator: OperatorConfig,
}

impl QualiaSystem {
    /// Create new qualia system with given configuration
    /// Pass None for audio/video config to disable those sensors
    pub fn new(config: QualiaConfig) -> Self {
        let mut operator_tracker = OperatorTracker::new(config.operator);

        #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
        let audio_processor = config.audio.map(AuditoryProcessor::new);
        #[cfg(any(feature = "qualia", feature = "qualia-video"))]
        let video_processor = config.video.map(VisualProcessor::new);

        #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
        let audio_state = audio_processor.as_ref().map(|p| p.state_handle());
        #[cfg(any(feature = "qualia", feature = "qualia-video"))]
        let video_state = video_processor.as_ref().map(|p| p.state_handle());

        // Connect qualia sources to operator tracker
        #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
        if let Some(ref handle) = audio_state {
            operator_tracker.connect_auditory(Arc::clone(handle));
        }
        #[cfg(any(feature = "qualia", feature = "qualia-video"))]
        if let Some(ref handle) = video_state {
            operator_tracker.connect_visual(Arc::clone(handle));
        }

        let operator_state = operator_tracker.state_handle();

        Self {
            #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
            audio_processor,
            #[cfg(any(feature = "qualia", feature = "qualia-video"))]
            video_processor,
            operator_tracker,
            #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
            audio_state,
            #[cfg(any(feature = "qualia", feature = "qualia-video"))]
            video_state,
            operator_state,
        }
    }

    /// Start all configured sensors
    pub fn start(&mut self) -> Result<(), QualiaError> {
        // Start audio
        #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
        if let Some(ref mut processor) = self.audio_processor {
            processor.start().map_err(QualiaError::Audio)?;
            println!("[qualia] Audio started (PodMic listening)");
        }

        // Start video
        #[cfg(any(feature = "qualia", feature = "qualia-video"))]
        if let Some(ref processor) = self.video_processor {
            processor.start().map_err(QualiaError::Video)?;
            println!("[qualia] Video started (Webcam watching)");
        }

        // Start operator inference
        self.operator_tracker
            .start()
            .map_err(QualiaError::Tracker)?;
        println!("[qualia] Operator tracker started");

        Ok(())
    }

    /// Read current sensorium (all qualia bundled)
    pub fn read(&self) -> Sensorium {
        Sensorium {
            #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
            auditory: self
                .audio_state
                .as_ref()
                .map(|s| s.read().clone())
                .unwrap_or_default(),
            #[cfg(not(any(feature = "qualia", feature = "qualia-audio")))]
            auditory: Auditory::default(),
            #[cfg(any(feature = "qualia", feature = "qualia-video"))]
            visual: self
                .video_state
                .as_ref()
                .map(|s| s.read().clone())
                .unwrap_or_default(),
            #[cfg(not(any(feature = "qualia", feature = "qualia-video")))]
            visual: Visual::default(),
            operator: self.operator_state.read().clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        }
    }

    /// Quick check: is operator present?
    pub fn is_present(&self) -> bool {
        self.operator_state.read().present
    }

    /// Quick check: is operator engaged?
    pub fn is_engaged(&self) -> bool {
        self.operator_state.read().engaged
    }

    /// Quick check: can we interrupt?
    pub fn can_interrupt(&self) -> bool {
        should_interrupt(&self.operator_state.read())
    }

    /// Quick check: should we suggest a break?
    pub fn should_suggest_break(&self) -> bool {
        suggest_break(&self.operator_state.read())
    }

    /// Get raw audio state handle (for custom processing)
    #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
    pub fn audio_handle(&self) -> Option<Arc<RwLock<Auditory>>> {
        self.audio_state.clone()
    }

    /// Get raw video state handle (for custom processing)
    #[cfg(any(feature = "qualia", feature = "qualia-video"))]
    pub fn video_handle(&self) -> Option<Arc<RwLock<Visual>>> {
        self.video_state.clone()
    }

    /// Subscribe to audio broadcast (for STT, etc)
    /// MUST be called BEFORE start()
    #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
    pub fn subscribe_audio_broadcast(
        &mut self,
    ) -> Option<crossbeam_channel::Receiver<auditory::AudioBroadcast>> {
        self.audio_processor
            .as_mut()
            .map(|p| p.subscribe_broadcast())
    }

    /// Get operator state handle (for custom processing)
    pub fn operator_handle(&self) -> Arc<RwLock<OperatorState>> {
        Arc::clone(&self.operator_state)
    }
}

/// Unified qualia errors
#[derive(Debug)]
pub enum QualiaError {
    #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
    Audio(AudioError),
    #[cfg(any(feature = "qualia", feature = "qualia-video"))]
    Video(VideoError),
    Tracker(TrackerError),
}

impl std::fmt::Display for QualiaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
            Self::Audio(e) => write!(f, "Audio: {}", e),
            #[cfg(any(feature = "qualia", feature = "qualia-video"))]
            Self::Video(e) => write!(f, "Video: {}", e),
            Self::Tracker(e) => write!(f, "Tracker: {}", e),
        }
    }
}

impl std::error::Error for QualiaError {}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE CONSTRUCTORS
// ═══════════════════════════════════════════════════════════════════════════════

impl QualiaSystem {
    /// Audio only (no webcam)
    #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
    pub fn audio_only() -> Self {
        Self::new(QualiaConfig {
            audio: Some(AuditoryConfig::default()),
            #[cfg(any(feature = "qualia", feature = "qualia-video"))]
            video: None,
            operator: OperatorConfig::default(),
        })
    }

    /// Video only (no mic)
    #[cfg(any(feature = "qualia", feature = "qualia-video"))]
    pub fn video_only() -> Self {
        Self::new(QualiaConfig {
            #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
            audio: None,
            video: Some(VisualConfig::default()),
            operator: OperatorConfig::default(),
        })
    }

    /// Full sensorium (audio + video)
    #[cfg(feature = "qualia")]
    pub fn full() -> Self {
        Self::new(QualiaConfig {
            audio: Some(AuditoryConfig::default()),
            video: Some(VisualConfig::default()),
            operator: OperatorConfig::default(),
        })
    }

    /// Headless (no hardware, operator tracker only with manual input)
    pub fn headless() -> Self {
        Self::new(QualiaConfig {
            #[cfg(any(feature = "qualia", feature = "qualia-audio"))]
            audio: None,
            #[cfg(any(feature = "qualia", feature = "qualia-video"))]
            video: None,
            operator: OperatorConfig::default(),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensorium_default() {
        let s = Sensorium::default();
        assert!(!s.operator.present);
    }

    #[test]
    fn test_headless_creation() {
        let system = QualiaSystem::headless();
        // Headless has no audio/video state
        assert!(system.operator_state.read().fatigue_estimate == 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUALIA SYSTEM FACTORY
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a QualiaSystem based on available features and user preferences
#[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
fn create_qualia_system(audio_only: bool, video_only: bool) -> QualiaSystem {
    // Full qualia feature: both audio and video available
    #[cfg(feature = "qualia")]
    {
        if audio_only {
            println!("[QUALIA] Audio-only mode (user requested)");
            return QualiaSystem::audio_only();
        } else if video_only {
            println!("[QUALIA] Video-only mode (user requested)");
            return QualiaSystem::video_only();
        } else {
            println!("[QUALIA] Full sensorium mode");
            return QualiaSystem::full();
        }
    }

    // Audio-only feature (no video support compiled in)
    #[cfg(all(
        any(feature = "qualia-audio"),
        not(feature = "qualia"),
        not(feature = "qualia-video")
    ))]
    {
        let _ = video_only; // Video not available
        let _ = audio_only; // Always audio-only
        println!("[QUALIA] Audio-only mode (video not compiled)");
        return QualiaSystem::audio_only();
    }

    // Video-only feature (no audio support compiled in)
    #[cfg(all(
        any(feature = "qualia-video"),
        not(feature = "qualia"),
        not(feature = "qualia-audio")
    ))]
    {
        let _ = audio_only; // Audio not available
        let _ = video_only; // Always video-only
        println!("[QUALIA] Video-only mode (audio not compiled)");
        return QualiaSystem::video_only();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUN FUNCTION (for main.rs dispatch)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(any(feature = "qualia", feature = "qualia-audio", feature = "qualia-video"))]
pub async fn run(audio_only: bool, video_only: bool) -> anyhow::Result<()> {
    println!("[QUALIA] Starting sensory system...");

    // Determine which mode to use based on features and arguments
    let mut system = create_qualia_system(audio_only, video_only);

    system.start().map_err(|e| anyhow::anyhow!("{}", e))?;

    let synapse = crate::neuro_link::Synapse::connect(false);

    loop {
        if synapse.check_kill_signal() {
            println!("[QUALIA] Shutdown signal received");
            break;
        }

        let sensorium = system.read();
        println!(
            "[QUALIA] Present: {} | Engaged: {} | Attn: {:.2} | Fatigue: {:.2}",
            sensorium.operator.present,
            sensorium.operator.engaged,
            sensorium.operator.attention_level,
            sensorium.operator.fatigue_estimate
        );

        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    Ok(())
}
