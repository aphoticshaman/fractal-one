//! ═══════════════════════════════════════════════════════════════════════════════
//! SPEECH-TO-TEXT — Whisper CLI Integration
//! ═══════════════════════════════════════════════════════════════════════════════
//! Transcribes voice input using OpenAI Whisper CLI (pip install openai-whisper).
//! Accumulates audio during speech, writes WAV, shells out to whisper.
//! ═══════════════════════════════════════════════════════════════════════════════

use crossbeam_channel::{bounded, Receiver, Sender};
use hound::{WavSpec, WavWriter};
use parking_lot::RwLock;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

/// STT state - what was heard
#[derive(Debug, Clone, Default)]
pub struct SttState {
    /// Last transcription
    pub transcript: String,
    /// Is currently listening/accumulating?
    pub listening: bool,
    /// Is currently transcribing?
    pub transcribing: bool,
    /// Seconds of audio accumulated
    pub buffer_seconds: f64,
    /// Timestamp of last transcription
    pub last_transcript_time: f64,
}

/// Configuration for STT
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// Whisper model size (tiny, base, small, medium, large)
    pub model: String,
    /// Minimum silence duration to trigger transcription (seconds)
    pub silence_trigger: f64,
    /// Maximum buffer duration before forced transcription (seconds)
    pub max_buffer: f64,
    /// Minimum buffer duration to attempt transcription (seconds)
    pub min_buffer: f64,
    /// Language hint (empty = auto-detect)
    pub language: String,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            model: "base.en".into(),
            silence_trigger: 2.0, // Wait 2s of silence before transcribing
            max_buffer: 30.0,
            min_buffer: 1.0, // Need at least 1s of audio
            language: "en".into(),
        }
    }
}

/// Audio chunk for processing
#[derive(Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub voice_detected: bool,
}

/// Speech-to-text processor
pub struct SttProcessor {
    config: SttConfig,
    state: Arc<RwLock<SttState>>,
    tx: Option<Sender<AudioChunk>>,
    transcript_rx: Option<Receiver<String>>,
}

impl SttProcessor {
    /// Create new STT processor
    pub fn new(config: SttConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(SttState::default())),
            tx: None,
            transcript_rx: None,
        }
    }

    /// Get shared state handle
    pub fn state_handle(&self) -> Arc<RwLock<SttState>> {
        Arc::clone(&self.state)
    }

    /// Get transcript receiver (for polling new transcripts)
    pub fn transcript_receiver(&self) -> Option<Receiver<String>> {
        self.transcript_rx.clone()
    }

    /// Find whisper executable (handles Windows Python paths)
    fn find_whisper() -> Option<String> {
        // Try direct command first
        if Command::new("whisper").arg("--help").output().is_ok() {
            return Some("whisper".into());
        }

        // Windows: check common Python Scripts locations
        #[cfg(windows)]
        {
            if let Ok(home) = std::env::var("USERPROFILE") {
                let candidates = [
                    format!(
                        "{}/AppData/Roaming/Python/Python314/Scripts/whisper.exe",
                        home
                    ),
                    format!(
                        "{}/AppData/Roaming/Python/Python313/Scripts/whisper.exe",
                        home
                    ),
                    format!(
                        "{}/AppData/Roaming/Python/Python312/Scripts/whisper.exe",
                        home
                    ),
                    format!(
                        "{}/AppData/Roaming/Python/Python311/Scripts/whisper.exe",
                        home
                    ),
                    format!(
                        "{}/AppData/Local/Programs/Python/Python314/Scripts/whisper.exe",
                        home
                    ),
                    format!(
                        "{}/AppData/Local/Programs/Python/Python313/Scripts/whisper.exe",
                        home
                    ),
                    format!(
                        "{}/AppData/Local/Programs/Python/Python312/Scripts/whisper.exe",
                        home
                    ),
                ];
                for path in candidates {
                    if std::path::Path::new(&path).exists() {
                        return Some(path);
                    }
                }
            }
        }

        None
    }

    /// Start the STT processor
    pub fn start(&mut self) -> Result<(), SttError> {
        // Check whisper CLI exists
        let whisper_path = Self::find_whisper().ok_or(SttError::WhisperNotFound)?;

        let (audio_tx, audio_rx) = bounded::<AudioChunk>(64);
        let (transcript_tx, transcript_rx) = bounded::<String>(16);

        self.tx = Some(audio_tx);
        self.transcript_rx = Some(transcript_rx);

        let state = Arc::clone(&self.state);
        let config = self.config.clone();

        std::thread::Builder::new()
            .name("stt-processor".into())
            .spawn(move || {
                Self::process_loop(audio_rx, transcript_tx, state, config, whisper_path);
            })
            .map_err(|e| SttError::ThreadError(e.to_string()))?;

        Ok(())
    }

    /// Feed audio chunk to processor
    pub fn feed(&self, chunk: AudioChunk) {
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(chunk);
        }
    }

    /// Main processing loop
    fn process_loop(
        rx: Receiver<AudioChunk>,
        transcript_tx: Sender<String>,
        state: Arc<RwLock<SttState>>,
        config: SttConfig,
        whisper_path: String,
    ) {
        let mut audio_buffer: Vec<f32> = Vec::new();
        let mut last_voice_time = Instant::now();
        let mut was_voice = false;
        let mut source_sample_rate = 48000_u32;

        let mut voice_debug_counter = 0u64;
        while let Ok(chunk) = rx.recv() {
            source_sample_rate = chunk.sample_rate;

            // Debug voice detection
            voice_debug_counter += 1;
            if voice_debug_counter % 50 == 1 {
                let rms: f32 = (chunk.samples.iter().map(|s| s * s).sum::<f32>()
                    / chunk.samples.len() as f32)
                    .sqrt();
                eprintln!(
                    "[stt-dbg] voice={} rms={:.4} buf={:.1}s",
                    chunk.voice_detected,
                    rms,
                    audio_buffer.len() as f64 / source_sample_rate as f64
                );
            }

            // Track voice state
            if chunk.voice_detected {
                last_voice_time = Instant::now();
                was_voice = true;

                // Accumulate audio
                audio_buffer.extend(&chunk.samples);

                state.write().listening = true;
            }

            // Update buffer duration estimate
            let buffer_seconds = audio_buffer.len() as f64 / source_sample_rate as f64;
            state.write().buffer_seconds = buffer_seconds;

            // Check if we should transcribe
            let silence_duration = last_voice_time.elapsed().as_secs_f64();
            let should_transcribe = was_voice
                && ((silence_duration > config.silence_trigger
                    && buffer_seconds > config.min_buffer)
                    || buffer_seconds > config.max_buffer);

            if should_transcribe && !audio_buffer.is_empty() {
                state.write().transcribing = true;
                state.write().listening = false;

                eprintln!(
                    "[stt] Transcribing {:.1}s of audio ({} samples)",
                    buffer_seconds,
                    audio_buffer.len()
                );

                // Write WAV to temp file
                let temp_path = std::env::temp_dir().join("fractal_stt.wav");
                if let Err(e) = Self::write_wav(&audio_buffer, source_sample_rate, &temp_path) {
                    eprintln!("[stt] WAV write error: {}", e);
                    audio_buffer.clear();
                    was_voice = false;
                    state.write().transcribing = false;
                    continue;
                }

                // Run whisper CLI
                let output = Command::new(&whisper_path)
                    .arg(&temp_path)
                    .arg("--model")
                    .arg(&config.model)
                    .arg("--language")
                    .arg(&config.language)
                    .arg("--output_format")
                    .arg("txt")
                    .arg("--output_dir")
                    .arg(std::env::temp_dir())
                    .output();

                match output {
                    Ok(out) => {
                        eprintln!("[stt] Whisper finished, status={}", out.status);
                        if out.status.success() {
                            // Read transcript from output file
                            let txt_path = std::env::temp_dir().join("fractal_stt.txt");
                            eprintln!("[stt] Looking for transcript at {:?}", txt_path);
                            if let Ok(transcript) = std::fs::read_to_string(&txt_path) {
                                let transcript = transcript.trim().to_string();
                                eprintln!("[stt] Got transcript: '{}'", transcript);

                                if !transcript.is_empty() {
                                    // Update state
                                    {
                                        let mut s = state.write();
                                        s.transcript = transcript.clone();
                                        s.last_transcript_time = std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap()
                                            .as_secs_f64();
                                    }

                                    // Send to channel
                                    let _ = transcript_tx.try_send(transcript);
                                }

                                // Cleanup
                                let _ = std::fs::remove_file(&txt_path);
                            }
                        } else {
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            let stdout = String::from_utf8_lossy(&out.stdout);
                            eprintln!("[stt] Whisper failed! status={}", out.status);
                            eprintln!("[stt] stdout: {}", stdout);
                            eprintln!("[stt] stderr: {}", stderr);
                        }
                    }
                    Err(e) => {
                        eprintln!("[stt] Failed to run whisper: {}", e);
                    }
                }

                // Cleanup temp WAV
                let _ = std::fs::remove_file(&temp_path);

                // Reset
                audio_buffer.clear();
                was_voice = false;
                state.write().transcribing = false;
            }
        }
    }

    /// Write audio buffer to WAV file
    fn write_wav(
        samples: &[f32],
        sample_rate: u32,
        path: &std::path::Path,
    ) -> Result<(), SttError> {
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer =
            WavWriter::create(path, spec).map_err(|e| SttError::WavError(e.to_string()))?;

        for &sample in samples {
            let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| SttError::WavError(e.to_string()))?;
        }

        writer
            .finalize()
            .map_err(|e| SttError::WavError(e.to_string()))?;

        Ok(())
    }

    /// Get current state snapshot
    pub fn read(&self) -> SttState {
        self.state.read().clone()
    }
}

/// STT errors
#[derive(Debug)]
pub enum SttError {
    WhisperNotFound,
    ThreadError(String),
    WavError(String),
}

impl std::fmt::Display for SttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WhisperNotFound => {
                write!(f, "Whisper CLI not found (pip install openai-whisper)")
            }
            Self::ThreadError(e) => write!(f, "STT thread error: {}", e),
            Self::WavError(e) => write!(f, "WAV error: {}", e),
        }
    }
}

impl std::error::Error for SttError {}
