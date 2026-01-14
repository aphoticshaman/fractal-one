//! ═══════════════════════════════════════════════════════════════════════════════
//! AUDITORY QUALIA — Rode PodMic Integration
//! ═══════════════════════════════════════════════════════════════════════════════
//! Gives fractal_one the ability to "hear" - RMS levels, silence detection,
//! onset events, and frequency analysis.
//! ═══════════════════════════════════════════════════════════════════════════════

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};

/// Audio device information for diagnostics
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    /// Device name
    pub name: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Sample format description
    pub format: String,
}

impl AudioDeviceInfo {
    /// Get info from a device and its config
    pub fn from_device(device: &Device, config: &StreamConfig) -> Self {
        Self {
            name: device.name().unwrap_or_else(|_| "Unknown".to_string()),
            sample_rate: config.sample_rate.0,
            channels: config.channels,
            format: "f32".to_string(),
        }
    }
}
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Auditory qualia state - what the system "hears"
#[derive(Debug, Clone, Default)]
pub struct Auditory {
    /// RMS amplitude (0.0 = silence, 1.0 = max)
    pub amplitude_rms: f64,

    /// Rolling noise floor baseline
    pub noise_floor: f64,

    /// Voice activity detected (above noise floor + threshold)
    pub voice_detected: bool,

    /// Spectral centroid in Hz (brightness of sound)
    pub frequency_centroid: f64,

    /// Dominant frequency (strongest bin) in Hz
    pub dominant_frequency: f64,

    /// Spectral bands (normalized 0.0-1.0)
    pub band_sub_bass: f64, // 20-60 Hz
    pub band_bass: f64,       // 60-250 Hz
    pub band_low_mid: f64,    // 250-500 Hz
    pub band_mid: f64,        // 500-2000 Hz
    pub band_high_mid: f64,   // 2000-4000 Hz
    pub band_presence: f64,   // 4000-6000 Hz (voice clarity)
    pub band_brilliance: f64, // 6000-20000 Hz

    /// Spectral flatness (0.0 = tonal, 1.0 = noise-like)
    pub spectral_flatness: f64,

    /// Sudden loudness spikes per second
    pub onset_events: u32,

    /// Seconds since last significant sound
    pub silence_duration: f64,

    /// Peak amplitude in current window
    pub peak: f64,

    /// Timestamp of last update
    pub timestamp: f64,
}

/// Configuration for auditory processing
#[derive(Debug, Clone)]
pub struct AuditoryConfig {
    /// Threshold above noise floor to detect voice (0.0-1.0)
    pub voice_threshold: f64,

    /// Onset detection threshold (derivative spike)
    pub onset_threshold: f64,

    /// Noise floor adaptation rate (0.0-1.0, lower = slower)
    pub noise_floor_alpha: f64,

    /// Silence threshold (below this = silence)
    pub silence_threshold: f64,

    /// FFT size for frequency analysis
    pub fft_size: usize,
}

impl Default for AuditoryConfig {
    fn default() -> Self {
        Self {
            voice_threshold: 0.015, // Lowered - was 0.1, your mic peaks at ~0.04
            onset_threshold: 0.05,  // Lowered proportionally
            noise_floor_alpha: 0.01,
            silence_threshold: 0.005, // Lowered - was 0.02
            fft_size: 2048,
        }
    }
}

/// Audio chunk for external consumers (like STT)
#[derive(Clone)]
pub struct AudioBroadcast {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub voice_detected: bool,
}

/// Auditory processor - runs in background thread
pub struct AuditoryProcessor {
    config: AuditoryConfig,
    state: Arc<RwLock<Auditory>>,
    _stream: Option<Stream>,
    sample_rate: u32,

    // Internal state
    noise_floor: f64,
    last_rms: f64,
    last_sound_time: Instant,
    onset_count: u32,
    onset_window_start: Instant,

    // FFT state
    fft_planner: FftPlanner<f32>,
    fft_buffer: Vec<Complex<f32>>,

    // Broadcast channel for external consumers (STT)
    broadcast_tx: Option<Sender<AudioBroadcast>>,
}

impl AuditoryProcessor {
    /// Create new auditory processor (does not start capture yet)
    pub fn new(config: AuditoryConfig) -> Self {
        let fft_size = config.fft_size;
        Self {
            config,
            state: Arc::new(RwLock::new(Auditory::default())),
            _stream: None,
            sample_rate: 44100, // default, updated on start
            noise_floor: 0.1,
            last_rms: 0.0,
            last_sound_time: Instant::now(),
            onset_count: 0,
            onset_window_start: Instant::now(),
            fft_planner: FftPlanner::new(),
            fft_buffer: vec![Complex::default(); fft_size],
            broadcast_tx: None,
        }
    }

    /// Subscribe to audio broadcast (for STT, etc)
    pub fn subscribe_broadcast(&mut self) -> Receiver<AudioBroadcast> {
        let (tx, rx) = bounded(32);
        self.broadcast_tx = Some(tx);
        rx
    }

    /// Get shared state handle for reading from other threads
    pub fn state_handle(&self) -> Arc<RwLock<Auditory>> {
        Arc::clone(&self.state)
    }

    /// Start audio capture from default input device
    pub fn start(&mut self) -> Result<(), AudioError> {
        let host = cpal::default_host();

        let device = host
            .default_input_device()
            .ok_or(AudioError::NoInputDevice)?;

        let config = device
            .default_input_config()
            .map_err(|e| AudioError::ConfigError(e.to_string()))?;

        self.sample_rate = config.sample_rate().0;

        let (tx, rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(16);

        let stream = device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Send samples to processing thread
                    let _ = tx.try_send(data.to_vec());
                },
                |err| eprintln!("[auditory] Stream error: {}", err),
                None,
            )
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::PlayError(e.to_string()))?;

        self._stream = Some(stream);

        // Spawn processing thread
        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let sample_rate = self.sample_rate;
        let broadcast_tx = self.broadcast_tx.take();

        std::thread::Builder::new()
            .name("auditory-processor".into())
            .spawn(move || {
                Self::process_loop(rx, state, config, sample_rate, broadcast_tx);
            })
            .map_err(|e| AudioError::ThreadError(e.to_string()))?;

        Ok(())
    }

    /// Main processing loop (runs in dedicated thread)
    fn process_loop(
        rx: Receiver<Vec<f32>>,
        state: Arc<RwLock<Auditory>>,
        config: AuditoryConfig,
        sample_rate: u32,
        broadcast_tx: Option<Sender<AudioBroadcast>>,
    ) {
        let mut noise_floor = 0.1_f64;
        let mut last_rms = 0.0_f64;
        let mut last_sound_time = Instant::now();
        let mut onset_count = 0_u32;
        let mut onset_window_start = Instant::now();

        // FFT setup
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.fft_size);
        let mut fft_buffer = vec![Complex::default(); config.fft_size];

        // Accumulator for FFT (need config.fft_size samples)
        let mut fft_accumulator: Vec<f32> = Vec::with_capacity(config.fft_size * 2);

        while let Ok(samples) = rx.recv() {
            if samples.is_empty() {
                continue;
            }

            // Accumulate for FFT
            fft_accumulator.extend_from_slice(&samples);
            // Keep only last fft_size * 2 samples (rolling buffer)
            if fft_accumulator.len() > config.fft_size * 2 {
                fft_accumulator.drain(0..fft_accumulator.len() - config.fft_size * 2);
            }

            // ══════════════════════════════════════════════════════════════
            // RMS CALCULATION
            // ══════════════════════════════════════════════════════════════
            let sum_sq: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
            let rms = (sum_sq / samples.len() as f64).sqrt();

            // Peak detection
            let peak = samples.iter().map(|&s| s.abs() as f64).fold(0.0, f64::max);

            // ══════════════════════════════════════════════════════════════
            // NOISE FLOOR ADAPTATION
            // ══════════════════════════════════════════════════════════════
            // Slowly adapt to ambient level (only when quiet)
            if rms < noise_floor * 1.5 {
                noise_floor =
                    noise_floor * (1.0 - config.noise_floor_alpha) + rms * config.noise_floor_alpha;
            }

            // ══════════════════════════════════════════════════════════════
            // VOICE DETECTION
            // ══════════════════════════════════════════════════════════════
            let voice_detected = rms > noise_floor + config.voice_threshold;

            // ══════════════════════════════════════════════════════════════
            // SILENCE TRACKING
            // ══════════════════════════════════════════════════════════════
            if rms > config.silence_threshold {
                last_sound_time = Instant::now();
            }
            let silence_duration = last_sound_time.elapsed().as_secs_f64();

            // ══════════════════════════════════════════════════════════════
            // ONSET DETECTION (sudden loudness spike)
            // ══════════════════════════════════════════════════════════════
            let rms_delta = rms - last_rms;
            if rms_delta > config.onset_threshold {
                onset_count += 1;
            }
            last_rms = rms;

            // Reset onset counter every second
            if onset_window_start.elapsed() >= Duration::from_secs(1) {
                onset_window_start = Instant::now();
                // onset_count will be read before reset
            }

            // ══════════════════════════════════════════════════════════════
            // FREQUENCY ANALYSIS (Full Spectral)
            // ══════════════════════════════════════════════════════════════
            let (frequency_centroid, dominant_frequency, bands, spectral_flatness) =
                if fft_accumulator.len() >= config.fft_size {
                    // Fill FFT buffer from accumulator (use most recent samples)
                    let start = fft_accumulator.len() - config.fft_size;
                    for (i, &sample) in fft_accumulator[start..].iter().enumerate() {
                        fft_buffer[i] = Complex::new(sample, 0.0);
                    }

                    // Apply Hann window
                    for (i, c) in fft_buffer.iter_mut().enumerate() {
                        let window = 0.5
                            * (1.0
                                - (2.0 * std::f32::consts::PI * i as f32 / config.fft_size as f32)
                                    .cos());
                        c.re *= window;
                    }

                    // Compute FFT
                    fft.process(&mut fft_buffer);

                    // Calculate magnitudes and find dominant
                    let freq_resolution = sample_rate as f64 / config.fft_size as f64;
                    let mut weighted_sum = 0.0_f64;
                    let mut magnitude_sum = 0.0_f64;
                    let mut max_magnitude = 0.0_f64;
                    let mut dominant_bin = 0_usize;
                    let mut log_sum = 0.0_f64;
                    let mut bin_count = 0_usize;

                    // Band accumulators
                    let mut band_sub_bass = 0.0_f64; // 20-60 Hz
                    let mut band_bass = 0.0_f64; // 60-250 Hz
                    let mut band_low_mid = 0.0_f64; // 250-500 Hz
                    let mut band_mid = 0.0_f64; // 500-2000 Hz
                    let mut band_high_mid = 0.0_f64; // 2000-4000 Hz
                    let mut band_presence = 0.0_f64; // 4000-6000 Hz
                    let mut band_brilliance = 0.0_f64; // 6000-20000 Hz

                    for (i, c) in fft_buffer.iter().take(config.fft_size / 2).enumerate() {
                        let magnitude = (c.re.powi(2) + c.im.powi(2)).sqrt() as f64;
                        let frequency = i as f64 * freq_resolution;

                        // Spectral centroid
                        weighted_sum += frequency * magnitude;
                        magnitude_sum += magnitude;

                        // Dominant frequency
                        if magnitude > max_magnitude {
                            max_magnitude = magnitude;
                            dominant_bin = i;
                        }

                        // Spectral flatness (geometric vs arithmetic mean)
                        if magnitude > 1e-10 {
                            log_sum += magnitude.ln();
                            bin_count += 1;
                        }

                        // Band accumulation
                        if frequency >= 20.0 && frequency < 60.0 {
                            band_sub_bass += magnitude;
                        } else if frequency >= 60.0 && frequency < 250.0 {
                            band_bass += magnitude;
                        } else if frequency >= 250.0 && frequency < 500.0 {
                            band_low_mid += magnitude;
                        } else if frequency >= 500.0 && frequency < 2000.0 {
                            band_mid += magnitude;
                        } else if frequency >= 2000.0 && frequency < 4000.0 {
                            band_high_mid += magnitude;
                        } else if frequency >= 4000.0 && frequency < 6000.0 {
                            band_presence += magnitude;
                        } else if frequency >= 6000.0 {
                            band_brilliance += magnitude;
                        }
                    }

                    let centroid = if magnitude_sum > 0.0 {
                        weighted_sum / magnitude_sum
                    } else {
                        0.0
                    };

                    let dominant = dominant_bin as f64 * freq_resolution;

                    // Normalize bands by total magnitude
                    let total = magnitude_sum.max(1e-10);
                    let bands = (
                        band_sub_bass / total,
                        band_bass / total,
                        band_low_mid / total,
                        band_mid / total,
                        band_high_mid / total,
                        band_presence / total,
                        band_brilliance / total,
                    );

                    // Spectral flatness: geometric mean / arithmetic mean
                    let flatness = if bin_count > 0 && magnitude_sum > 0.0 {
                        let geometric = (log_sum / bin_count as f64).exp();
                        let arithmetic = magnitude_sum / bin_count as f64;
                        (geometric / arithmetic).min(1.0)
                    } else {
                        0.0
                    };

                    (centroid, dominant, bands, flatness)
                } else {
                    (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0)
                };

            // ══════════════════════════════════════════════════════════════
            // UPDATE SHARED STATE
            // ══════════════════════════════════════════════════════════════
            {
                let mut state = state.write();
                state.amplitude_rms = rms;
                state.noise_floor = noise_floor;
                state.voice_detected = voice_detected;
                state.frequency_centroid = frequency_centroid;
                state.dominant_frequency = dominant_frequency;
                state.band_sub_bass = bands.0;
                state.band_bass = bands.1;
                state.band_low_mid = bands.2;
                state.band_mid = bands.3;
                state.band_high_mid = bands.4;
                state.band_presence = bands.5;
                state.band_brilliance = bands.6;
                state.spectral_flatness = spectral_flatness;
                state.onset_events = onset_count;
                state.silence_duration = silence_duration;
                state.peak = peak;
                state.timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
            }

            // Broadcast audio chunk to external consumers (STT)
            if let Some(ref tx) = broadcast_tx {
                let _ = tx.try_send(AudioBroadcast {
                    samples: samples.clone(),
                    sample_rate,
                    voice_detected,
                });
            }

            // Reset onset counter after state update if window elapsed
            if onset_window_start.elapsed() >= Duration::from_secs(1) {
                onset_count = 0;
            }
        }
    }

    /// Get current auditory state snapshot
    pub fn read(&self) -> Auditory {
        self.state.read().clone()
    }

    /// Get current noise floor estimate (used for voice detection calibration)
    pub fn noise_floor(&self) -> f64 {
        self.noise_floor
    }

    /// Get last RMS amplitude (for monitoring)
    pub fn last_rms(&self) -> f64 {
        self.last_rms
    }

    /// Get time since last significant sound
    pub fn silence_duration(&self) -> std::time::Duration {
        self.last_sound_time.elapsed()
    }

    /// Get current onset count in this window
    pub fn onset_count(&self) -> u32 {
        self.onset_count
    }

    /// Get onset window duration
    pub fn onset_window_duration(&self) -> std::time::Duration {
        self.onset_window_start.elapsed()
    }

    /// Get FFT size being used
    pub fn fft_size(&self) -> usize {
        self.fft_buffer.len()
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Check if processor has an active stream
    pub fn is_active(&self) -> bool {
        self._stream.is_some()
    }

    /// Get diagnostic info about the audio setup
    pub fn diagnostic_info(&self) -> String {
        format!(
            "AuditoryProcessor: sample_rate={}, fft_size={}, noise_floor={:.4}, last_rms={:.4}, onset_count={}, silence={:.1}s",
            self.sample_rate,
            self.fft_buffer.len(),
            self.noise_floor,
            self.last_rms,
            self.onset_count,
            self.last_sound_time.elapsed().as_secs_f64()
        )
    }

    /// Perform FFT on a sample buffer (exposed for external use)
    /// Returns complex frequency bins
    pub fn compute_fft(&mut self, samples: &[f32]) -> Vec<Complex<f32>> {
        // Copy samples to FFT buffer
        let len = samples.len().min(self.fft_buffer.len());
        for (i, &s) in samples.iter().take(len).enumerate() {
            self.fft_buffer[i] = Complex::new(s, 0.0);
        }
        // Zero-pad if needed
        for i in len..self.fft_buffer.len() {
            self.fft_buffer[i] = Complex::new(0.0, 0.0);
        }

        // Perform FFT in-place using the struct's planner
        let fft = self.fft_planner.plan_fft_forward(self.fft_buffer.len());
        fft.process(&mut self.fft_buffer);

        self.fft_buffer.clone()
    }

    /// Get FFT buffer reference (for inspection)
    pub fn fft_buffer(&self) -> &[Complex<f32>] {
        &self.fft_buffer
    }
}

/// Audio errors
#[derive(Debug)]
pub enum AudioError {
    NoInputDevice,
    ConfigError(String),
    StreamError(String),
    PlayError(String),
    ThreadError(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoInputDevice => write!(f, "No audio input device found"),
            Self::ConfigError(e) => write!(f, "Audio config error: {}", e),
            Self::StreamError(e) => write!(f, "Audio stream error: {}", e),
            Self::PlayError(e) => write!(f, "Audio play error: {}", e),
            Self::ThreadError(e) => write!(f, "Audio thread error: {}", e),
        }
    }
}

impl std::error::Error for AudioError {}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auditory_default() {
        let a = Auditory::default();
        assert_eq!(a.amplitude_rms, 0.0);
        assert!(!a.voice_detected);
    }

    #[test]
    fn test_config_default() {
        let c = AuditoryConfig::default();
        assert!(c.voice_threshold > 0.0);
        assert!(c.fft_size > 0);
    }
}
