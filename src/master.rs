//! ═══════════════════════════════════════════════════════════════════════════════
//! MASTER — Audio Mastering Engine
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! A complete mastering chain:
//! 1. High-pass filter (remove rumble)
//! 2. Parametric EQ (tone shaping)
//! 3. Compression (dynamic control)
//! 4. Limiter (peak control)
//! 5. LUFS normalization (loudness targeting)
//! 6. Resampling (44.1kHz for CD/streaming)
//! 7. Dithering (16-bit for distribution)
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════════
// BIQUAD FILTER (for EQ)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    Peak,
    LowShelf,
    HighShelf,
}

#[derive(Debug, Clone)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // State for stereo
    x1_l: f64,
    x2_l: f64,
    y1_l: f64,
    y2_l: f64,
    x1_r: f64,
    x2_r: f64,
    y1_r: f64,
    y2_r: f64,
}

impl Biquad {
    pub fn new(filter_type: FilterType, sample_rate: f64, freq: f64, q: f64, gain_db: f64) -> Self {
        let omega = 2.0 * PI * freq / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);
        let a = 10.0_f64.powf(gain_db / 40.0);

        let (b0, b1, b2, a0, a1, a2) = match filter_type {
            FilterType::LowPass => {
                let b1 = 1.0 - cos_omega;
                let b0 = b1 / 2.0;
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::HighPass => {
                let b0 = (1.0 + cos_omega) / 2.0;
                let b1 = -(1.0 + cos_omega);
                let b2 = b0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::BandPass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_omega;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::Peak => {
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_omega;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha / a;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::LowShelf => {
                let sqrt_a = a.sqrt();
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
                let a2 = (a + 1.0) + (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            FilterType::HighShelf => {
                let sqrt_a = a.sqrt();
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_omega + 2.0 * sqrt_a * alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
                let a2 = (a + 1.0) - (a - 1.0) * cos_omega - 2.0 * sqrt_a * alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1_l: 0.0,
            x2_l: 0.0,
            y1_l: 0.0,
            y2_l: 0.0,
            x1_r: 0.0,
            x2_r: 0.0,
            y1_r: 0.0,
            y2_r: 0.0,
        }
    }

    pub fn process_stereo(&mut self, left: f64, right: f64) -> (f64, f64) {
        // Left channel
        let out_l = self.b0 * left + self.b1 * self.x1_l + self.b2 * self.x2_l
            - self.a1 * self.y1_l
            - self.a2 * self.y2_l;
        self.x2_l = self.x1_l;
        self.x1_l = left;
        self.y2_l = self.y1_l;
        self.y1_l = out_l;

        // Right channel
        let out_r = self.b0 * right + self.b1 * self.x1_r + self.b2 * self.x2_r
            - self.a1 * self.y1_r
            - self.a2 * self.y2_r;
        self.x2_r = self.x1_r;
        self.x1_r = right;
        self.y2_r = self.y1_r;
        self.y1_r = out_r;

        (out_l, out_r)
    }

    pub fn reset(&mut self) {
        self.x1_l = 0.0;
        self.x2_l = 0.0;
        self.y1_l = 0.0;
        self.y2_l = 0.0;
        self.x1_r = 0.0;
        self.x2_r = 0.0;
        self.y1_r = 0.0;
        self.y2_r = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPRESSOR
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Compressor {
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    release_ms: f64,
    makeup_db: f64,
    sample_rate: f64,
    envelope_l: f64,
    envelope_r: f64,
}

impl Compressor {
    pub fn new(
        sample_rate: f64,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        makeup_db: f64,
    ) -> Self {
        Self {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            sample_rate,
            envelope_l: 0.0,
            envelope_r: 0.0,
        }
    }

    pub fn process_stereo(&mut self, left: f64, right: f64) -> (f64, f64) {
        let attack_coef = (-1.0 / (self.attack_ms * 0.001 * self.sample_rate)).exp();
        let release_coef = (-1.0 / (self.release_ms * 0.001 * self.sample_rate)).exp();

        // Convert to dB
        let level_l = (left.abs() + 1e-10).log10() * 20.0;
        let level_r = (right.abs() + 1e-10).log10() * 20.0;

        // Envelope follower
        let coef_l = if level_l > self.envelope_l {
            attack_coef
        } else {
            release_coef
        };
        let coef_r = if level_r > self.envelope_r {
            attack_coef
        } else {
            release_coef
        };
        self.envelope_l = coef_l * self.envelope_l + (1.0 - coef_l) * level_l;
        self.envelope_r = coef_r * self.envelope_r + (1.0 - coef_r) * level_r;

        // Compute gain reduction
        let gr_l = if self.envelope_l > self.threshold_db {
            (self.envelope_l - self.threshold_db) * (1.0 - 1.0 / self.ratio)
        } else {
            0.0
        };
        let gr_r = if self.envelope_r > self.threshold_db {
            (self.envelope_r - self.threshold_db) * (1.0 - 1.0 / self.ratio)
        } else {
            0.0
        };

        // Apply gain reduction + makeup
        let gain_l = 10.0_f64.powf((-gr_l + self.makeup_db) / 20.0);
        let gain_r = 10.0_f64.powf((-gr_r + self.makeup_db) / 20.0);

        (left * gain_l, right * gain_r)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIMITER (lookahead brickwall)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Limiter {
    #[allow(dead_code)]
    ceiling_db: f64,
    ceiling_linear: f64,
    release_ms: f64,
    sample_rate: f64,
    lookahead_samples: usize,
    buffer_l: Vec<f64>,
    buffer_r: Vec<f64>,
    write_pos: usize,
    gain: f64,
}

impl Limiter {
    pub fn new(sample_rate: f64, ceiling_db: f64, release_ms: f64, lookahead_ms: f64) -> Self {
        let lookahead_samples = (lookahead_ms * 0.001 * sample_rate) as usize;
        Self {
            ceiling_db,
            ceiling_linear: 10.0_f64.powf(ceiling_db / 20.0),
            release_ms,
            sample_rate,
            lookahead_samples,
            buffer_l: vec![0.0; lookahead_samples + 1],
            buffer_r: vec![0.0; lookahead_samples + 1],
            write_pos: 0,
            gain: 1.0,
        }
    }

    pub fn process_stereo(&mut self, left: f64, right: f64) -> (f64, f64) {
        // Store in lookahead buffer
        self.buffer_l[self.write_pos] = left;
        self.buffer_r[self.write_pos] = right;

        // Find peak in lookahead window
        let mut peak = 0.0_f64;
        for i in 0..self.buffer_l.len() {
            peak = peak.max(self.buffer_l[i].abs()).max(self.buffer_r[i].abs());
        }

        // Calculate target gain
        let target_gain = if peak > self.ceiling_linear {
            self.ceiling_linear / peak
        } else {
            1.0
        };

        // Smooth gain changes
        let release_coef = (-1.0 / (self.release_ms * 0.001 * self.sample_rate)).exp();
        if target_gain < self.gain {
            self.gain = target_gain; // Instant attack
        } else {
            self.gain = release_coef * self.gain + (1.0 - release_coef) * target_gain;
        }

        // Read from delayed position
        let read_pos = (self.write_pos + 1) % self.buffer_l.len();
        let out_l = self.buffer_l[read_pos] * self.gain;
        let out_r = self.buffer_r[read_pos] * self.gain;

        self.write_pos = (self.write_pos + 1) % self.buffer_l.len();

        (out_l, out_r)
    }

    /// Flush the limiter (process remaining samples in buffer)
    pub fn flush(&mut self) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(self.lookahead_samples);
        for _ in 0..self.lookahead_samples {
            out.push(self.process_stereo(0.0, 0.0));
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LUFS MEASUREMENT (ITU-R BS.1770)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct LufsMeter {
    // K-weighting filters
    high_shelf: Biquad,
    high_pass: Biquad,
    // Gating
    block_size: usize,
    overlap: usize,
    buffer_l: Vec<f64>,
    buffer_r: Vec<f64>,
    blocks: Vec<f64>, // Power per block
}

impl LufsMeter {
    pub fn new(sample_rate: f64) -> Self {
        // K-weighting stage 1: high shelf +4dB at 1681Hz
        let high_shelf = Biquad::new(FilterType::HighShelf, sample_rate, 1681.0, 0.7, 4.0);
        // K-weighting stage 2: high pass at 38Hz
        let high_pass = Biquad::new(FilterType::HighPass, sample_rate, 38.0, 0.5, 0.0);

        let block_size = (0.4 * sample_rate) as usize; // 400ms blocks
        let overlap = block_size * 3 / 4; // 75% overlap

        Self {
            high_shelf,
            high_pass,
            block_size,
            overlap,
            buffer_l: Vec::new(),
            buffer_r: Vec::new(),
            blocks: Vec::new(),
        }
    }

    pub fn process(&mut self, left: f64, right: f64) {
        // Apply K-weighting
        let (l, r) = self.high_shelf.process_stereo(left, right);
        let (l, r) = self.high_pass.process_stereo(l, r);

        self.buffer_l.push(l);
        self.buffer_r.push(r);

        // Process blocks
        while self.buffer_l.len() >= self.block_size {
            let mut power = 0.0;
            for i in 0..self.block_size {
                power += self.buffer_l[i].powi(2) + self.buffer_r[i].powi(2);
            }
            power /= self.block_size as f64 * 2.0; // Stereo normalization

            self.blocks.push(power);

            // Remove overlap portion
            let remove = self.block_size - self.overlap;
            self.buffer_l.drain(0..remove);
            self.buffer_r.drain(0..remove);
        }
    }

    pub fn integrated_loudness(&self) -> f64 {
        if self.blocks.is_empty() {
            return -70.0;
        }

        // Absolute gate: -70 LUFS
        let abs_threshold = 10.0_f64.powf(-70.0 / 10.0);
        let above_abs: Vec<f64> = self
            .blocks
            .iter()
            .filter(|&&p| p > abs_threshold)
            .copied()
            .collect();

        if above_abs.is_empty() {
            return -70.0;
        }

        // Relative gate: -10 LU below ungated mean
        let mean_power: f64 = above_abs.iter().sum::<f64>() / above_abs.len() as f64;
        let rel_threshold = mean_power * 10.0_f64.powf(-10.0 / 10.0);

        let above_rel: Vec<f64> = above_abs
            .iter()
            .filter(|&&p| p > rel_threshold)
            .copied()
            .collect();

        if above_rel.is_empty() {
            return -70.0;
        }

        let final_power: f64 = above_rel.iter().sum::<f64>() / above_rel.len() as f64;
        -0.691 + 10.0 * final_power.log10()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DITHERING (TPDF)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct Dither {
    rng_state: u64,
}

impl Dither {
    pub fn new() -> Self {
        Self {
            rng_state: 0x853c49e6748fea9b,
        }
    }

    fn next_random(&mut self) -> f64 {
        // xorshift64
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        self.rng_state = self.rng_state.wrapping_mul(0x2545F4914F6CDD1D);
        (self.rng_state as f64) / (u64::MAX as f64) - 0.5
    }

    /// Apply TPDF dither for 16-bit quantization
    pub fn dither_16bit(&mut self, sample: f64) -> i16 {
        // TPDF: sum of two uniform distributions
        let dither = (self.next_random() + self.next_random()) / 32768.0;
        let scaled = (sample + dither) * 32767.0;
        scaled.round().clamp(-32768.0, 32767.0) as i16
    }
}

impl Default for Dither {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MASTERING CHAIN
// ═══════════════════════════════════════════════════════════════════════════════

/// Mastering preset
#[derive(Debug, Clone)]
pub struct MasterPreset {
    pub name: String,
    // High-pass
    pub highpass_freq: f64,
    // EQ bands: (freq, gain_db, q)
    pub eq_bands: Vec<(f64, f64, f64)>,
    // High shelf
    pub high_shelf_freq: f64,
    pub high_shelf_gain_db: f64,
    // Low shelf
    pub low_shelf_freq: f64,
    pub low_shelf_gain_db: f64,
    // Compression
    pub comp_threshold_db: f64,
    pub comp_ratio: f64,
    pub comp_attack_ms: f64,
    pub comp_release_ms: f64,
    pub comp_makeup_db: f64,
    // Limiting
    pub limiter_ceiling_db: f64,
    // Target loudness
    pub target_lufs: f64,
}

impl Default for MasterPreset {
    fn default() -> Self {
        Self::streaming()
    }
}

impl MasterPreset {
    /// Streaming preset (Spotify, Apple Music, etc.)
    pub fn streaming() -> Self {
        Self {
            name: "Streaming".into(),
            highpass_freq: 30.0,
            eq_bands: vec![
                (80.0, 1.0, 1.0),   // Slight bass warmth
                (250.0, -1.5, 1.5), // Cut mud
                (3000.0, 1.5, 1.2), // Presence
                (8000.0, 1.0, 0.8), // Air
            ],
            high_shelf_freq: 10000.0,
            high_shelf_gain_db: 1.5,
            low_shelf_freq: 100.0,
            low_shelf_gain_db: 1.0,
            comp_threshold_db: -18.0,
            comp_ratio: 2.0,
            comp_attack_ms: 30.0,
            comp_release_ms: 200.0,
            comp_makeup_db: 2.0,
            limiter_ceiling_db: -1.0,
            target_lufs: -14.0,
        }
    }

    /// CD/Loud preset
    pub fn cd_loud() -> Self {
        Self {
            name: "CD Loud".into(),
            highpass_freq: 25.0,
            eq_bands: vec![
                (60.0, 2.0, 1.0),    // Bass boost
                (200.0, -2.0, 1.5),  // Cut mud
                (2500.0, 2.0, 1.0),  // Presence
                (10000.0, 1.5, 0.7), // Brightness
            ],
            high_shelf_freq: 12000.0,
            high_shelf_gain_db: 2.0,
            low_shelf_freq: 80.0,
            low_shelf_gain_db: 2.0,
            comp_threshold_db: -14.0,
            comp_ratio: 4.0,
            comp_attack_ms: 10.0,
            comp_release_ms: 100.0,
            comp_makeup_db: 4.0,
            limiter_ceiling_db: -0.3,
            target_lufs: -9.0,
        }
    }

    /// Transparent/Gentle preset
    pub fn transparent() -> Self {
        Self {
            name: "Transparent".into(),
            highpass_freq: 20.0,
            eq_bands: vec![],
            high_shelf_freq: 8000.0,
            high_shelf_gain_db: 0.5,
            low_shelf_freq: 100.0,
            low_shelf_gain_db: 0.0,
            comp_threshold_db: -24.0,
            comp_ratio: 1.5,
            comp_attack_ms: 50.0,
            comp_release_ms: 300.0,
            comp_makeup_db: 1.0,
            limiter_ceiling_db: -1.0,
            target_lufs: -14.0,
        }
    }
}

/// The mastering engine
pub struct MasteringEngine {
    pub preset: MasterPreset,
    #[allow(dead_code)]
    sample_rate: f64,
    target_sample_rate: f64,
    // DSP chain
    highpass: Biquad,
    eq_filters: Vec<Biquad>,
    low_shelf: Biquad,
    high_shelf: Biquad,
    compressor: Compressor,
    limiter: Limiter,
    lufs_meter: LufsMeter,
    dither: Dither,
}

impl MasteringEngine {
    pub fn new(sample_rate: f64, preset: MasterPreset) -> Self {
        let target_sample_rate = 44100.0;

        // Build filter chain
        let highpass = Biquad::new(
            FilterType::HighPass,
            sample_rate,
            preset.highpass_freq,
            0.707,
            0.0,
        );

        let eq_filters: Vec<Biquad> = preset
            .eq_bands
            .iter()
            .map(|&(freq, gain, q)| Biquad::new(FilterType::Peak, sample_rate, freq, q, gain))
            .collect();

        let low_shelf = Biquad::new(
            FilterType::LowShelf,
            sample_rate,
            preset.low_shelf_freq,
            0.707,
            preset.low_shelf_gain_db,
        );
        let high_shelf = Biquad::new(
            FilterType::HighShelf,
            sample_rate,
            preset.high_shelf_freq,
            0.707,
            preset.high_shelf_gain_db,
        );

        let compressor = Compressor::new(
            sample_rate,
            preset.comp_threshold_db,
            preset.comp_ratio,
            preset.comp_attack_ms,
            preset.comp_release_ms,
            preset.comp_makeup_db,
        );

        let limiter = Limiter::new(sample_rate, preset.limiter_ceiling_db, 50.0, 5.0);

        let lufs_meter = LufsMeter::new(sample_rate);
        let dither = Dither::new();

        Self {
            preset,
            sample_rate,
            target_sample_rate,
            highpass,
            eq_filters,
            low_shelf,
            high_shelf,
            compressor,
            limiter,
            lufs_meter,
            dither,
        }
    }

    /// Process a single stereo sample through the mastering chain
    pub fn process_sample(&mut self, left: f64, right: f64) -> (f64, f64) {
        // 1. High-pass (remove rumble)
        let (l, r) = self.highpass.process_stereo(left, right);

        // 2. Parametric EQ
        let (mut l, mut r) = (l, r);
        for eq in &mut self.eq_filters {
            let (nl, nr) = eq.process_stereo(l, r);
            l = nl;
            r = nr;
        }

        // 3. Shelving EQ
        let (l, r) = self.low_shelf.process_stereo(l, r);
        let (l, r) = self.high_shelf.process_stereo(l, r);

        // 4. Compression
        let (l, r) = self.compressor.process_stereo(l, r);

        // 5. Limiting
        let (l, r) = self.limiter.process_stereo(l, r);

        // 6. Meter for LUFS
        self.lufs_meter.process(l, r);

        (l, r)
    }

    /// Get current integrated loudness
    pub fn get_lufs(&self) -> f64 {
        self.lufs_meter.integrated_loudness()
    }

    /// Calculate gain needed to hit target LUFS
    pub fn calculate_normalization_gain(&self) -> f64 {
        let current = self.get_lufs();
        let target = self.preset.target_lufs;
        10.0_f64.powf((target - current) / 20.0)
    }

    /// Apply dithering for 16-bit output
    pub fn dither_sample(&mut self, sample: f64) -> i16 {
        self.dither.dither_16bit(sample)
    }

    /// Get target sample rate
    pub fn target_sample_rate(&self) -> u32 {
        self.target_sample_rate as u32
    }

    /// Flush limiter buffer
    pub fn flush_limiter(&mut self) -> Vec<(f64, f64)> {
        self.limiter.flush()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MASTER FUNCTION (full pipeline)
// ═══════════════════════════════════════════════════════════════════════════════

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Master an audio file
pub fn master_file(
    input_path: &str,
    output_path: &str,
    preset: MasterPreset,
) -> Result<MasterResult, String> {
    // Read input WAV
    let mut reader =
        hound::WavReader::open(input_path).map_err(|e| format!("Failed to open input: {}", e))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    if channels != 2 {
        return Err(format!("Expected stereo input, got {} channels", channels));
    }

    println!(
        "  Input: {}Hz, {}-bit, {} channels",
        spec.sample_rate, spec.bits_per_sample, channels
    );

    // Read all samples
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f64 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap() as f64).collect(),
    };

    let num_samples = samples.len() / 2;
    println!(
        "  Duration: {:.1}s ({} samples)",
        num_samples as f64 / sample_rate,
        num_samples
    );

    // Deinterleave to stereo
    let mut left: Vec<f64> = Vec::with_capacity(num_samples);
    let mut right: Vec<f64> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        left.push(samples[i * 2]);
        right.push(samples[i * 2 + 1]);
    }

    // Create mastering engine
    let mut engine = MasteringEngine::new(sample_rate, preset.clone());

    println!("  Preset: {}", preset.name);
    println!("  Processing...");

    // First pass: process through chain and measure loudness
    let mut processed_l: Vec<f64> = Vec::with_capacity(num_samples);
    let mut processed_r: Vec<f64> = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let (l, r) = engine.process_sample(left[i], right[i]);
        processed_l.push(l);
        processed_r.push(r);
    }

    // Flush limiter
    let flush = engine.flush_limiter();
    for (l, r) in flush {
        processed_l.push(l);
        processed_r.push(r);
    }

    // Calculate LUFS and normalization gain
    let measured_lufs = engine.get_lufs();
    let norm_gain = engine.calculate_normalization_gain();
    println!("  Measured LUFS: {:.1}", measured_lufs);
    println!("  Target LUFS: {:.1}", preset.target_lufs);
    println!("  Normalization gain: {:.2}dB", 20.0 * norm_gain.log10());

    // Apply normalization
    for i in 0..processed_l.len() {
        processed_l[i] *= norm_gain;
        processed_r[i] *= norm_gain;
    }

    // Resample if needed
    let target_rate = 44100.0;
    let (final_l, final_r) = if (sample_rate - target_rate).abs() > 1.0 {
        println!(
            "  Resampling: {}Hz -> {}Hz",
            sample_rate as u32, target_rate as u32
        );

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler =
            SincFixedIn::<f64>::new(target_rate / sample_rate, 2.0, params, processed_l.len(), 2)
                .map_err(|e| format!("Resampler error: {}", e))?;

        let input = vec![processed_l.clone(), processed_r.clone()];
        let output = resampler
            .process(&input, None)
            .map_err(|e| format!("Resampling error: {}", e))?;

        (output[0].clone(), output[1].clone())
    } else {
        (processed_l, processed_r)
    };

    // Find true peak after all processing
    let mut true_peak = 0.0_f64;
    for i in 0..final_l.len() {
        true_peak = true_peak.max(final_l[i].abs()).max(final_r[i].abs());
    }
    let true_peak_db = 20.0 * true_peak.log10();
    println!("  True peak: {:.1}dBTP", true_peak_db);

    // Dither and write output
    println!("  Dithering to 16-bit...");
    let mut dither = Dither::new();

    let out_spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, out_spec)
        .map_err(|e| format!("Failed to create output: {}", e))?;

    for i in 0..final_l.len() {
        writer
            .write_sample(dither.dither_16bit(final_l[i]))
            .map_err(|e| format!("Write error: {}", e))?;
        writer
            .write_sample(dither.dither_16bit(final_r[i]))
            .map_err(|e| format!("Write error: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Finalize error: {}", e))?;

    println!("  Output: {}", output_path);

    Ok(MasterResult {
        input_sample_rate: sample_rate as u32,
        output_sample_rate: 44100,
        input_duration_secs: num_samples as f64 / sample_rate,
        output_duration_secs: final_l.len() as f64 / 44100.0,
        measured_lufs,
        target_lufs: preset.target_lufs,
        true_peak_dbtp: true_peak_db,
        normalization_db: 20.0 * norm_gain.log10(),
    })
}

/// Result of mastering
#[derive(Debug, Clone)]
pub struct MasterResult {
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub input_duration_secs: f64,
    pub output_duration_secs: f64,
    pub measured_lufs: f64,
    pub target_lufs: f64,
    pub true_peak_dbtp: f64,
    pub normalization_db: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEGANOGRAPHY — Ultrasonic Message Embedding
// ═══════════════════════════════════════════════════════════════════════════════
//
// Embeds text messages in the ultrasonic band (18-20kHz) using FSK modulation.
// Humans mostly can't hear this range, but spectrum analyzers will detect it.
// The message is encoded as Morse code patterns for maximum detectability.

/// Morse code timing constants (in samples at 44.1kHz)
const MORSE_DOT: usize = 2205; // 50ms
const MORSE_DASH: usize = 6615; // 150ms
const MORSE_GAP: usize = 2205; // 50ms between elements
const MORSE_LETTER_GAP: usize = 6615; // 150ms between letters
const MORSE_WORD_GAP: usize = 15435; // 350ms between words

/// Morse code lookup
fn char_to_morse(c: char) -> Option<&'static str> {
    match c.to_ascii_uppercase() {
        'A' => Some(".-"),
        'B' => Some("-..."),
        'C' => Some("-.-."),
        'D' => Some("-.."),
        'E' => Some("."),
        'F' => Some("..-."),
        'G' => Some("--."),
        'H' => Some("...."),
        'I' => Some(".."),
        'J' => Some(".---"),
        'K' => Some("-.-"),
        'L' => Some(".-.."),
        'M' => Some("--"),
        'N' => Some("-."),
        'O' => Some("---"),
        'P' => Some(".--."),
        'Q' => Some("--.-"),
        'R' => Some(".-."),
        'S' => Some("..."),
        'T' => Some("-"),
        'U' => Some("..-"),
        'V' => Some("...-"),
        'W' => Some(".--"),
        'X' => Some("-..-"),
        'Y' => Some("-.--"),
        'Z' => Some("--.."),
        '0' => Some("-----"),
        '1' => Some(".----"),
        '2' => Some("..---"),
        '3' => Some("...--"),
        '4' => Some("....-"),
        '5' => Some("....."),
        '6' => Some("-...."),
        '7' => Some("--..."),
        '8' => Some("---.."),
        '9' => Some("----."),
        ' ' => Some(" "), // Word gap
        ',' => Some("--..--"),
        '.' => Some(".-.-.-"),
        _ => None,
    }
}

/// Generate morse-encoded ultrasonic signal
#[allow(dead_code)]
fn generate_morse_signal(message: &str, sample_rate: f64, carrier_freq: f64) -> Vec<f64> {
    let mut signal = Vec::new();

    // Generate a tone burst
    let gen_tone = |duration: usize, freq: f64| -> Vec<f64> {
        let mut tone = Vec::with_capacity(duration);
        for i in 0..duration {
            let t = i as f64 / sample_rate;
            // Smooth envelope to avoid clicks
            let env = if i < 100 {
                i as f64 / 100.0
            } else if i > duration - 100 {
                (duration - i) as f64 / 100.0
            } else {
                1.0
            };
            tone.push(env * (2.0 * PI * freq * t).sin());
        }
        tone
    };

    // Generate silence
    let gen_silence = |duration: usize| -> Vec<f64> { vec![0.0; duration] };

    // Scale timing for sample rate
    let scale = sample_rate / 44100.0;
    let dot = (MORSE_DOT as f64 * scale) as usize;
    let dash = (MORSE_DASH as f64 * scale) as usize;
    let gap = (MORSE_GAP as f64 * scale) as usize;
    let letter_gap = (MORSE_LETTER_GAP as f64 * scale) as usize;
    let word_gap = (MORSE_WORD_GAP as f64 * scale) as usize;

    let mut first_element = true;

    for c in message.chars() {
        if let Some(morse) = char_to_morse(c) {
            if morse == " " {
                signal.extend(gen_silence(word_gap));
                first_element = true;
            } else {
                // Letter gap before this letter (if not first)
                if !first_element {
                    signal.extend(gen_silence(letter_gap));
                }

                let mut first_in_letter = true;
                for element in morse.chars() {
                    // Element gap
                    if !first_in_letter {
                        signal.extend(gen_silence(gap));
                    }

                    match element {
                        '.' => signal.extend(gen_tone(dot, carrier_freq)),
                        '-' => signal.extend(gen_tone(dash, carrier_freq)),
                        _ => {}
                    }
                    first_in_letter = false;
                }
                first_element = false;
            }
        }
    }

    signal
}

/// Spread spectrum PRNG for frequency hopping
#[allow(dead_code)]
struct SpreadSpectrum {
    state: u64,
    base_freq: f64,
    bandwidth: f64,
}

#[allow(dead_code)]
impl SpreadSpectrum {
    fn new(seed: u64, base_freq: f64, bandwidth: f64) -> Self {
        Self {
            state: seed,
            base_freq,
            bandwidth,
        }
    }

    /// Get next pseudo-random frequency in the band
    fn next_freq(&mut self) -> f64 {
        // xorshift64
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state = self.state.wrapping_mul(0x2545F4914F6CDD1D);

        // Map to frequency band
        let normalized = (self.state as f64) / (u64::MAX as f64);
        self.base_freq + normalized * self.bandwidth
    }

    /// Get time jitter (±samples)
    fn jitter(&mut self, max_jitter: usize) -> i32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state = self.state.wrapping_mul(0x2545F4914F6CDD1D);

        let normalized = (self.state as f64) / (u64::MAX as f64);
        ((normalized - 0.5) * 2.0 * max_jitter as f64) as i32
    }

    /// Get amplitude variation
    fn amplitude_var(&mut self, base: f64, variance: f64) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state = self.state.wrapping_mul(0x2545F4914F6CDD1D);

        let normalized = (self.state as f64) / (u64::MAX as f64);
        base * (1.0 + (normalized - 0.5) * variance)
    }
}

/// Generate obfuscated spread-spectrum morse signal
/// Uses frequency hopping + time jitter + amplitude variation
#[allow(dead_code)]
fn generate_obfuscated_morse(
    message: &str,
    sample_rate: f64,
    base_freq: f64,
    seed: u64,
) -> Vec<f64> {
    let mut signal = Vec::new();
    let mut ss = SpreadSpectrum::new(seed, base_freq, 1500.0); // 1.5kHz spread

    // Generate a frequency-hopping tone burst with envelope
    let gen_tone = |ss: &mut SpreadSpectrum, duration: usize| -> Vec<f64> {
        let mut tone = Vec::with_capacity(duration);
        let freq = ss.next_freq();
        let amp = ss.amplitude_var(1.0, 0.4); // ±40% amplitude variation

        for i in 0..duration {
            let t = i as f64 / sample_rate;
            // Smooth envelope with randomized attack/release
            let attack = 50 + (ss.jitter(30).abs() as usize);
            let release = 50 + (ss.jitter(30).abs() as usize);

            let env = if i < attack {
                (i as f64 / attack as f64).powi(2) // Squared for softer attack
            } else if i > duration.saturating_sub(release) {
                let remaining = duration - i;
                (remaining as f64 / release as f64).powi(2)
            } else {
                1.0
            };

            // Add slight frequency wobble (natural drift)
            let wobble = 1.0 + 0.002 * (t * 3.7).sin();
            tone.push(env * amp * (2.0 * PI * freq * wobble * t).sin());
        }
        tone
    };

    // Generate "noisy" silence with artifacts
    let gen_silence = |ss: &mut SpreadSpectrum, duration: usize| -> Vec<f64> {
        let mut silence = Vec::with_capacity(duration);
        for i in 0..duration {
            // Occasional micro-bursts (like natural HF artifacts)
            if ss.jitter(100) > 95 {
                let burst_len = 20 + ss.jitter(20).unsigned_abs() as usize;
                let freq = ss.next_freq();
                for j in 0..burst_len.min(duration - i) {
                    let t = j as f64 / sample_rate;
                    let env = 1.0 - (j as f64 / burst_len as f64);
                    silence.push(env * 0.3 * (2.0 * PI * freq * t).sin());
                }
            } else {
                silence.push(0.0);
            }
        }
        // Pad to correct length
        silence.resize(duration, 0.0);
        silence
    };

    // Scale timing with jitter
    let scale = sample_rate / 44100.0;
    let base_dot = (MORSE_DOT as f64 * scale) as usize;
    let base_dash = (MORSE_DASH as f64 * scale) as usize;
    let base_gap = (MORSE_GAP as f64 * scale) as usize;
    let base_letter_gap = (MORSE_LETTER_GAP as f64 * scale) as usize;
    let base_word_gap = (MORSE_WORD_GAP as f64 * scale) as usize;

    let mut first_element = true;

    for c in message.chars() {
        if let Some(morse) = char_to_morse(c) {
            if morse == " " {
                let jittered = (base_word_gap as i32 + ss.jitter(base_word_gap / 4)) as usize;
                signal.extend(gen_silence(&mut ss, jittered.max(100)));
                first_element = true;
            } else {
                if !first_element {
                    let jittered =
                        (base_letter_gap as i32 + ss.jitter(base_letter_gap / 4)) as usize;
                    signal.extend(gen_silence(&mut ss, jittered.max(100)));
                }

                let mut first_in_letter = true;
                for element in morse.chars() {
                    if !first_in_letter {
                        let jittered = (base_gap as i32 + ss.jitter(base_gap / 3)) as usize;
                        signal.extend(gen_silence(&mut ss, jittered.max(50)));
                    }

                    match element {
                        '.' => {
                            let jittered = (base_dot as i32 + ss.jitter(base_dot / 4)) as usize;
                            signal.extend(gen_tone(&mut ss, jittered.max(100)));
                        }
                        '-' => {
                            let jittered = (base_dash as i32 + ss.jitter(base_dash / 4)) as usize;
                            signal.extend(gen_tone(&mut ss, jittered.max(100)));
                        }
                        _ => {}
                    }
                    first_in_letter = false;
                }
                first_element = false;
            }
        }
    }

    signal
}

/// Embed a message using CLEAN, VISIBLE morse in the ultrasonic band
///
/// This is NOT hidden - it's designed to be OBVIOUS on a spectrogram.
/// Audiophiles will see clean morse code patterns at 19kHz and think:
/// "Wait one fucking second..."
///
/// But it's above most speakers' range, so:
/// - Inaudible on 99% of playback systems
/// - Won't trigger "hidden audio" rejection (it's just high frequency content)
/// - Visually obvious to anyone who opens a spectrogram
fn generate_clean_morse(message: &str, sample_rate: f64, carrier_freq: f64) -> Vec<f64> {
    let mut signal = Vec::new();

    // Clean tone generation - no obfuscation, pure sine
    let gen_tone = |duration: usize, freq: f64| -> Vec<f64> {
        let mut tone = Vec::with_capacity(duration);
        for i in 0..duration {
            let t = i as f64 / sample_rate;
            // Smooth envelope to avoid clicks (but clean, not randomized)
            let attack = 200;
            let release = 200;
            let env = if i < attack {
                (i as f64 / attack as f64).powi(2)
            } else if i > duration.saturating_sub(release) {
                let remaining = duration - i;
                (remaining as f64 / release as f64).powi(2)
            } else {
                1.0
            };
            tone.push(env * (2.0 * PI * freq * t).sin());
        }
        tone
    };

    let gen_silence = |duration: usize| -> Vec<f64> { vec![0.0; duration] };

    // Standard morse timing - clean and regular
    let scale = sample_rate / 44100.0;
    let dot = (MORSE_DOT as f64 * scale) as usize;
    let dash = (MORSE_DASH as f64 * scale) as usize;
    let gap = (MORSE_GAP as f64 * scale) as usize;
    let letter_gap = (MORSE_LETTER_GAP as f64 * scale) as usize;
    let word_gap = (MORSE_WORD_GAP as f64 * scale) as usize;

    let mut first_element = true;

    for c in message.chars() {
        if let Some(morse) = char_to_morse(c) {
            if morse == " " {
                signal.extend(gen_silence(word_gap));
                first_element = true;
            } else {
                if !first_element {
                    signal.extend(gen_silence(letter_gap));
                }

                let mut first_in_letter = true;
                for element in morse.chars() {
                    if !first_in_letter {
                        signal.extend(gen_silence(gap));
                    }

                    match element {
                        '.' => signal.extend(gen_tone(dot, carrier_freq)),
                        '-' => signal.extend(gen_tone(dash, carrier_freq)),
                        _ => {}
                    }
                    first_in_letter = false;
                }
                first_element = false;
            }
        }
    }

    signal
}

/// Embed a visible ultrasonic message
///
/// NOT steganography - this is meant to be FOUND.
/// Clean morse at 19kHz that lights up like a Christmas tree on any spectrogram.
/// Plausibly "just high frequency content" to automated systems.
/// Obviously intentional to anyone who looks.
pub fn embed_message(
    input_path: &str,
    output_path: &str,
    message: &str,
    carrier_freq: f64,
) -> Result<(), String> {
    let mut reader =
        hound::WavReader::open(input_path).map_err(|e| format!("Failed to open input: {}", e))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    if carrier_freq >= sample_rate / 2.0 {
        return Err(format!(
            "Carrier {:.0}Hz exceeds Nyquist ({:.0}Hz)",
            carrier_freq,
            sample_rate / 2.0
        ));
    }

    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f64 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap() as f64).collect(),
    };

    let num_samples = samples.len() / channels;

    // Generate CLEAN morse - visible on spectrogram
    // Message repeated 3x throughout the track
    let full_message = format!("... {} ... {} ... {} ...", message, message, message);
    let morse_signal = generate_clean_morse(&full_message, sample_rate, carrier_freq);

    // Amplitude: -30dB = 0.03
    // Visible on spectrogram, but still below audible threshold at 19kHz
    // Most humans can't hear above 17kHz anyway, and this is quiet
    let embed_amplitude = 0.03;

    let mut output: Vec<f64> = samples.clone();

    for (i, &morse_sample) in morse_signal.iter().enumerate() {
        if i >= num_samples {
            break;
        }

        for ch in 0..channels {
            let idx = i * channels + ch;
            if idx < output.len() {
                output[idx] += morse_sample * embed_amplitude;
                output[idx] = output[idx].clamp(-0.99, 0.99);
            }
        }
    }

    let mut writer = hound::WavWriter::create(output_path, spec)
        .map_err(|e| format!("Failed to create output: {}", e))?;

    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f64;
            for sample in output {
                writer
                    .write_sample((sample * max) as i32)
                    .map_err(|e| format!("Write error: {}", e))?;
            }
        }
        hound::SampleFormat::Float => {
            for sample in output {
                writer
                    .write_sample(sample as f32)
                    .map_err(|e| format!("Write error: {}", e))?;
            }
        }
    }

    writer
        .finalize()
        .map_err(|e| format!("Finalize error: {}", e))?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECODER — Extract morse from ultrasonic band
// ═══════════════════════════════════════════════════════════════════════════════

/// Decode morse from the ultrasonic band of an audio file
pub fn decode_message(
    input_path: &str,
    carrier_freq: f64,
    bandwidth: f64,
) -> Result<String, String> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let mut reader =
        hound::WavReader::open(input_path).map_err(|e| format!("Failed to open: {}", e))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    println!(
        "  Analyzing: {}Hz, {}-bit, {} channels",
        spec.sample_rate, spec.bits_per_sample, channels
    );
    println!(
        "  Target band: {:.0}Hz ± {:.0}Hz",
        carrier_freq,
        bandwidth / 2.0
    );

    // Read samples (mono mix)
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f64;
            let all: Vec<f64> = reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f64 / max)
                .collect();
            // Mono mix
            all.chunks(channels)
                .map(|c| c.iter().sum::<f64>() / channels as f64)
                .collect()
        }
        hound::SampleFormat::Float => {
            let all: Vec<f64> = reader.samples::<f32>().map(|s| s.unwrap() as f64).collect();
            all.chunks(channels)
                .map(|c| c.iter().sum::<f64>() / channels as f64)
                .collect()
        }
    };

    // Bandpass filter around carrier frequency using FFT
    // Smaller hop for better time resolution
    let fft_size = 2048;
    let hop_size = 256; // ~5.8ms per sample at 44.1kHz - good for 50ms dots
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let _ifft = planner.plan_fft_inverse(fft_size);

    let freq_lo = carrier_freq - bandwidth / 2.0;
    let freq_hi = carrier_freq + bandwidth / 2.0;
    let bin_lo = (freq_lo * fft_size as f64 / sample_rate) as usize;
    let bin_hi = (freq_hi * fft_size as f64 / sample_rate) as usize;

    println!("  FFT bins: {} - {} (of {})", bin_lo, bin_hi, fft_size / 2);

    // Extract envelope of the target band
    let mut envelope: Vec<f64> = Vec::new();

    for chunk_start in (0..samples.len().saturating_sub(fft_size)).step_by(hop_size) {
        let mut buffer: Vec<Complex<f64>> = samples[chunk_start..chunk_start + fft_size]
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();

        fft.process(&mut buffer);

        // Sum energy in target band
        let mut band_energy = 0.0;
        for i in bin_lo..=bin_hi.min(fft_size / 2) {
            band_energy += buffer[i].norm_sqr();
        }
        envelope.push(band_energy.sqrt());
    }

    if envelope.is_empty() {
        return Err("No audio data to analyze".into());
    }

    // Normalize envelope
    let max_env = envelope.iter().cloned().fold(0.0_f64, f64::max);
    if max_env > 0.0 {
        for e in &mut envelope {
            *e /= max_env;
        }
    }

    // Threshold detection
    let threshold = 0.15;
    let morse_bits: Vec<bool> = envelope.iter().map(|&e| e > threshold).collect();

    // Convert to morse string
    // Each envelope sample represents hop_size samples = hop_size/sample_rate seconds
    let time_per_sample = hop_size as f64 / sample_rate;
    let dot_duration = MORSE_DOT as f64 / 44100.0; // ~50ms
    let dash_duration = MORSE_DASH as f64 / 44100.0; // ~150ms
    let dot_samples = (dot_duration / time_per_sample) as usize;
    let dash_samples = (dash_duration / time_per_sample) as usize;
    let _gap_samples = dot_samples;
    let letter_gap_samples = dash_samples;
    let word_gap_samples = (MORSE_WORD_GAP as f64 / 44100.0 / time_per_sample) as usize;

    println!(
        "  Timing: dot={}samples, dash={}samples, word_gap={}samples",
        dot_samples, dash_samples, word_gap_samples
    );

    // Run-length encode
    let mut runs: Vec<(bool, usize)> = Vec::new();
    if !morse_bits.is_empty() {
        let mut current = morse_bits[0];
        let mut count = 1;
        for &bit in &morse_bits[1..] {
            if bit == current {
                count += 1;
            } else {
                runs.push((current, count));
                current = bit;
                count = 1;
            }
        }
        runs.push((current, count));
    }

    // Decode runs to morse symbols
    let mut morse_string = String::new();

    for (is_tone, length) in runs {
        if is_tone {
            // Tone: dot or dash?
            if length >= (dot_samples + dash_samples) / 2 {
                morse_string.push('-');
            } else {
                morse_string.push('.');
            }
        } else {
            // Silence: element gap, letter gap, or word gap?
            if length >= word_gap_samples / 2 {
                morse_string.push_str("   "); // Word gap
            } else if length >= letter_gap_samples / 2 {
                morse_string.push(' '); // Letter gap
            }
            // Element gap: no separator needed
        }
    }

    println!(
        "  Raw morse: {}",
        &morse_string[..morse_string.len().min(80)]
    );

    // Decode morse to text
    let decoded = decode_morse_to_text(&morse_string);

    Ok(decoded)
}

/// Convert morse string to text
fn decode_morse_to_text(morse: &str) -> String {
    let mut result = String::new();

    // Split by word gaps (3+ spaces)
    for word in morse.split("   ") {
        if !result.is_empty() && !result.ends_with(' ') {
            result.push(' ');
        }
        // Split by letter gaps
        for letter in word.split_whitespace() {
            if let Some(c) = morse_to_char(letter) {
                result.push(c);
            }
        }
    }

    result
}

/// Convert single morse sequence to character
fn morse_to_char(morse: &str) -> Option<char> {
    match morse {
        ".-" => Some('A'),
        "-..." => Some('B'),
        "-.-." => Some('C'),
        "-.." => Some('D'),
        "." => Some('E'),
        "..-." => Some('F'),
        "--." => Some('G'),
        "...." => Some('H'),
        ".." => Some('I'),
        ".---" => Some('J'),
        "-.-" => Some('K'),
        ".-.." => Some('L'),
        "--" => Some('M'),
        "-." => Some('N'),
        "---" => Some('O'),
        ".--." => Some('P'),
        "--.-" => Some('Q'),
        ".-." => Some('R'),
        "..." => Some('S'),
        "-" => Some('T'),
        "..-" => Some('U'),
        "...-" => Some('V'),
        ".--" => Some('W'),
        "-..-" => Some('X'),
        "-.--" => Some('Y'),
        "--.." => Some('Z'),
        "-----" => Some('0'),
        ".----" => Some('1'),
        "..---" => Some('2'),
        "...--" => Some('3'),
        "....-" => Some('4'),
        "....." => Some('5'),
        "-...." => Some('6'),
        "--..." => Some('7'),
        "---.." => Some('8'),
        "----." => Some('9'),
        "--..--" => Some(','),
        ".-.-.-" => Some('.'),
        _ => None,
    }
}
