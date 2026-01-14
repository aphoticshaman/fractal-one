//! ═══════════════════════════════════════════════════════════════════════════════
//! VISUAL QUALIA — Webcam Integration
//! ═══════════════════════════════════════════════════════════════════════════════
//! Gives fractal_one the ability to "see" - luminance, motion detection,
//! presence inference, and environmental awareness.
//! ═══════════════════════════════════════════════════════════════════════════════

use crossbeam_channel::{bounded, Receiver, Sender};
use image::{GrayImage, ImageBuffer, Luma, Rgb};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Visual qualia state - what the system "sees"
#[derive(Debug, Clone, Default)]
pub struct Visual {
    /// Average luminance (0.0 = black, 1.0 = white)
    pub luminance: f64,

    /// Rate of luminance change per second
    pub luminance_delta: f64,

    /// Motion magnitude (frame difference intensity)
    pub motion_magnitude: f64,

    /// Presence confidence (simple motion-based, 0.0-1.0)
    pub presence_confidence: f64,

    /// Estimated color temperature (2700K warm, 6500K cool)
    pub color_temp_kelvin: f64,

    /// Frame entropy (visual complexity/information)
    pub frame_entropy: f64,

    /// Frames processed per second
    pub fps: f64,

    /// Timestamp of last update
    pub timestamp: f64,
}

/// Configuration for visual processing
#[derive(Debug, Clone)]
pub struct VisualConfig {
    /// Target FPS (lower = less CPU)
    pub target_fps: u32,

    /// Motion threshold for presence detection
    pub motion_threshold: f64,

    /// Presence decay rate per second (without motion)
    pub presence_decay: f64,

    /// Presence growth rate per second (with motion)
    pub presence_growth: f64,

    /// Downscale factor for processing (higher = faster, less accurate)
    pub downscale: u32,

    /// Camera index (0 = default)
    pub camera_index: u32,
}

impl Default for VisualConfig {
    fn default() -> Self {
        Self {
            target_fps: 10, // 10 fps is plenty for presence detection
            motion_threshold: 0.02,
            presence_decay: 0.1,  // Lose presence confidence over 10 seconds
            presence_growth: 0.5, // Gain presence confidence over 2 seconds
            downscale: 4,         // Process at 1/4 resolution
            camera_index: 0,
        }
    }
}

/// Visual frame broadcast for external consumers (like CV processing)
#[derive(Clone)]
pub struct VisualBroadcast {
    /// Grayscale frame data
    pub gray_frame: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Current luminance
    pub luminance: f64,
    /// Motion detected
    pub motion_detected: bool,
}

/// Visual processor - runs in background thread
pub struct VisualProcessor {
    config: VisualConfig,
    state: Arc<RwLock<Visual>>,
    /// Broadcast channel for external consumers
    broadcast_tx: Option<Sender<VisualBroadcast>>,
}

impl VisualProcessor {
    /// Create new visual processor
    pub fn new(config: VisualConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(Visual::default())),
            broadcast_tx: None,
        }
    }

    /// Subscribe to visual frame broadcast (for CV processing, etc)
    pub fn subscribe_broadcast(&mut self) -> Receiver<VisualBroadcast> {
        let (tx, rx) = bounded(16);
        self.broadcast_tx = Some(tx);
        rx
    }

    /// Get shared state handle for reading from other threads
    pub fn state_handle(&self) -> Arc<RwLock<Visual>> {
        Arc::clone(&self.state)
    }

    /// Start video capture from webcam
    pub fn start(&self) -> Result<(), VideoError> {
        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let broadcast_tx = self.broadcast_tx.clone();

        std::thread::Builder::new()
            .name("visual-processor".into())
            .spawn(move || {
                if let Err(e) = Self::capture_loop(state, config, broadcast_tx) {
                    eprintln!("[visual] Capture loop error: {}", e);
                }
            })
            .map_err(|e| VideoError::ThreadError(e.to_string()))?;

        Ok(())
    }

    /// Main capture loop (runs in dedicated thread)
    fn capture_loop(
        state: Arc<RwLock<Visual>>,
        config: VisualConfig,
        broadcast_tx: Option<Sender<VisualBroadcast>>,
    ) -> Result<(), VideoError> {
        // Initialize camera
        let index = CameraIndex::Index(config.camera_index);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        let mut camera =
            Camera::new(index, requested).map_err(|e| VideoError::CameraInit(e.to_string()))?;

        camera
            .open_stream()
            .map_err(|e| VideoError::StreamOpen(e.to_string()))?;

        let frame_duration = Duration::from_secs_f64(1.0 / config.target_fps as f64);

        let mut prev_gray: Option<GrayImage> = None;
        let mut prev_luminance = 0.5_f64;
        let mut presence_confidence = 0.0_f64;
        let mut last_frame_time = Instant::now();
        let mut frame_count = 0_u32;
        let mut fps_timer = Instant::now();
        let mut current_fps = 0.0_f64;

        loop {
            let frame_start = Instant::now();

            // Capture frame
            let frame = match camera.frame() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[visual] Frame capture error: {}", e);
                    std::thread::sleep(frame_duration);
                    continue;
                }
            };

            // Decode to image buffer
            let decoded = match frame.decode_image::<RgbFormat>() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[visual] Frame decode error: {}", e);
                    std::thread::sleep(frame_duration);
                    continue;
                }
            };

            // Downscale for processing
            let (w, h) = (
                decoded.width() / config.downscale,
                decoded.height() / config.downscale,
            );

            let resized =
                image::imageops::resize(&decoded, w, h, image::imageops::FilterType::Nearest);

            // Convert to grayscale
            let gray: GrayImage = image::imageops::grayscale(&resized);

            // ══════════════════════════════════════════════════════════════
            // LUMINANCE CALCULATION
            // ══════════════════════════════════════════════════════════════
            let luminance: f64 =
                gray.pixels().map(|p| p.0[0] as f64).sum::<f64>() / (w * h) as f64 / 255.0;

            // ══════════════════════════════════════════════════════════════
            // LUMINANCE DELTA
            // ══════════════════════════════════════════════════════════════
            let dt = last_frame_time.elapsed().as_secs_f64().max(0.001);
            let luminance_delta = (luminance - prev_luminance) / dt;
            prev_luminance = luminance;
            last_frame_time = Instant::now();

            // ══════════════════════════════════════════════════════════════
            // MOTION DETECTION
            // ══════════════════════════════════════════════════════════════
            let motion_magnitude = if let Some(ref prev) = prev_gray {
                let total_diff: f64 = gray
                    .pixels()
                    .zip(prev.pixels())
                    .map(|(curr, prev)| (curr.0[0] as i32 - prev.0[0] as i32).abs() as f64)
                    .sum();
                total_diff / (w * h) as f64 / 255.0
            } else {
                0.0
            };

            // ══════════════════════════════════════════════════════════════
            // PRESENCE DETECTION
            // ══════════════════════════════════════════════════════════════
            if motion_magnitude > config.motion_threshold {
                // Motion detected → increase presence
                presence_confidence = (presence_confidence + config.presence_growth * dt).min(1.0);
            } else {
                // No motion → decay presence
                presence_confidence = (presence_confidence - config.presence_decay * dt).max(0.0);
            }

            // ══════════════════════════════════════════════════════════════
            // COLOR TEMPERATURE ESTIMATION
            // ══════════════════════════════════════════════════════════════
            // Simple estimation based on R/B ratio
            let (r_sum, b_sum) = resized.pixels().fold((0.0_f64, 0.0_f64), |(r, b), p| {
                (r + p.0[0] as f64, b + p.0[2] as f64)
            });
            let rb_ratio = if b_sum > 0.0 { r_sum / b_sum } else { 1.0 };
            // Map ratio to approximate Kelvin (very rough)
            let color_temp_kelvin = 6500.0 - (rb_ratio - 1.0) * 2000.0;
            let color_temp_kelvin = color_temp_kelvin.clamp(2700.0, 10000.0);

            // ══════════════════════════════════════════════════════════════
            // FRAME ENTROPY (Visual Complexity)
            // ══════════════════════════════════════════════════════════════
            let frame_entropy = Self::calculate_entropy(&gray);

            // ══════════════════════════════════════════════════════════════
            // FPS TRACKING
            // ══════════════════════════════════════════════════════════════
            frame_count += 1;
            if fps_timer.elapsed() >= Duration::from_secs(1) {
                current_fps = frame_count as f64;
                frame_count = 0;
                fps_timer = Instant::now();
            }

            // ══════════════════════════════════════════════════════════════
            // UPDATE SHARED STATE
            // ══════════════════════════════════════════════════════════════
            {
                let mut state = state.write();
                state.luminance = luminance;
                state.luminance_delta = luminance_delta;
                state.motion_magnitude = motion_magnitude;
                state.presence_confidence = presence_confidence;
                state.color_temp_kelvin = color_temp_kelvin;
                state.frame_entropy = frame_entropy;
                state.fps = current_fps;
                state.timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
            }

            // Broadcast frame to external consumers
            if let Some(ref tx) = broadcast_tx {
                let gray_ref = prev_gray.as_ref().unwrap_or(&gray);
                let _ = tx.try_send(VisualBroadcast {
                    gray_frame: gray_ref.as_raw().clone(),
                    width: w,
                    height: h,
                    luminance,
                    motion_detected: motion_magnitude > config.motion_threshold,
                });
            }

            // Store for next frame comparison
            prev_gray = Some(gray);

            // Rate limit
            let elapsed = frame_start.elapsed();
            if elapsed < frame_duration {
                std::thread::sleep(frame_duration - elapsed);
            }
        }
    }

    /// Calculate Shannon entropy of grayscale image
    fn calculate_entropy(gray: &GrayImage) -> f64 {
        let mut histogram = [0_u32; 256];
        let total = gray.pixels().count() as f64;

        for p in gray.pixels() {
            histogram[p.0[0] as usize] += 1;
        }

        histogram
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.log2()
            })
            .sum()
    }

    /// Get current visual state snapshot
    pub fn read(&self) -> Visual {
        self.state.read().clone()
    }

    /// Create a synthetic test frame (for testing without camera)
    pub fn create_test_frame(width: u32, height: u32, luminance: u8) -> GrayImage {
        ImageBuffer::from_pixel(width, height, Luma([luminance]))
    }

    /// Create an RGB test frame
    pub fn create_rgb_test_frame(
        width: u32,
        height: u32,
        r: u8,
        g: u8,
        b: u8,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        ImageBuffer::from_pixel(width, height, Rgb([r, g, b]))
    }

    /// Calculate entropy of a grayscale image (exposed for external use)
    pub fn entropy(gray: &GrayImage) -> f64 {
        Self::calculate_entropy(gray)
    }
}

/// Video errors
#[derive(Debug)]
pub enum VideoError {
    CameraInit(String),
    StreamOpen(String),
    ThreadError(String),
}

impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CameraInit(e) => write!(f, "Camera init error: {}", e),
            Self::StreamOpen(e) => write!(f, "Stream open error: {}", e),
            Self::ThreadError(e) => write!(f, "Video thread error: {}", e),
        }
    }
}

impl std::error::Error for VideoError {}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_default() {
        let v = Visual::default();
        assert_eq!(v.luminance, 0.0);
        assert_eq!(v.presence_confidence, 0.0);
    }

    #[test]
    fn test_config_default() {
        let c = VisualConfig::default();
        assert_eq!(c.target_fps, 10);
        assert!(c.motion_threshold > 0.0);
    }

    #[test]
    fn test_entropy_calculation() {
        // Uniform image should have low entropy
        let uniform: GrayImage = ImageBuffer::from_pixel(10, 10, Luma([128u8]));
        let entropy = VisualProcessor::calculate_entropy(&uniform);
        assert!(entropy < 0.1);
    }
}
