//! ═══════════════════════════════════════════════════════════════════════════════
//! TIME — Dual-clock TimePoint for Monotonic + Wall Time
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Why two clocks:
//! - Monotonic (Instant): Never goes backward, immune to NTP jumps, for duration math
//! - Wall (SystemTime): Human-readable, correlates with external events, can jump
//!
//! Skew between them indicates external clock manipulation or system issues.
//! ═══════════════════════════════════════════════════════════════════════════════

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A point in time with both monotonic and wall-clock representations
#[derive(Debug, Clone, Copy)]
pub struct TimePoint {
    /// Monotonic clock (for durations, never goes backward)
    pub mono: Instant,
    /// Wall clock (for correlation with external events)
    pub wall: SystemTime,
}

impl TimePoint {
    /// Capture current time on both clocks
    pub fn now() -> Self {
        Self {
            mono: Instant::now(),
            wall: SystemTime::now(),
        }
    }

    /// Create from components (for testing/reconstruction)
    pub fn from_parts(mono: Instant, wall: SystemTime) -> Self {
        Self { mono, wall }
    }

    /// Elapsed time since this point (monotonic)
    pub fn elapsed(&self) -> Duration {
        self.mono.elapsed()
    }

    /// Duration since another TimePoint (monotonic)
    pub fn duration_since(&self, earlier: &TimePoint) -> Duration {
        self.mono.duration_since(earlier.mono)
    }

    /// Wall clock as Unix timestamp (seconds)
    pub fn unix_secs(&self) -> u64 {
        self.wall
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Wall clock as Unix timestamp (milliseconds)
    pub fn unix_millis(&self) -> u128 {
        self.wall
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }

    /// Compute skew between wall and mono clocks relative to a reference point.
    /// Positive = wall clock running fast, Negative = wall clock running slow.
    pub fn clock_skew(&self, reference: &TimePoint) -> Duration {
        let mono_delta = self.mono.duration_since(reference.mono);
        let wall_delta = self
            .wall
            .duration_since(reference.wall)
            .unwrap_or(Duration::ZERO);

        if wall_delta > mono_delta {
            wall_delta - mono_delta
        } else {
            mono_delta - wall_delta
        }
    }

    /// Signed skew (positive = wall fast, negative = wall slow)
    pub fn clock_skew_signed(&self, reference: &TimePoint) -> f64 {
        let mono_delta = self.mono.duration_since(reference.mono).as_secs_f64();
        let wall_delta = self
            .wall
            .duration_since(reference.wall)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        wall_delta - mono_delta
    }
}

impl Default for TimePoint {
    fn default() -> Self {
        Self::now()
    }
}

/// Time interval with start and end points
#[derive(Debug, Clone, Copy)]
pub struct TimeSpan {
    pub start: TimePoint,
    pub end: TimePoint,
}

impl TimeSpan {
    pub fn new(start: TimePoint, end: TimePoint) -> Self {
        Self { start, end }
    }

    pub fn duration(&self) -> Duration {
        self.end.duration_since(&self.start)
    }

    pub fn contains(&self, point: &TimePoint) -> bool {
        point.mono >= self.start.mono && point.mono <= self.end.mono
    }
}

/// Rolling window for time-based aggregation
#[derive(Debug)]
pub struct TimeWindow {
    window_size: Duration,
    reference: TimePoint,
}

impl TimeWindow {
    pub fn new(window_size: Duration) -> Self {
        Self {
            window_size,
            reference: TimePoint::now(),
        }
    }

    /// Check if a point is within the current window
    pub fn contains(&self, point: &TimePoint) -> bool {
        let now = TimePoint::now();
        let window_start = now.mono.checked_sub(self.window_size).unwrap_or(now.mono);
        point.mono >= window_start && point.mono <= now.mono
    }

    /// Get the window start time
    pub fn window_start(&self) -> Instant {
        let now = Instant::now();
        now.checked_sub(self.window_size).unwrap_or(now)
    }

    /// Reset reference point
    pub fn reset(&mut self) {
        self.reference = TimePoint::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timepoint_elapsed() {
        let t1 = TimePoint::now();
        thread::sleep(Duration::from_millis(10));
        let elapsed = t1.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_timepoint_duration_since() {
        let t1 = TimePoint::now();
        thread::sleep(Duration::from_millis(10));
        let t2 = TimePoint::now();
        let duration = t2.duration_since(&t1);
        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_clock_skew_minimal() {
        let t1 = TimePoint::now();
        thread::sleep(Duration::from_millis(50));
        let t2 = TimePoint::now();

        // Under normal conditions, skew should be minimal
        let skew = t2.clock_skew(&t1);
        assert!(
            skew < Duration::from_millis(10),
            "Unexpected clock skew: {:?}",
            skew
        );
    }

    #[test]
    fn test_time_window() {
        let window = TimeWindow::new(Duration::from_secs(1));
        let now = TimePoint::now();
        assert!(window.contains(&now));
    }
}
