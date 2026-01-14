//! ═══════════════════════════════════════════════════════════════════════════════
//! ENVIRONMENTAL TELEMETRY — Real Sensor Data
//! ═══════════════════════════════════════════════════════════════════════════════
//! Not text about the world. Actual data from the world.
//! This is what LLMs lack - they only know what they've read.
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::observations::{Observation, ObservationBatch};
use crate::stats::Ewma;
use crate::time::TimePoint;
use std::collections::HashMap;

/// Types of environmental sensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorType {
    /// System resource sensors
    CpuLoad,
    MemoryUsage,
    DiskIO,
    NetworkIO,

    /// Temporal sensors
    WallClock,
    ProcessUptime,
    SystemUptime,

    /// External data sensors
    ApiLatency,
    ExternalFeed,
    DatabaseState,

    /// Physical sensors (when available)
    Temperature,
    PowerDraw,

    /// Meta sensors
    ErrorRate,
    ThroughputRate,
    QueueDepth,
}

/// A single sensor reading
#[derive(Debug, Clone)]
pub struct SensorReading {
    pub sensor_type: SensorType,
    pub value: f64,
    pub unit: String,
    pub confidence: f64,
    pub timestamp: TimePoint,
    pub is_stale: bool,
}

impl SensorReading {
    pub fn new(sensor_type: SensorType, value: f64, unit: &str) -> Self {
        Self {
            sensor_type,
            value,
            unit: unit.to_string(),
            confidence: 1.0,
            timestamp: TimePoint::now(),
            is_stale: false,
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn mark_stale(&mut self, max_age_ms: u64) {
        let age = TimePoint::now().duration_since(&self.timestamp);
        self.is_stale = age.as_millis() as u64 > max_age_ms;
        if self.is_stale {
            self.confidence *= 0.5; // Reduce confidence for stale data
        }
    }
}

/// Snapshot of the entire environment
#[derive(Debug, Clone)]
pub struct EnvironmentSnapshot {
    pub readings: HashMap<SensorType, SensorReading>,
    pub confidence: f64,
    pub timestamp: TimePoint,
    pub sensor_count: usize,
    pub stale_count: usize,
}

impl Default for EnvironmentSnapshot {
    fn default() -> Self {
        Self {
            readings: HashMap::new(),
            confidence: 0.5,
            timestamp: TimePoint::now(),
            sensor_count: 0,
            stale_count: 0,
        }
    }
}

impl EnvironmentSnapshot {
    pub fn get(&self, sensor: SensorType) -> Option<&SensorReading> {
        self.readings.get(&sensor)
    }

    pub fn value(&self, sensor: SensorType) -> Option<f64> {
        self.readings.get(&sensor).map(|r| r.value)
    }
}

/// Configuration for environmental telemetry
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Maximum age before marking sensor data stale (ms)
    pub stale_threshold_ms: u64,
    /// Smoothing factor for sensor value averaging
    pub smoothing_alpha: f64,
    /// Minimum confidence to include in snapshot
    pub min_confidence: f64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            stale_threshold_ms: 5000,
            smoothing_alpha: 0.1,
            min_confidence: 0.3,
        }
    }
}

/// Telemetry reading result
#[derive(Debug, Clone)]
pub struct TelemetryReading {
    pub snapshot: EnvironmentSnapshot,
    pub delta: HashMap<SensorType, f64>,
    pub anomalies: Vec<SensorAnomaly>,
}

#[derive(Debug, Clone)]
pub struct SensorAnomaly {
    pub sensor: SensorType,
    pub expected: f64,
    pub actual: f64,
    pub severity: f64,
}

/// Environmental Telemetry System
pub struct EnvironmentalTelemetry {
    config: TelemetryConfig,
    sensors: HashMap<SensorType, SensorState>,
    last_snapshot: EnvironmentSnapshot,
}

struct SensorState {
    ewma: Ewma,
    last_value: f64,
    reading_count: u64,
    last_update: TimePoint,
}

impl EnvironmentalTelemetry {
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            config,
            sensors: HashMap::new(),
            last_snapshot: EnvironmentSnapshot::default(),
        }
    }

    /// Process observations and update telemetry
    pub fn process(&mut self, observations: &ObservationBatch) -> EnvironmentSnapshot {
        let now = TimePoint::now();

        // Extract sensor readings from observations
        for obs in observations.iter() {
            if let Some((sensor_type, value)) = self.parse_observation(obs) {
                self.update_sensor(sensor_type, value, now);
            }
        }

        // Also collect system metrics directly
        self.collect_system_metrics(now);

        // Build snapshot
        self.build_snapshot(now)
    }

    fn parse_observation(&self, obs: &Observation) -> Option<(SensorType, f64)> {
        // Map observation keys to sensor types
        use crate::observations::ObsKey;

        let sensor_type = match obs.key {
            ObsKey::ThermalUtilization => Some(SensorType::CpuLoad),
            ObsKey::RespLatMs => Some(SensorType::ApiLatency),
            ObsKey::NetRttMs => Some(SensorType::NetworkIO),
            ObsKey::PainIntensity => Some(SensorType::ErrorRate),
            ObsKey::CtxUtilization => Some(SensorType::MemoryUsage),
            _ => None,
        }?;

        // ObsValue is a struct with a value field
        let value = obs.value.value;

        Some((sensor_type, value))
    }

    fn update_sensor(&mut self, sensor_type: SensorType, value: f64, now: TimePoint) {
        let state = self
            .sensors
            .entry(sensor_type)
            .or_insert_with(|| SensorState {
                ewma: Ewma::new(self.config.smoothing_alpha),
                last_value: value,
                reading_count: 0,
                last_update: now,
            });

        state.ewma.update(value);
        state.last_value = value;
        state.reading_count += 1;
        state.last_update = now;
    }

    fn collect_system_metrics(&mut self, now: TimePoint) {
        // Collect actual system metrics using sysinfo
        // This is the grounding - real data, not text about data

        #[cfg(not(target_arch = "wasm32"))]
        {
            use sysinfo::System;

            let mut sys = System::new_all();
            sys.refresh_all();

            // CPU load - average across all CPUs
            let cpus = sys.cpus();
            let cpu_usage = if !cpus.is_empty() {
                cpus.iter().map(|c| c.cpu_usage() as f64).sum::<f64>() / cpus.len() as f64
            } else {
                0.0
            };
            self.update_sensor(SensorType::CpuLoad, cpu_usage, now);

            // Memory usage
            let total_mem = sys.total_memory() as f64;
            let used_mem = sys.used_memory() as f64;
            let mem_pct = if total_mem > 0.0 {
                (used_mem / total_mem) * 100.0
            } else {
                0.0
            };
            self.update_sensor(SensorType::MemoryUsage, mem_pct, now);

            // System uptime
            let uptime = System::uptime() as f64;
            self.update_sensor(SensorType::SystemUptime, uptime, now);
        }

        // Wall clock (always available)
        let wall_clock = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.update_sensor(SensorType::WallClock, wall_clock, now);
    }

    fn build_snapshot(&mut self, now: TimePoint) -> EnvironmentSnapshot {
        let mut readings = HashMap::new();
        let mut stale_count = 0;

        for (sensor_type, state) in &self.sensors {
            let mut reading = SensorReading::new(
                *sensor_type,
                state.ewma.value(),
                self.unit_for_sensor(*sensor_type),
            );

            // Check staleness
            let age_ms = now.duration_since(&state.last_update).as_millis() as u64;
            if age_ms > self.config.stale_threshold_ms {
                reading.is_stale = true;
                reading.confidence *= 0.5;
                stale_count += 1;
            }

            // Confidence based on reading count
            let count_confidence = (state.reading_count as f64 / 100.0).min(1.0);
            reading.confidence *= count_confidence;

            if reading.confidence >= self.config.min_confidence {
                readings.insert(*sensor_type, reading);
            }
        }

        let sensor_count = readings.len();
        let confidence = if sensor_count > 0 {
            let total_conf: f64 = readings.values().map(|r| r.confidence).sum();
            (total_conf / sensor_count as f64)
                * (1.0 - (stale_count as f64 / sensor_count as f64) * 0.5)
        } else {
            0.0
        };

        let snapshot = EnvironmentSnapshot {
            readings,
            confidence,
            timestamp: now,
            sensor_count,
            stale_count,
        };

        self.last_snapshot = snapshot.clone();
        snapshot
    }

    fn unit_for_sensor(&self, sensor: SensorType) -> &'static str {
        match sensor {
            SensorType::CpuLoad => "%",
            SensorType::MemoryUsage => "%",
            SensorType::DiskIO => "bytes/s",
            SensorType::NetworkIO => "bytes/s",
            SensorType::WallClock => "unix_seconds",
            SensorType::ProcessUptime => "seconds",
            SensorType::SystemUptime => "seconds",
            SensorType::ApiLatency => "ms",
            SensorType::ExternalFeed => "items",
            SensorType::DatabaseState => "connections",
            SensorType::Temperature => "celsius",
            SensorType::PowerDraw => "watts",
            SensorType::ErrorRate => "errors/s",
            SensorType::ThroughputRate => "ops/s",
            SensorType::QueueDepth => "items",
        }
    }

    /// Get the last snapshot
    pub fn last_snapshot(&self) -> &EnvironmentSnapshot {
        &self.last_snapshot
    }

    /// Check if a specific sensor is available and fresh
    pub fn sensor_available(&self, sensor: SensorType) -> bool {
        self.sensors
            .get(&sensor)
            .map(|s| {
                let age = TimePoint::now().duration_since(&s.last_update);
                age.as_millis() as u64 <= self.config.stale_threshold_ms
            })
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_creation() {
        let telemetry = EnvironmentalTelemetry::new(TelemetryConfig::default());
        let snapshot = telemetry.last_snapshot();
        assert_eq!(snapshot.sensor_count, 0);
    }

    #[test]
    fn test_sensor_reading() {
        let reading = SensorReading::new(SensorType::CpuLoad, 45.5, "%");
        assert_eq!(reading.sensor_type, SensorType::CpuLoad);
        assert!((reading.value - 45.5).abs() < 0.01);
        assert!(!reading.is_stale);
    }
}
