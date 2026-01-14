//! ═══════════════════════════════════════════════════════════════════════════════
//! NOCI_PULSE — Nociception ↔ NeuroLink Integration
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Encodes nociception damage state into Pulse payload for IPC transmission.
//! Uses payload[8..16] for damage telemetry while preserving payload[0..8] for
//! existing jitter history.
//!
//! Payload layout:
//! - `[0..8]`   - Jitter history (existing)
//! - `[8]`      - total_damage (0.0-1.0)
//! - `[9]`      - worst_zone_damage (0.0-1.0)
//! - `[10]`     - pain_count (active pain signals)
//! - `[11]`     - in_pain flag (0.0 or 1.0)
//! - `[12]`     - sensitivity (1.0 = normal)
//! - `[13]`     - damage_velocity (rate of change)
//! - `[14..16]` - reserved for thermoception integration
//! - `[16..32]` - zone-specific damage (up to 16 zones)
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::neuro_link::Pulse;
use crate::nociception::Nociceptor;
use crate::stats::float_cmp_f32;

/// Payload indices for nociception data
pub mod payload_idx {
    // Jitter history: 0-7 (existing)
    pub const TOTAL_DAMAGE: usize = 8;
    pub const WORST_ZONE_DAMAGE: usize = 9;
    pub const PAIN_COUNT: usize = 10;
    pub const IN_PAIN: usize = 11;
    pub const SENSITIVITY: usize = 12;
    pub const DAMAGE_VELOCITY: usize = 13;
    // Reserved for thermoception: 14-15
    pub const ZONE_DAMAGE_START: usize = 16;
    pub const ZONE_DAMAGE_END: usize = 32;
}

/// Encoded nociception state for IPC
#[derive(Debug, Clone, Default)]
pub struct NociPulseData {
    pub total_damage: f32,
    pub worst_zone_damage: f32,
    pub pain_count: u32,
    pub in_pain: bool,
    pub sensitivity: f32,
    pub damage_velocity: f32,
    pub zone_damages: Vec<(String, f32)>,
}

impl NociPulseData {
    /// Extract nociception data from a Pulse
    pub fn from_pulse(pulse: &Pulse) -> Self {
        Self {
            total_damage: pulse.payload[payload_idx::TOTAL_DAMAGE],
            worst_zone_damage: pulse.payload[payload_idx::WORST_ZONE_DAMAGE],
            pain_count: pulse.payload[payload_idx::PAIN_COUNT] as u32,
            in_pain: pulse.payload[payload_idx::IN_PAIN] > 0.5,
            sensitivity: pulse.payload[payload_idx::SENSITIVITY],
            damage_velocity: pulse.payload[payload_idx::DAMAGE_VELOCITY],
            zone_damages: Vec::new(), // Zone names not transmitted, only values
        }
    }

    /// Encode nociception data into a Pulse payload
    pub fn encode_into(&self, payload: &mut [f32; 32]) {
        payload[payload_idx::TOTAL_DAMAGE] = self.total_damage;
        payload[payload_idx::WORST_ZONE_DAMAGE] = self.worst_zone_damage;
        payload[payload_idx::PAIN_COUNT] = self.pain_count as f32;
        payload[payload_idx::IN_PAIN] = if self.in_pain { 1.0 } else { 0.0 };
        payload[payload_idx::SENSITIVITY] = self.sensitivity;
        payload[payload_idx::DAMAGE_VELOCITY] = self.damage_velocity;

        // Encode zone damages (up to 16)
        for (i, (_, damage)) in self.zone_damages.iter().take(16).enumerate() {
            payload[payload_idx::ZONE_DAMAGE_START + i] = *damage;
        }
    }

    /// Check if this represents a critical damage state
    pub fn is_critical(&self) -> bool {
        self.total_damage > 0.8 || self.worst_zone_damage > 0.9
    }

    /// Check if healthy
    pub fn is_healthy(&self) -> bool {
        self.total_damage < 0.2 && !self.in_pain
    }

    /// Get damage trend
    pub fn trend(&self) -> DamageTrend {
        if self.damage_velocity > 0.1 {
            DamageTrend::Worsening
        } else if self.damage_velocity < -0.05 {
            DamageTrend::Recovering
        } else {
            DamageTrend::Stable
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DamageTrend {
    Worsening,
    Stable,
    Recovering,
}

/// Trait for types that can be encoded into pulse payload
pub trait PulseEncodable {
    fn encode_to_payload(&self, payload: &mut [f32; 32]);
}

/// Encoder that tracks state for velocity calculation
pub struct NociPulseEncoder {
    last_total_damage: f32,
    last_timestamp_ms: u64,
}

impl NociPulseEncoder {
    pub fn new() -> Self {
        Self {
            last_total_damage: 0.0,
            last_timestamp_ms: 0,
        }
    }

    /// Encode nociceptor state into pulse payload
    pub fn encode(&mut self, nociceptor: &Nociceptor, timestamp_ms: u64, payload: &mut [f32; 32]) {
        let damage_state = nociceptor.damage_state();

        // Calculate velocity
        let dt_ms = timestamp_ms.saturating_sub(self.last_timestamp_ms);
        let velocity = if dt_ms > 0 {
            (damage_state.total - self.last_total_damage) / (dt_ms as f32 / 1000.0)
        } else {
            0.0
        };

        // Update tracking
        self.last_total_damage = damage_state.total;
        self.last_timestamp_ms = timestamp_ms;

        // Find worst zone damage
        let worst_zone_damage = damage_state
            .by_location
            .values()
            .copied()
            .fold(0.0f32, f32::max);

        // Encode
        payload[payload_idx::TOTAL_DAMAGE] = damage_state.total;
        payload[payload_idx::WORST_ZONE_DAMAGE] = worst_zone_damage;
        payload[payload_idx::PAIN_COUNT] = 0.0; // Would need access to active_pains
        payload[payload_idx::IN_PAIN] = if nociceptor.in_pain() { 1.0 } else { 0.0 };
        payload[payload_idx::SENSITIVITY] = 1.0; // Default, could expose from Nociceptor
        payload[payload_idx::DAMAGE_VELOCITY] = velocity;

        // Encode per-zone damage (sorted by damage level)
        let mut zones: Vec<_> = damage_state.by_location.iter().collect();
        zones.sort_by(|a, b| float_cmp_f32(b.1, a.1));

        for (i, (_, &damage)) in zones.iter().take(16).enumerate() {
            payload[payload_idx::ZONE_DAMAGE_START + i] = damage;
        }
    }
}

impl Default for NociPulseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Decoder for receiving nociception data from pulses
pub struct NociPulseDecoder;

impl NociPulseDecoder {
    /// Decode nociception data from pulse
    pub fn decode(pulse: &Pulse) -> NociPulseData {
        NociPulseData::from_pulse(pulse)
    }

    /// Check if pulse indicates critical damage
    pub fn is_critical(pulse: &Pulse) -> bool {
        pulse.payload[payload_idx::TOTAL_DAMAGE] > 0.8
            || pulse.payload[payload_idx::WORST_ZONE_DAMAGE] > 0.9
    }

    /// Check if pulse indicates active pain
    pub fn in_pain(pulse: &Pulse) -> bool {
        pulse.payload[payload_idx::IN_PAIN] > 0.5
    }

    /// Get damage velocity from pulse
    pub fn damage_velocity(pulse: &Pulse) -> f32 {
        pulse.payload[payload_idx::DAMAGE_VELOCITY]
    }
}

/// Helper to create a pulse with nociception data
pub fn create_noci_pulse(
    id: u64,
    telemetry_sequence: u64,
    jitter_ms: f64,
    cpu_load_percent: f64,
    current_interval_ms: u64,
    nociceptor: &Nociceptor,
    encoder: &mut NociPulseEncoder,
) -> Pulse {
    let mut payload = [0.0f32; 32];

    // Encode nociception data
    encoder.encode(nociceptor, id, &mut payload);

    Pulse {
        id,
        telemetry_sequence,
        jitter_ms,
        cpu_load_percent,
        current_interval_ms,
        bad_actor_id: 0,
        entropy_damping: 0.0,
        payload,
        scheduler_override: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let data = NociPulseData {
            total_damage: 0.5,
            worst_zone_damage: 0.7,
            pain_count: 3,
            in_pain: true,
            sensitivity: 1.2,
            damage_velocity: 0.05,
            zone_damages: vec![("zone_a".to_string(), 0.3), ("zone_b".to_string(), 0.7)],
        };

        let mut payload = [0.0f32; 32];
        data.encode_into(&mut payload);

        let pulse = Pulse {
            id: 1,
            telemetry_sequence: 1,
            jitter_ms: 0.0,
            cpu_load_percent: 0.0,
            current_interval_ms: 80,
            bad_actor_id: 0,
            entropy_damping: 0.0,
            payload,
            scheduler_override: 0,
        };

        let decoded = NociPulseData::from_pulse(&pulse);

        assert!((decoded.total_damage - 0.5).abs() < 0.001);
        assert!((decoded.worst_zone_damage - 0.7).abs() < 0.001);
        assert_eq!(decoded.pain_count, 3);
        assert!(decoded.in_pain);
    }

    #[test]
    fn test_critical_detection() {
        let mut payload = [0.0f32; 32];
        payload[payload_idx::TOTAL_DAMAGE] = 0.85;

        let pulse = Pulse {
            id: 1,
            telemetry_sequence: 1,
            jitter_ms: 0.0,
            cpu_load_percent: 0.0,
            current_interval_ms: 80,
            bad_actor_id: 0,
            entropy_damping: 0.0,
            payload,
            scheduler_override: 0,
        };

        assert!(NociPulseDecoder::is_critical(&pulse));
    }
}
