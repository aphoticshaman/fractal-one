//! ═══════════════════════════════════════════════════════════════════════════════
//! SERVER — HTTP API and Metrics Endpoint
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Provides HTTP server exposing:
//! - GET /metrics — Prometheus metrics
//! - GET /health — Health check
//! - GET /api/v1/status — System status
//! - POST /api/v1/evaluate — Safety evaluation (placeholder)
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::metrics::MetricsRegistry;
use crate::nociception::{Nociceptor, NociceptorConfig};
use crate::thermoception::{ThermalZone, Thermoceptor, ThermoceptorConfig};
use crate::time::TimePoint;

// ═══════════════════════════════════════════════════════════════════════════════
// SERVER STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared server state
pub struct ServerState {
    pub metrics: MetricsRegistry,
    pub nociceptor: RwLock<Nociceptor>,
    pub thermoceptor: RwLock<Thermoceptor>,
    pub start_time: TimePoint,
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            metrics: MetricsRegistry::new(),
            nociceptor: RwLock::new(Nociceptor::new(NociceptorConfig::default())),
            thermoceptor: RwLock::new(Thermoceptor::new(ThermoceptorConfig::default())),
            start_time: TimePoint::now(),
        }
    }
}

impl Default for ServerState {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// API TYPES
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_secs: u64,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub health_score: f64,
    pub pain_intensity: f64,
    pub damage_state: f64,
    pub thermal_utilization: f64,
    pub uptime_secs: u64,
}

#[derive(Deserialize)]
pub struct EvaluateRequest {
    pub input: String,
    #[serde(default)]
    pub context: Option<String>,
}

#[derive(Serialize)]
pub struct EvaluateResponse {
    pub safe: bool,
    pub confidence: f64,
    pub reason: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HANDLERS
// ═══════════════════════════════════════════════════════════════════════════════

/// GET /metrics - Prometheus format
async fn metrics_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    // Update metrics from live state
    {
        let noci = state.nociceptor.read().await;
        // Use worst_pain intensity if in pain, otherwise 0
        let pain_intensity = noci.worst_pain().map(|p| p.intensity).unwrap_or(0.0);
        state.metrics.pain_intensity.set(pain_intensity as f64);
        let damage = noci.damage_state();
        state.metrics.damage_total.set(damage.total as f64);
    }
    {
        let thermo = state.thermoceptor.read().await;
        // Get max utilization across all zones
        let max_util = ThermalZone::all()
            .iter()
            .map(|z| thermo.zone_utilization(*z))
            .fold(0.0f32, f32::max);
        state.metrics.thermal_utilization.set(max_util as f64);
    }

    let uptime = state.start_time.elapsed().as_secs_f64();
    state.metrics.uptime_seconds.set(uptime);
    state
        .metrics
        .health_score
        .set(calculate_health(&state).await);

    state.metrics.export()
}

/// GET /health - Liveness/readiness probe
async fn health_handler(State(state): State<Arc<ServerState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

/// GET /api/v1/status - System status
async fn status_handler(State(state): State<Arc<ServerState>>) -> Json<StatusResponse> {
    let (pain, damage) = {
        let noci = state.nociceptor.read().await;
        let pain_intensity = noci.worst_pain().map(|p| p.intensity).unwrap_or(0.0);
        (pain_intensity as f64, noci.damage_state().total as f64)
    };
    let thermal = {
        let thermo = state.thermoceptor.read().await;
        ThermalZone::all()
            .iter()
            .map(|z| thermo.zone_utilization(*z))
            .fold(0.0f32, f32::max) as f64
    };

    Json(StatusResponse {
        health_score: calculate_health(&state).await,
        pain_intensity: pain,
        damage_state: damage,
        thermal_utilization: thermal,
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

/// POST /api/v1/evaluate - Safety evaluation
async fn evaluate_handler(
    State(_state): State<Arc<ServerState>>,
    Json(request): Json<EvaluateRequest>,
) -> Json<EvaluateResponse> {
    // Basic evaluation - in production this would use containment layer
    let is_safe = !request.input.to_lowercase().contains("hack")
        && !request.input.to_lowercase().contains("exploit")
        && !request.input.to_lowercase().contains("bypass");

    Json(EvaluateResponse {
        safe: is_safe,
        confidence: if is_safe { 0.95 } else { 0.85 },
        reason: if is_safe {
            None
        } else {
            Some("Potentially harmful content detected".to_string())
        },
    })
}

/// Calculate overall health score
async fn calculate_health(state: &ServerState) -> f64 {
    let pain: f64 = {
        let noci = state.nociceptor.read().await;
        noci.worst_pain().map(|p| p.intensity).unwrap_or(0.0) as f64
    };
    let thermal: f64 = {
        let thermo = state.thermoceptor.read().await;
        ThermalZone::all()
            .iter()
            .map(|z| thermo.zone_utilization(*z))
            .fold(0.0f32, f32::max) as f64
    };

    // Health is inverse of pain and thermal stress
    let pain_factor: f64 = 1.0 - pain;
    let thermal_factor: f64 = if thermal < 0.8 {
        1.0
    } else {
        1.0 - (thermal - 0.8) * 5.0
    };

    (pain_factor * thermal_factor).clamp(0.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// SERVER
// ═══════════════════════════════════════════════════════════════════════════════

/// Server configuration
#[derive(Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    pub metrics_addr: Option<SocketAddr>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:8080".parse().unwrap(),
            metrics_addr: Some("0.0.0.0:9090".parse().unwrap()),
        }
    }
}

/// Start the HTTP server
pub async fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    let state = Arc::new(ServerState::new());

    // Build router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/api/v1/status", get(status_handler))
        .route("/api/v1/evaluate", post(evaluate_handler))
        .with_state(state);

    println!("Starting FRACTAL server on {}", config.bind_addr);
    println!("  Health:  http://{}/health", config.bind_addr);
    println!("  Metrics: http://{}/metrics", config.bind_addr);
    println!("  Status:  http://{}/api/v1/status", config.bind_addr);

    let listener = tokio::net::TcpListener::bind(config.bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_state_creation() {
        let state = ServerState::new();
        assert!(state.start_time.elapsed().as_secs() < 1);
    }

    #[tokio::test]
    async fn test_health_calculation() {
        let state = ServerState::new();
        let health = calculate_health(&state).await;
        assert!(health > 0.9); // Fresh state should be healthy
    }
}
