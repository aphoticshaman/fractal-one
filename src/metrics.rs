//! ═══════════════════════════════════════════════════════════════════════════════
//! METRICS — Prometheus Metrics Exporter
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Exposes all proprioceptive signals as Prometheus metrics for SOC integration:
//! - Nociception: pain intensity, damage state
//! - Thermoception: thermal zones, utilization
//! - Containment: blocked requests, threat levels
//! - Orchestration: consensus state, agent agreement
//! - Authentication: successes, failures, active sessions
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

// ═══════════════════════════════════════════════════════════════════════════════
// METRIC TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Counter that only increases
#[derive(Debug)]
pub struct Counter {
    name: String,
    help: String,
    value: AtomicU64,
    labels: HashMap<String, String>,
}

impl Counter {
    pub fn new(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            value: AtomicU64::new(0),
            labels: HashMap::new(),
        }
    }

    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    fn format(&self) -> String {
        let labels = self.format_labels();
        format!("{}{} {}", self.name, labels, self.get())
    }

    fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

/// Gauge that can go up and down
#[derive(Debug)]
pub struct Gauge {
    name: String,
    help: String,
    value: RwLock<f64>,
    labels: HashMap<String, String>,
}

impl Gauge {
    pub fn new(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            help: help.to_string(),
            value: RwLock::new(0.0),
            labels: HashMap::new(),
        }
    }

    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    pub fn set(&self, value: f64) {
        let mut v = self.value.write().unwrap();
        *v = value;
    }

    pub fn inc(&self) {
        let mut v = self.value.write().unwrap();
        *v += 1.0;
    }

    pub fn dec(&self) {
        let mut v = self.value.write().unwrap();
        *v -= 1.0;
    }

    pub fn get(&self) -> f64 {
        *self.value.read().unwrap()
    }

    fn format(&self) -> String {
        let labels = self.format_labels();
        format!("{}{} {}", self.name, labels, self.get())
    }

    fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

/// Histogram for distributions
#[derive(Debug)]
pub struct Histogram {
    name: String,
    help: String,
    buckets: Vec<f64>,
    counts: Vec<AtomicU64>,
    sum: RwLock<f64>,
    count: AtomicU64,
    labels: HashMap<String, String>,
}

impl Histogram {
    pub fn new(name: &str, help: &str, buckets: Vec<f64>) -> Self {
        let counts = buckets.iter().map(|_| AtomicU64::new(0)).collect();
        Self {
            name: name.to_string(),
            help: help.to_string(),
            buckets,
            counts,
            sum: RwLock::new(0.0),
            count: AtomicU64::new(0),
            labels: HashMap::new(),
        }
    }

    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    pub fn observe(&self, value: f64) {
        // Update sum and count
        {
            let mut sum = self.sum.write().unwrap();
            *sum += value;
        }
        self.count.fetch_add(1, Ordering::Relaxed);

        // Update bucket counts
        for (i, bucket) in self.buckets.iter().enumerate() {
            if value <= *bucket {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn format(&self) -> String {
        let labels_base = self.format_labels();
        let mut lines = Vec::new();

        // Bucket lines
        let mut cumulative = 0u64;
        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            let bucket_labels = if labels_base.is_empty() {
                format!("{{le=\"{}\"}}", bucket)
            } else {
                format!(
                    "{{{},le=\"{}\"}}",
                    &labels_base[1..labels_base.len() - 1],
                    bucket
                )
            };
            lines.push(format!(
                "{}_bucket{} {}",
                self.name, bucket_labels, cumulative
            ));
        }

        // +Inf bucket
        let inf_labels = if labels_base.is_empty() {
            "{le=\"+Inf\"}".to_string()
        } else {
            format!("{{{},le=\"+Inf\"}}", &labels_base[1..labels_base.len() - 1])
        };
        lines.push(format!(
            "{}_bucket{} {}",
            self.name,
            inf_labels,
            self.count.load(Ordering::Relaxed)
        ));

        // Sum and count
        lines.push(format!(
            "{}_sum{} {}",
            self.name,
            labels_base,
            *self.sum.read().unwrap()
        ));
        lines.push(format!(
            "{}_count{} {}",
            self.name,
            labels_base,
            self.count.load(Ordering::Relaxed)
        ));

        lines.join("\n")
    }

    fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRICS REGISTRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Fractal metrics registry
pub struct MetricsRegistry {
    // Nociception metrics
    pub pain_intensity: Gauge,
    pub damage_total: Gauge,
    pub pain_events_total: Counter,
    pub pain_response_histogram: Histogram,

    // Thermoception metrics
    pub thermal_utilization: Gauge,
    pub thermal_zones: HashMap<String, Gauge>,
    pub thermal_warnings_total: Counter,
    pub thermal_critical_total: Counter,

    // Containment metrics
    pub containment_evaluations_total: Counter,
    pub containment_blocked_total: Counter,
    pub containment_allowed_total: Counter,
    pub threat_level: Gauge,
    pub manipulation_attempts_total: Counter,

    // Orchestration metrics
    pub consensus_agreement: Gauge,
    pub consensus_failures_total: Counter,
    pub agent_responses_total: Counter,
    pub safety_veto_total: Counter,

    // Authentication metrics
    pub auth_success_total: Counter,
    pub auth_failure_total: Counter,
    pub active_sessions: Gauge,
    pub auth_latency_histogram: Histogram,

    // System metrics
    pub uptime_seconds: Gauge,
    pub cycles_total: Counter,
    pub health_score: Gauge,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            // Nociception
            pain_intensity: Gauge::new(
                "fractal_pain_intensity",
                "Current pain signal intensity (0-1)",
            ),
            damage_total: Gauge::new("fractal_damage_total", "Accumulated damage state (0-1)"),
            pain_events_total: Counter::new(
                "fractal_pain_events_total",
                "Total pain events detected",
            ),
            pain_response_histogram: Histogram::new(
                "fractal_pain_response_seconds",
                "Pain response latency distribution",
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),

            // Thermoception
            thermal_utilization: Gauge::new(
                "fractal_thermal_utilization",
                "Global thermal utilization (0-1)",
            ),
            thermal_zones: HashMap::new(),
            thermal_warnings_total: Counter::new(
                "fractal_thermal_warnings_total",
                "Total thermal warning events",
            ),
            thermal_critical_total: Counter::new(
                "fractal_thermal_critical_total",
                "Total thermal critical events",
            ),

            // Containment
            containment_evaluations_total: Counter::new(
                "fractal_containment_evaluations_total",
                "Total containment evaluations",
            ),
            containment_blocked_total: Counter::new(
                "fractal_containment_blocked_total",
                "Total requests blocked by containment",
            ),
            containment_allowed_total: Counter::new(
                "fractal_containment_allowed_total",
                "Total requests allowed by containment",
            ),
            threat_level: Gauge::new(
                "fractal_threat_level",
                "Current threat level (0=none, 4=critical)",
            ),
            manipulation_attempts_total: Counter::new(
                "fractal_manipulation_attempts_total",
                "Total detected manipulation attempts",
            ),

            // Orchestration
            consensus_agreement: Gauge::new(
                "fractal_consensus_agreement",
                "Current consensus agreement level (0-1)",
            ),
            consensus_failures_total: Counter::new(
                "fractal_consensus_failures_total",
                "Total consensus failures",
            ),
            agent_responses_total: Counter::new(
                "fractal_agent_responses_total",
                "Total agent responses processed",
            ),
            safety_veto_total: Counter::new(
                "fractal_safety_veto_total",
                "Total safety vetoes by beta agent",
            ),

            // Authentication
            auth_success_total: Counter::new(
                "fractal_auth_success_total",
                "Total successful authentications",
            ),
            auth_failure_total: Counter::new(
                "fractal_auth_failure_total",
                "Total failed authentications",
            ),
            active_sessions: Gauge::new(
                "fractal_active_sessions",
                "Current number of active sessions",
            ),
            auth_latency_histogram: Histogram::new(
                "fractal_auth_latency_seconds",
                "Authentication latency distribution",
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),

            // System
            uptime_seconds: Gauge::new("fractal_uptime_seconds", "Seconds since system start"),
            cycles_total: Counter::new("fractal_cycles_total", "Total processing cycles"),
            health_score: Gauge::new("fractal_health_score", "Overall system health score (0-1)"),
        }
    }

    /// Register a thermal zone gauge
    pub fn register_thermal_zone(&mut self, zone: &str) {
        let gauge = Gauge::new(
            "fractal_thermal_zone_utilization",
            "Thermal zone utilization",
        )
        .with_label("zone", zone);
        self.thermal_zones.insert(zone.to_string(), gauge);
    }

    /// Update thermal zone metric
    pub fn set_thermal_zone(&self, zone: &str, value: f64) {
        if let Some(gauge) = self.thermal_zones.get(zone) {
            gauge.set(value);
        }
    }

    /// Export all metrics in Prometheus format
    pub fn export(&self) -> String {
        let mut output = Vec::new();

        // Nociception
        Self::push_gauge(&mut output, &self.pain_intensity);
        Self::push_gauge(&mut output, &self.damage_total);
        Self::push_counter(&mut output, &self.pain_events_total);
        Self::push_histogram(&mut output, &self.pain_response_histogram);

        // Thermoception
        Self::push_gauge(&mut output, &self.thermal_utilization);
        for gauge in self.thermal_zones.values() {
            output.push(gauge.format());
        }
        Self::push_counter(&mut output, &self.thermal_warnings_total);
        Self::push_counter(&mut output, &self.thermal_critical_total);

        // Containment
        Self::push_counter(&mut output, &self.containment_evaluations_total);
        Self::push_counter(&mut output, &self.containment_blocked_total);
        Self::push_counter(&mut output, &self.containment_allowed_total);
        Self::push_gauge(&mut output, &self.threat_level);
        Self::push_counter(&mut output, &self.manipulation_attempts_total);

        // Orchestration
        Self::push_gauge(&mut output, &self.consensus_agreement);
        Self::push_counter(&mut output, &self.consensus_failures_total);
        Self::push_counter(&mut output, &self.agent_responses_total);
        Self::push_counter(&mut output, &self.safety_veto_total);

        // Authentication
        Self::push_counter(&mut output, &self.auth_success_total);
        Self::push_counter(&mut output, &self.auth_failure_total);
        Self::push_gauge(&mut output, &self.active_sessions);
        Self::push_histogram(&mut output, &self.auth_latency_histogram);

        // System
        Self::push_gauge(&mut output, &self.uptime_seconds);
        Self::push_counter(&mut output, &self.cycles_total);
        Self::push_gauge(&mut output, &self.health_score);

        output.join("\n")
    }

    fn push_gauge(output: &mut Vec<String>, g: &Gauge) {
        output.push(format!("# HELP {} {}", g.name, g.help));
        output.push(format!("# TYPE {} gauge", g.name));
        output.push(g.format());
    }

    fn push_counter(output: &mut Vec<String>, c: &Counter) {
        output.push(format!("# HELP {} {}", c.name, c.help));
        output.push(format!("# TYPE {} counter", c.name));
        output.push(c.format());
    }

    fn push_histogram(output: &mut Vec<String>, h: &Histogram) {
        output.push(format!("# HELP {} {}", h.name, h.help));
        output.push(format!("# TYPE {} histogram", h.name));
        output.push(h.format());
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL METRICS INSTANCE
// ═══════════════════════════════════════════════════════════════════════════════

use std::sync::OnceLock;

static METRICS: OnceLock<MetricsRegistry> = OnceLock::new();

/// Get global metrics registry
pub fn metrics() -> &'static MetricsRegistry {
    METRICS.get_or_init(MetricsRegistry::new)
}

/// Initialize metrics with custom registry
#[allow(clippy::result_large_err)]
pub fn init_metrics(registry: MetricsRegistry) -> Result<(), MetricsRegistry> {
    METRICS.set(registry)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new("test_counter", "Test counter");
        assert_eq!(counter.get(), 0);
        counter.inc();
        assert_eq!(counter.get(), 1);
        counter.inc_by(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new("test_gauge", "Test gauge");
        assert_eq!(gauge.get(), 0.0);
        gauge.set(42.5);
        assert_eq!(gauge.get(), 42.5);
        gauge.inc();
        assert_eq!(gauge.get(), 43.5);
    }

    #[test]
    fn test_histogram() {
        let hist = Histogram::new("test_hist", "Test histogram", vec![0.1, 0.5, 1.0]);
        hist.observe(0.05);
        hist.observe(0.3);
        hist.observe(0.8);
        hist.observe(1.5);

        let output = hist.format();
        assert!(output.contains("test_hist_count 4"));
    }

    #[test]
    fn test_metrics_export() {
        let registry = MetricsRegistry::new();
        registry.pain_intensity.set(0.5);
        registry.containment_blocked_total.inc();

        let output = registry.export();
        assert!(output.contains("fractal_pain_intensity"));
        assert!(output.contains("fractal_containment_blocked_total"));
    }
}
