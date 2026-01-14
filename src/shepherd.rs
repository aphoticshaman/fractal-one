//! ═══════════════════════════════════════════════════════════════════════════════
//! SHEPHERD — Conflict Early Warning via Nucleation Detection
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Fetches GDELT data and runs Shepherd Dynamics to detect pre-conflict signatures
//! using phase transition theory from statistical physics.
//!
//! ## Epistemic Limitations — READ BEFORE USE
//!
//! **This is experimental research software, not a validated forecasting system.**
//!
//! - **No proven predictive validity**: The nucleation model has not been validated
//!   against historical conflict onset with proper out-of-sample testing.
//! - **GDELT data quality**: GDELT is machine-coded from news, not ground truth.
//!   Media coverage ≠ actual events. Biased toward English sources.
//! - **Base rate problem**: Major conflicts are rare events. Any detector will
//!   have either high false positive rates or miss most true positives.
//! - **Feedback loops**: If predictions influence actions, they may self-fulfill
//!   or self-negate, invalidating the model.
//! - **Goldstein scale limitations**: The conflict/cooperation scale is from 1992
//!   and may not capture modern conflict dynamics.
//!
//! ## Appropriate Use
//!
//! - Research into conflict dynamics and early warning methodology
//! - Generating hypotheses for human analysts to investigate
//! - Educational demonstrations of phase transition concepts
//!
//! ## Inappropriate Use
//!
//! - Automated decision-making about interventions
//! - Investment decisions based on conflict predictions
//! - Claims of reliable conflict forecasting
//! ═══════════════════════════════════════════════════════════════════════════════

use nucleation::{AlertLevel, ShepherdDynamics, VarianceConfig};
use std::collections::HashMap;

use crate::stats::float_cmp;

/// GDELT event (minimal fields)
#[derive(Debug, Clone)]
pub struct GdeltEvent {
    pub date: String,
    pub actor1_country: Option<String>,
    pub actor2_country: Option<String>,
    pub goldstein: f64,
    pub num_mentions: u32,
}

/// Fetch GDELT data for a single day
pub fn fetch_gdelt_day(date: &str) -> Result<Vec<GdeltEvent>, String> {
    let url = format!(
        "http://data.gdeltproject.org/events/{}.export.CSV.zip",
        date
    );

    let response = reqwest::blocking::get(&url).map_err(|e| format!("HTTP error: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }

    let bytes = response.bytes().map_err(|e| format!("Read error: {}", e))?;

    // Unzip
    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| format!("Zip error: {}", e))?;

    let mut csv_file = archive
        .by_index(0)
        .map_err(|e| format!("Zip entry error: {}", e))?;

    let mut csv_data = String::new();
    std::io::Read::read_to_string(&mut csv_file, &mut csv_data)
        .map_err(|e| format!("Read error: {}", e))?;

    // Parse
    let mut events = Vec::new();
    for line in csv_data.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 35 {
            continue;
        }

        let goldstein: f64 = match fields.get(30).and_then(|s| s.parse().ok()) {
            Some(g) => g,
            None => continue,
        };

        events.push(GdeltEvent {
            date: fields.get(1).unwrap_or(&"").to_string(),
            actor1_country: fields.get(7).map(|s| s.chars().take(3).collect()),
            actor2_country: fields.get(17).map(|s| s.chars().take(3).collect()),
            goldstein,
            num_mentions: fields.get(31).and_then(|s| s.parse().ok()).unwrap_or(1),
        });
    }

    Ok(events)
}

/// Aggregate events by country pair into observation vectors
pub fn aggregate_to_observations(
    events: &[GdeltEvent],
    n_bins: usize,
) -> HashMap<String, Vec<f64>> {
    // Goldstein scale is -10 to +10, bin into n_bins
    let mut country_events: HashMap<String, Vec<f64>> = HashMap::new();

    for event in events {
        for c in [&event.actor1_country, &event.actor2_country].into_iter().flatten() {
            if c.len() >= 2 {
                country_events
                    .entry(c.clone())
                    .or_default()
                    .push(event.goldstein);
            }
        }
    }

    // Convert to histogram distributions
    let mut observations = HashMap::new();
    for (country, goldsteins) in country_events {
        if goldsteins.len() < 10 {
            continue; // Skip countries with too few events
        }

        let mut bins = vec![0.0; n_bins];
        for g in &goldsteins {
            // Map -10..+10 to 0..n_bins
            let normalized = (g + 10.0) / 20.0; // 0..1
            let bin_idx = ((normalized * n_bins as f64) as usize).min(n_bins - 1);
            bins[bin_idx] += 1.0;
        }

        // Normalize to probability distribution
        let sum: f64 = bins.iter().sum();
        if sum > 0.0 {
            for b in &mut bins {
                *b /= sum;
            }
        }

        observations.insert(country, bins);
    }

    observations
}

/// Run Shepherd Dynamics on GDELT data
pub async fn run_monitor(days: usize, focus_country: Option<&str>) -> anyhow::Result<()> {
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m SHEPHERD DYNAMICS — Conflict Early Warning Monitor\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();

    let n_bins = 20; // Goldstein histogram bins
    let mut shepherd =
        ShepherdDynamics::new(n_bins).with_variance_config(VarianceConfig::sensitive());

    // Generate date range
    let today = chrono::Utc::now().date_naive();
    let mut dates: Vec<String> = Vec::new();
    for i in (1..=days).rev() {
        let date = today - chrono::Duration::days(i as i64);
        dates.push(date.format("%Y%m%d").to_string());
    }

    println!("Fetching {} days of GDELT data...", days);

    let mut all_events = Vec::new();
    for (i, date) in dates.iter().enumerate() {
        print!("\r  [{}/{}] {}...", i + 1, days, date);
        std::io::Write::flush(&mut std::io::stdout())?;

        match fetch_gdelt_day(date) {
            Ok(events) => {
                all_events.extend(events);
            }
            Err(e) => {
                eprintln!(" failed: {}", e);
            }
        }
    }
    println!(
        "\r  Fetched {} events from {} days          ",
        all_events.len(),
        days
    );

    // Filter by country if specified
    if let Some(country) = focus_country {
        all_events.retain(|e| {
            e.actor1_country.as_deref() == Some(country)
                || e.actor2_country.as_deref() == Some(country)
        });
        println!(
            "  Filtered to {} events involving {}",
            all_events.len(),
            country
        );
    }

    // Group events by date
    let mut events_by_date: HashMap<String, Vec<GdeltEvent>> = HashMap::new();
    for event in all_events {
        events_by_date
            .entry(event.date.clone())
            .or_default()
            .push(event);
    }

    println!("\nRunning Shepherd Dynamics analysis...");

    // Process each day chronologically
    let mut all_alerts = Vec::new();
    for (i, date) in dates.iter().enumerate() {
        let timestamp = i as f64;

        if let Some(day_events) = events_by_date.get(date) {
            let observations = aggregate_to_observations(day_events, n_bins);

            // Register/update actors
            for (country, obs) in &observations {
                if shepherd.actors().iter().any(|&a| a == country) {
                    let alerts = shepherd.update_actor(country, obs, timestamp);
                    all_alerts.extend(alerts);
                } else {
                    shepherd.register_actor(country.clone(), Some(obs.clone()));
                }
            }
        }
    }

    // Get current state
    let potentials = shepherd.all_potentials();

    // Print results
    println!();
    println!("\x1b[33m═══ TOP CONFLICT POTENTIALS ═══\x1b[0m");
    println!();

    let mut sorted_potentials = potentials.clone();
    sorted_potentials.sort_by(|a, b| float_cmp(&b.phi, &a.phi));

    println!(
        "{:<12} {:<12} {:>10} {:>12}",
        "Actor A", "Actor B", "Phi", "Divergence"
    );
    println!("{}", "-".repeat(50));

    for p in sorted_potentials.iter().take(15) {
        let color = if p.phi > 1.5 {
            "\x1b[31m"
        }
        // Red
        else if p.phi > 1.0 {
            "\x1b[33m"
        }
        // Yellow
        else {
            "\x1b[0m"
        };

        println!(
            "{}{:<12} {:<12} {:>10.3} {:>12.3}\x1b[0m",
            color,
            p.actor_a,
            p.actor_b,
            p.phi,
            p.kl_a_b + p.kl_b_a
        );
    }

    // Print alerts
    if !all_alerts.is_empty() {
        println!();
        println!("\x1b[31m═══ NUCLEATION ALERTS ═══\x1b[0m");
        println!();

        for alert in all_alerts
            .iter()
            .filter(|a| a.alert_level >= AlertLevel::Yellow)
        {
            let color = match alert.alert_level {
                AlertLevel::Red => "\x1b[31m",
                AlertLevel::Orange => "\x1b[33m",
                AlertLevel::Yellow => "\x1b[93m",
                _ => "\x1b[0m",
            };
            println!("{}{}\x1b[0m", color, alert.message);
        }
    } else {
        println!();
        println!("\x1b[32mNo nucleation alerts in analysis period.\x1b[0m");
    }

    println!();
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");

    Ok(())
}

/// Country code to name
pub fn country_name(code: &str) -> &str {
    match code {
        "USA" => "United States",
        "RUS" => "Russia",
        "CHN" => "China",
        "UKR" => "Ukraine",
        "ISR" => "Israel",
        "IRN" => "Iran",
        "SYR" => "Syria",
        "PSE" => "Palestine",
        "LBN" => "Lebanon",
        "YEM" => "Yemen",
        "SAU" => "Saudi Arabia",
        "TUR" => "Turkey",
        "PRK" => "North Korea",
        "KOR" => "South Korea",
        "TWN" => "Taiwan",
        "IND" => "India",
        "PAK" => "Pakistan",
        _ => code,
    }
}
