//! ═══════════════════════════════════════════════════════════════════════════════
//! MCSI — Media Conflict Salience Index
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Formerly "RED/BLUE Oracle". Renamed after null test (v0.4) revealed faction
//! labels add no signal beyond random assignment.
//!
//! WHAT THIS MEASURES:
//! - Total conflict-related media coverage volume
//! - Violence/coercion event rates in GDELT data
//! - Cross-country and temporal comparisons
//!
//! WHAT THIS DOES NOT MEASURE:
//! - Actual partisan/faction behavior (L1 ecological inference confirmed)
//! - RED vs BLUE divergence (null test z = -2.26, no signal)
//!
//! VALIDATED:
//! ✓ US vs Canada (67x volume difference)
//! ✓ Historical pattern detection (BLM, elections)
//! ✓ Total conflict salience as indicator
//!
//! NOT VALIDATED:
//! ✗ Faction decomposition (L6 null test FAILED)
//! ✗ Partisan conflict measurement
//!
//! See ORACLE_METHODOLOGY.md for full documentation including null test results.
//! ═══════════════════════════════════════════════════════════════════════════════

use nucleation::{AlertLevel, ShepherdDynamics, VarianceConfig};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// FACTION BEHAVIORAL VECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Behavioral dimensions for each faction
#[derive(Debug, Clone, Default)]
pub struct FactionBehavior {
    // ─── Rhetoric Intensity (0.0 - 1.0) ───
    /// Hostile language toward outgroup
    pub rhetoric_hostility: f64,
    /// Dehumanizing language (vermin, enemies, etc.)
    pub rhetoric_dehumanization: f64,
    /// Violence-adjacent language (fight, destroy, eliminate)
    pub rhetoric_violence_adjacent: f64,
    /// Delegitimization of institutions/elections
    pub rhetoric_delegitimization: f64,

    // ─── Action Intensity ───
    /// Protest events per period
    pub protest_frequency: f64,
    /// Counter-protest events per period
    pub counter_protest_frequency: f64,
    /// Political violence incidents
    pub violence_incidents: f64,
    /// Armed/militia activity signals
    pub militia_signals: f64,

    // ─── Institutional Behavior ───
    /// Cross-party legislative cooperation (inverse = polarization)
    pub legislative_cooperation: f64,
    /// Legal/court challenges filed
    pub legal_challenges: f64,
    /// Norm violations (unprecedented actions)
    pub norm_violations: f64,
    /// Election certification challenges
    pub election_challenges: f64,

    // ─── Geographic ───
    /// Geographic concentration index (clustering vs dispersed)
    pub geographic_concentration: f64,
    /// State-level non-compliance signals
    pub state_defiance: f64,
}

impl FactionBehavior {
    /// Convert to histogram distribution for Shepherd
    pub fn to_distribution(&self, n_bins: usize) -> Vec<f64> {
        // Flatten all dimensions into a single distribution
        let values = [
            self.rhetoric_hostility,
            self.rhetoric_dehumanization,
            self.rhetoric_violence_adjacent,
            self.rhetoric_delegitimization,
            self.protest_frequency,
            self.counter_protest_frequency,
            self.violence_incidents,
            self.militia_signals,
            self.legislative_cooperation,
            self.legal_challenges,
            self.norm_violations,
            self.election_challenges,
            self.geographic_concentration,
            self.state_defiance,
        ];

        // Bin each dimension
        let mut bins = vec![0.0; n_bins];
        for (_i, &v) in values.iter().enumerate() {
            let bin_idx = ((v * n_bins as f64) as usize).min(n_bins - 1);
            bins[bin_idx] += 1.0;
        }

        // Normalize
        let sum: f64 = bins.iter().sum();
        if sum > 0.0 {
            for b in &mut bins {
                *b /= sum;
            }
        }

        bins
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA SOURCES
// ═══════════════════════════════════════════════════════════════════════════════

/// Data source trait - each source implements this
pub trait DataSource: Send + Sync {
    /// Source name for logging
    fn name(&self) -> &str;

    /// Fetch latest data and update faction behaviors
    fn fetch(&self) -> Result<SourceUpdate, SourceError>;

    /// How stale can this source be before warning (seconds)
    fn max_staleness(&self) -> u64;
}

/// Update from a data source
#[derive(Debug, Clone)]
pub struct SourceUpdate {
    pub timestamp: f64,
    pub red_deltas: FactionBehavior,
    pub blue_deltas: FactionBehavior,
    pub confidence: f64, // 0-1, how reliable is this update
}

/// Data source errors
#[derive(Debug)]
pub enum SourceError {
    NetworkError(String),
    ParseError(String),
    RateLimited,
    AuthenticationFailed,
    NotAvailable(String),
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NetworkError(e) => write!(f, "Network error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
            Self::RateLimited => write!(f, "Rate limited"),
            Self::AuthenticationFailed => write!(f, "Authentication failed"),
            Self::NotAvailable(e) => write!(f, "Not available: {}", e),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOTEVIEW SOURCE — Congressional Voting Polarization
// ═══════════════════════════════════════════════════════════════════════════════

pub struct VoteviewSource {
    /// Base URL for data
    base_url: String,
    /// Historical polarization for trend detection
    #[allow(dead_code)]
    polarization_history: Vec<(u16, f64)>, // (congress, polarization)
}

impl VoteviewSource {
    pub fn new() -> Self {
        Self {
            base_url: "https://voteview.com/static/data/out/members".into(),
            polarization_history: Vec::new(),
        }
    }

    /// Fetch and parse member data for a specific congress/chamber
    fn fetch_members(
        &self,
        congress: u16,
        chamber: &str,
    ) -> Result<Vec<VoteviewMember>, SourceError> {
        let _url = format!(
            "{}/{}{}_{}_members.csv",
            self.base_url,
            chamber.chars().next().unwrap().to_uppercase(),
            congress,
            "members"
        );

        // Actually: URL format is H119_members.csv or S119_members.csv
        let url = format!(
            "{}/{}{}_members.csv",
            self.base_url,
            if chamber == "house" { "H" } else { "S" },
            congress
        );

        let response = reqwest::blocking::get(&url)
            .map_err(|e| SourceError::NetworkError(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            return Err(SourceError::NetworkError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let text = response
            .text()
            .map_err(|e| SourceError::NetworkError(format!("Read error: {}", e)))?;

        self.parse_members_csv(&text)
    }

    /// Parse VOTEVIEW CSV format (handles quoted fields with commas)
    fn parse_members_csv(&self, csv_text: &str) -> Result<Vec<VoteviewMember>, SourceError> {
        let mut members = Vec::new();
        let mut lines = csv_text.lines();

        // Parse header to find column indices
        let header = lines
            .next()
            .ok_or_else(|| SourceError::ParseError("Empty CSV".into()))?;

        let columns: Vec<&str> = header.split(',').collect();

        let find_col = |name: &str| -> Option<usize> { columns.iter().position(|&c| c == name) };

        let col_party = find_col("party_code")
            .ok_or_else(|| SourceError::ParseError("Missing party_code column".into()))?;
        let col_nominate1 = find_col("nominate_dim1")
            .ok_or_else(|| SourceError::ParseError("Missing nominate_dim1 column".into()))?;
        let col_nominate2 = find_col("nominate_dim2");
        let col_bioname = find_col("bioname");
        let col_state = find_col("state_abbrev");

        // Parse data rows
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }

            // Parse CSV with quote handling
            let fields = parse_csv_line(line);

            let party_code: u16 = fields
                .get(col_party)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let nominate_dim1: f64 = fields
                .get(col_nominate1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);

            let nominate_dim2: f64 = col_nominate2
                .and_then(|i| fields.get(i))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);

            // Skip members without valid NOMINATE scores (empty = not yet computed)
            // Only skip if the field is empty, not if it's 0.0
            let nom1_str = fields.get(col_nominate1).map(|s| s.as_str()).unwrap_or("");
            if nom1_str.is_empty() {
                continue;
            }

            members.push(VoteviewMember {
                party_code,
                nominate_dim1,
                nominate_dim2,
                bioname: col_bioname
                    .and_then(|i| fields.get(i))
                    .cloned()
                    .unwrap_or_default(),
                state: col_state
                    .and_then(|i| fields.get(i))
                    .cloned()
                    .unwrap_or_default(),
            });
        }

        Ok(members)
    }

    /// Compute polarization metrics from member data
    fn compute_polarization(&self, members: &[VoteviewMember]) -> PolarizationMetrics {
        // Separate by party
        // 100 = Democrat, 200 = Republican
        let dems: Vec<f64> = members
            .iter()
            .filter(|m| m.party_code == 100)
            .map(|m| m.nominate_dim1)
            .collect();

        let reps: Vec<f64> = members
            .iter()
            .filter(|m| m.party_code == 200)
            .map(|m| m.nominate_dim1)
            .collect();

        // Compute medians
        let dem_median = median(&dems);
        let rep_median = median(&reps);

        // Polarization = distance between party medians
        // Historically ranged from ~0.5 (1970s) to ~1.0+ (2020s)
        let polarization = (rep_median - dem_median).abs();

        // Compute overlap (how many members cross the midpoint)
        let midpoint = (dem_median + rep_median) / 2.0;
        let dems_right_of_mid = dems.iter().filter(|&&x| x > midpoint).count();
        let reps_left_of_mid = reps.iter().filter(|&&x| x < midpoint).count();
        let overlap_count = dems_right_of_mid + reps_left_of_mid;
        let overlap_ratio = overlap_count as f64 / members.len() as f64;

        // Extremism: standard deviation within each party (higher = more extreme wings)
        let dem_std = std_dev(&dems);
        let rep_std = std_dev(&reps);

        PolarizationMetrics {
            polarization,
            dem_median,
            rep_median,
            overlap_ratio,
            dem_extremism: dem_std,
            rep_extremism: rep_std,
            dem_count: dems.len(),
            rep_count: reps.len(),
        }
    }
}

/// VOTEVIEW member data
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct VoteviewMember {
    party_code: u16,    // 100 = Dem, 200 = Rep
    nominate_dim1: f64, // -1 (liberal) to +1 (conservative)
    nominate_dim2: f64, // Second dimension (historically race/civil rights)
    bioname: String,
    state: String,
}

/// Polarization metrics computed from VOTEVIEW
#[derive(Debug, Clone)]
pub struct PolarizationMetrics {
    /// Distance between party medians (0.5 = low, 1.0+ = high)
    pub polarization: f64,
    /// Democrat median on dim1 (negative = liberal)
    pub dem_median: f64,
    /// Republican median on dim1 (positive = conservative)
    pub rep_median: f64,
    /// Fraction of members crossing party midpoint (0 = sorted, >0 = overlap)
    pub overlap_ratio: f64,
    /// Std dev of Democrat positions (higher = extreme wing)
    pub dem_extremism: f64,
    /// Std dev of Republican positions
    pub rep_extremism: f64,
    /// Number of Democrats
    pub dem_count: usize,
    /// Number of Republicans
    pub rep_count: usize,
}

/// Parse a CSV line handling quoted fields with commas
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for c in line.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    fields.push(current);

    fields
}

impl DataSource for VoteviewSource {
    fn name(&self) -> &str {
        "VOTEVIEW"
    }

    fn fetch(&self) -> Result<SourceUpdate, SourceError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Fetch current congress (119th as of 2025-2027)
        let congress = 119_u16;

        // Fetch both chambers
        let house_members = self.fetch_members(congress, "house")?;
        let senate_members = self.fetch_members(congress, "senate")?;

        let mut all_members = house_members;
        all_members.extend(senate_members);

        if all_members.is_empty() {
            return Err(SourceError::ParseError("No members parsed".into()));
        }

        let metrics = self.compute_polarization(&all_members);

        eprintln!(
            "[VOTEVIEW] Congress {}: polarization={:.3}, overlap={:.1}%, D={} R={}",
            congress,
            metrics.polarization,
            metrics.overlap_ratio * 100.0,
            metrics.dem_count,
            metrics.rep_count
        );

        // Map to FactionBehavior
        // legislative_cooperation is INVERSE of polarization
        // Normalize: polarization of 0.5 = cooperation 0.5, polarization of 1.0 = cooperation 0.0
        let cooperation = (1.0 - metrics.polarization).max(0.0).min(1.0);

        // Higher extremism = more norm violations
        let dem_norm_violations = (metrics.dem_extremism * 2.0).min(1.0);
        let rep_norm_violations = (metrics.rep_extremism * 2.0).min(1.0);

        let mut red_deltas = FactionBehavior::default();
        red_deltas.legislative_cooperation = cooperation;
        red_deltas.norm_violations = rep_norm_violations;
        // Extremism can indicate delegitimization tendency
        red_deltas.rhetoric_delegitimization = (metrics.rep_extremism * 1.5).min(1.0);

        let mut blue_deltas = FactionBehavior::default();
        blue_deltas.legislative_cooperation = cooperation;
        blue_deltas.norm_violations = dem_norm_violations;
        blue_deltas.rhetoric_delegitimization = (metrics.dem_extremism * 1.5).min(1.0);

        Ok(SourceUpdate {
            timestamp,
            red_deltas,
            blue_deltas,
            confidence: 0.95, // Congressional voting is high-quality ground truth
        })
    }

    fn max_staleness(&self) -> u64 {
        86400 * 7
    } // Weekly updates OK
}

/// Compute median of a slice
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Compute standard deviation of a slice
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════
// GDELT SOURCE — News Sentiment & Domestic Events
// ═══════════════════════════════════════════════════════════════════════════════

pub struct GdeltSource {
    /// Days of data to fetch
    lookback_days: usize,
}

/// GDELT event for domestic analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GdeltDomesticEvent {
    date: String,
    actor1_name: String,
    actor1_country: String,
    actor1_type: String,
    actor2_name: String,
    actor2_country: String,
    actor2_type: String,
    event_code: String, // CAMEO event code
    goldstein: f64,     // -10 (conflict) to +10 (cooperation)
    num_mentions: u32,
    num_sources: u32,
    avg_tone: f64, // Negative = hostile, positive = friendly
    action_geo_country: String,
    source_url: String, // Column 58 (0-indexed 57) - news source URL
}

impl GdeltSource {
    pub fn new(lookback_days: usize) -> Self {
        Self { lookback_days }
    }

    /// Fetch GDELT data for a single day (reuses shepherd logic)
    fn fetch_day(&self, date: &str) -> Result<Vec<GdeltDomesticEvent>, SourceError> {
        let url = format!(
            "http://data.gdeltproject.org/events/{}.export.CSV.zip",
            date
        );

        let response = reqwest::blocking::get(&url)
            .map_err(|e| SourceError::NetworkError(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            return Err(SourceError::NetworkError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|e| SourceError::NetworkError(format!("Read error: {}", e)))?;

        // Unzip
        let cursor = std::io::Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| SourceError::ParseError(format!("Zip error: {}", e)))?;

        let mut csv_file = archive
            .by_index(0)
            .map_err(|e| SourceError::ParseError(format!("Zip entry error: {}", e)))?;

        let mut csv_data = String::new();
        std::io::Read::read_to_string(&mut csv_file, &mut csv_data)
            .map_err(|e| SourceError::ParseError(format!("Read error: {}", e)))?;

        self.parse_gdelt_csv(&csv_data)
    }

    /// Parse GDELT CSV and filter for US domestic events
    fn parse_gdelt_csv(&self, csv_data: &str) -> Result<Vec<GdeltDomesticEvent>, SourceError> {
        let mut events = Vec::new();

        for line in csv_data.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 58 {
                continue;
            }

            // GDELT column indices (0-indexed):
            // 5: Actor1Name, 7: Actor1CountryCode, 12: Actor1Type1Code
            // 15: Actor2Name, 17: Actor2CountryCode, 22: Actor2Type1Code
            // 26: EventCode, 30: GoldsteinScale, 31: NumMentions
            // 32: NumSources, 34: AvgTone
            // 51: ActionGeo_CountryCode (2-letter: US)

            let actor1_country = fields.get(7).unwrap_or(&"").to_string();
            let actor2_country = fields.get(17).unwrap_or(&"").to_string();
            let action_geo = fields.get(51).unwrap_or(&"").to_string();

            // Filter for US domestic: both actors US or empty, action in US
            if !Self::is_us_domestic(&actor1_country, &actor2_country, &action_geo) {
                continue;
            }

            let goldstein: f64 = fields.get(30).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let avg_tone: f64 = fields.get(34).and_then(|s| s.parse().ok()).unwrap_or(0.0);

            events.push(GdeltDomesticEvent {
                date: fields.get(1).unwrap_or(&"").to_string(),
                actor1_name: fields.get(5).unwrap_or(&"").to_string(),
                actor1_country,
                actor1_type: fields.get(12).unwrap_or(&"").to_string(),
                actor2_name: fields.get(15).unwrap_or(&"").to_string(),
                actor2_country,
                actor2_type: fields.get(22).unwrap_or(&"").to_string(),
                event_code: fields.get(26).unwrap_or(&"").to_string(),
                goldstein,
                num_mentions: fields.get(31).and_then(|s| s.parse().ok()).unwrap_or(1),
                num_sources: fields.get(32).and_then(|s| s.parse().ok()).unwrap_or(1),
                avg_tone,
                action_geo_country: fields.get(51).unwrap_or(&"").to_string(),
                source_url: fields.get(57).unwrap_or(&"").to_string(),
            });
        }

        Ok(events)
    }

    /// Filter GDELT events for US domestic only
    fn is_us_domestic(actor1_country: &str, actor2_country: &str, action_geo: &str) -> bool {
        Self::is_country_domestic(actor1_country, actor2_country, action_geo, "US", "USA")
    }

    /// Filter GDELT events for a specific country
    fn is_country_domestic(
        actor1_country: &str,
        actor2_country: &str,
        action_geo: &str,
        geo_code: &str,
        actor_code: &str,
    ) -> bool {
        // Action must be in target country
        if action_geo != geo_code {
            return false;
        }
        // At least one actor should be from target country or unspecified (domestic actor)
        actor1_country == actor_code
            || actor1_country.is_empty()
            || actor2_country == actor_code
            || actor2_country.is_empty()
    }

    /// Parse GDELT CSV and filter for specific country's domestic events
    pub fn parse_gdelt_csv_for_country(
        &self,
        csv_data: &str,
        geo_code: &str,
        actor_code: &str,
    ) -> Result<Vec<GdeltDomesticEvent>, SourceError> {
        let mut events = Vec::new();

        for line in csv_data.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 58 {
                continue;
            }

            let actor1_country = fields.get(7).unwrap_or(&"").to_string();
            let actor2_country = fields.get(17).unwrap_or(&"").to_string();
            let action_geo = fields.get(51).unwrap_or(&"").to_string();

            if !Self::is_country_domestic(
                &actor1_country,
                &actor2_country,
                &action_geo,
                geo_code,
                actor_code,
            ) {
                continue;
            }

            let goldstein: f64 = fields.get(30).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let avg_tone: f64 = fields.get(34).and_then(|s| s.parse().ok()).unwrap_or(0.0);

            events.push(GdeltDomesticEvent {
                date: fields.get(1).unwrap_or(&"").to_string(),
                actor1_name: fields.get(5).unwrap_or(&"").to_string(),
                actor1_country,
                actor1_type: fields.get(12).unwrap_or(&"").to_string(),
                actor2_name: fields.get(15).unwrap_or(&"").to_string(),
                actor2_country,
                actor2_type: fields.get(22).unwrap_or(&"").to_string(),
                event_code: fields.get(26).unwrap_or(&"").to_string(),
                goldstein,
                num_mentions: fields.get(31).and_then(|s| s.parse().ok()).unwrap_or(1),
                num_sources: fields.get(32).and_then(|s| s.parse().ok()).unwrap_or(1),
                avg_tone,
                action_geo_country: fields.get(51).unwrap_or(&"").to_string(),
                source_url: fields.get(57).unwrap_or(&"").to_string(),
            });
        }

        Ok(events)
    }

    /// Fetch and parse for a specific country
    pub fn fetch_day_for_country(
        &self,
        date: &str,
        geo_code: &str,
        actor_code: &str,
    ) -> Result<Vec<GdeltDomesticEvent>, SourceError> {
        let url = format!(
            "http://data.gdeltproject.org/events/{}.export.CSV.zip",
            date
        );

        let response = reqwest::blocking::get(&url)
            .map_err(|e| SourceError::NetworkError(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            return Err(SourceError::NetworkError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|e| SourceError::NetworkError(format!("Read error: {}", e)))?;

        let cursor = std::io::Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| SourceError::ParseError(format!("Zip error: {}", e)))?;

        let mut csv_file = archive
            .by_index(0)
            .map_err(|e| SourceError::ParseError(format!("Zip entry error: {}", e)))?;

        let mut csv_data = String::new();
        std::io::Read::read_to_string(&mut csv_file, &mut csv_data)
            .map_err(|e| SourceError::ParseError(format!("Read error: {}", e)))?;

        self.parse_gdelt_csv_for_country(&csv_data, geo_code, actor_code)
    }

    /// Classify by news source domain (more reliable than actor names for GDELT)
    fn classify_source(url: &str) -> Option<Faction> {
        let url_lower = url.to_lowercase();

        // Conservative/Right-leaning sources
        let red_sources = [
            "foxnews.com",
            "dailycaller.com",
            "breitbart.com",
            "townhall.com",
            "washingtonexaminer.com",
            "nypost.com",
            "newsmax.com",
            "oann.com",
            "thefederalist.com",
            "dailywire.com",
            "nationalreview.com",
            "freebeacon.com",
            "pjmedia.com",
            "hotair.com",
            "redstate.com",
            "westernjournal.com",
            "blazemedia.com",
            "twitchy.com",
        ];

        // Liberal/Left-leaning sources
        let blue_sources = [
            "msnbc.com",
            "huffpost.com",
            "vox.com",
            "slate.com",
            "rawstory.com",
            "thedailybeast.com",
            "salon.com",
            "motherjones.com",
            "theguardian.com",
            "commondreams.org",
            "truthout.org",
            "alternet.org",
            "democracynow.org",
            "jacobin.com",
            "theintercept.com",
            "dailykos.com",
            "talkingpointsmemo.com",
            "crooksandliars.com",
            "mediaite.com",
        ];

        for src in &red_sources {
            if url_lower.contains(src) {
                return Some(Faction::Red);
            }
        }

        for src in &blue_sources {
            if url_lower.contains(src) {
                return Some(Faction::Blue);
            }
        }

        None
    }

    /// Classify actor as RED or BLUE based on name/type (backup for non-source classification)
    fn classify_actor(name: &str, actor_type: &str) -> Option<Faction> {
        let name_lower = name.to_lowercase();
        let type_lower = actor_type.to_lowercase();

        // Republican/Conservative indicators
        let red_keywords = [
            "republican",
            "gop",
            "conservative",
            "trump",
            "maga",
            "desantis",
            "mccarthy",
            "mcconnell",
            "heritage",
            "federalist",
            "tea party",
            "freedom caucus",
            "right-wing",
            "evangelical",
        ];

        // Democrat/Liberal indicators
        let blue_keywords = [
            "democrat",
            "liberal",
            "progressive",
            "biden",
            "harris",
            "pelosi",
            "schumer",
            "aoc",
            "ocasio",
            "sanders",
            "warren",
            "left-wing",
            "planned parenthood",
            "aclu",
            "moveon",
        ];

        for kw in &red_keywords {
            if name_lower.contains(kw) || type_lower.contains(kw) {
                return Some(Faction::Red);
            }
        }

        for kw in &blue_keywords {
            if name_lower.contains(kw) || type_lower.contains(kw) {
                return Some(Faction::Blue);
            }
        }

        None
    }

    /// Compute faction metrics from events
    fn compute_faction_metrics(
        &self,
        events: &[GdeltDomesticEvent],
    ) -> (FactionMetrics, FactionMetrics) {
        let mut red = FactionMetrics::default();
        let mut blue = FactionMetrics::default();

        for event in events {
            // Primary: classify by news source domain
            let source_faction = Self::classify_source(&event.source_url);

            // Fallback: classify by actor names
            let actor_faction = Self::classify_actor(&event.actor1_name, &event.actor1_type)
                .or_else(|| Self::classify_actor(&event.actor2_name, &event.actor2_type));

            // Use source classification if available, otherwise actor
            let faction = source_faction.or(actor_faction);

            // Assign event metrics to detected faction
            if let Some(f) = faction {
                let metrics = match f {
                    Faction::Red => &mut red,
                    Faction::Blue => &mut blue,
                };

                metrics.event_count += 1;
                metrics.total_mentions += event.num_mentions as u64;
                metrics.goldstein_sum += event.goldstein;
                metrics.tone_sum += event.avg_tone;

                // Categorize by CAMEO event code
                // 14x = Protest, 17x = Coerce, 18x = Assault, 19x = Fight, 20x = Unconventional violence
                let code_prefix: u32 = event
                    .event_code
                    .chars()
                    .take(2)
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0);

                match code_prefix {
                    14 => metrics.protest_events += 1,
                    17 | 18 => metrics.coercion_events += 1,
                    19 | 20 => metrics.violence_events += 1,
                    _ => {}
                }
            }
        }

        (red, blue)
    }
}

/// Aggregated metrics for a faction from GDELT
#[derive(Debug, Clone, Default)]
struct FactionMetrics {
    event_count: u64,
    total_mentions: u64,
    goldstein_sum: f64,
    tone_sum: f64,
    protest_events: u64,
    coercion_events: u64,
    violence_events: u64,
}

impl DataSource for GdeltSource {
    fn name(&self) -> &str {
        "GDELT"
    }

    fn fetch(&self) -> Result<SourceUpdate, SourceError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Generate date range
        let today = chrono::Utc::now().date_naive();
        let mut all_events = Vec::new();

        eprintln!(
            "[GDELT] Fetching {} days of US domestic events...",
            self.lookback_days
        );

        for i in (1..=self.lookback_days).rev() {
            let date = today - chrono::Duration::days(i as i64);
            let date_str = date.format("%Y%m%d").to_string();

            eprint!("[GDELT] Fetching {}... ", date_str);
            match self.fetch_day(&date_str) {
                Ok(events) => {
                    eprintln!("{} US domestic events", events.len());
                    all_events.extend(events);
                }
                Err(e) => {
                    eprintln!("FAILED: {}", e);
                }
            }
        }

        if all_events.is_empty() {
            return Err(SourceError::NotAvailable("No GDELT events fetched".into()));
        }

        let (red_metrics, blue_metrics) = self.compute_faction_metrics(&all_events);

        eprintln!(
            "[GDELT] {} US domestic events, RED:{} BLUE:{} tagged",
            all_events.len(),
            red_metrics.event_count,
            blue_metrics.event_count
        );

        // Map to FactionBehavior
        let normalize = |count: u64, total: u64| -> f64 {
            if total == 0 {
                0.0
            } else {
                (count as f64 / total as f64).min(1.0)
            }
        };

        let max_events = red_metrics.event_count.max(blue_metrics.event_count).max(1);

        // Tone: negative = hostile, map to rhetoric_hostility
        let red_avg_tone = if red_metrics.event_count > 0 {
            red_metrics.tone_sum / red_metrics.event_count as f64
        } else {
            0.0
        };
        let blue_avg_tone = if blue_metrics.event_count > 0 {
            blue_metrics.tone_sum / blue_metrics.event_count as f64
        } else {
            0.0
        };

        let mut red_deltas = FactionBehavior::default();
        red_deltas.rhetoric_hostility = (-red_avg_tone / 10.0).max(0.0).min(1.0);
        red_deltas.protest_frequency = normalize(red_metrics.protest_events, max_events);
        red_deltas.violence_incidents = normalize(red_metrics.violence_events, max_events);

        let mut blue_deltas = FactionBehavior::default();
        blue_deltas.rhetoric_hostility = (-blue_avg_tone / 10.0).max(0.0).min(1.0);
        blue_deltas.protest_frequency = normalize(blue_metrics.protest_events, max_events);
        blue_deltas.violence_incidents = normalize(blue_metrics.violence_events, max_events);

        Ok(SourceUpdate {
            timestamp,
            red_deltas,
            blue_deltas,
            confidence: 0.6, // Medium - keyword matching is imprecise
        })
    }

    fn max_staleness(&self) -> u64 {
        86400
    } // Daily updates
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACLED SOURCE — Protests & Political Violence
// ═══════════════════════════════════════════════════════════════════════════════

pub struct AcledSource {
    /// API key (free registration at acleddata.com required)
    api_key: Option<String>,
    /// Email used for ACLED registration
    email: Option<String>,
    /// Days of data to fetch
    lookback_days: usize,
}

/// ACLED event structure
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AcledEvent {
    event_date: String,
    event_type: String,     // Protests, Riots, Violence against civilians, etc.
    sub_event_type: String, // Peaceful protest, Violent demonstration, Attack, etc.
    actor1: String,
    actor2: String,
    fatalities: u32,
    notes: String,
}

impl AcledSource {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.or_else(|| std::env::var("ACLED_API_KEY").ok()),
            email: std::env::var("ACLED_EMAIL").ok(),
            lookback_days: 30,
        }
    }

    /// Fetch ACLED events for US
    fn fetch_events(&self) -> Result<Vec<AcledEvent>, SourceError> {
        let api_key = self.api_key.as_ref().ok_or_else(|| {
            SourceError::NotAvailable(
                "ACLED_API_KEY not set. Register free at acleddata.com".into(),
            )
        })?;
        let email = self
            .email
            .as_ref()
            .ok_or_else(|| SourceError::NotAvailable("ACLED_EMAIL not set".into()))?;

        // Calculate date range (last N days)
        let today = chrono::Utc::now().date_naive();
        let start = today - chrono::Duration::days(self.lookback_days as i64);
        let start_str = start.format("%Y-%m-%d").to_string();
        let end_str = today.format("%Y-%m-%d").to_string();

        // ACLED API URL
        // https://acleddata.com/api/acled/read?_format=json&country=United States&event_date=2026-01-01|2026-01-08&event_date_where=BETWEEN
        let url = format!(
            "https://api.acleddata.com/acled/read?key={}&email={}&country=United%20States&event_date={}|{}&event_date_where=BETWEEN&limit=10000",
            api_key, email, start_str, end_str
        );

        eprintln!("[ACLED] Fetching US events {}..{}", start_str, end_str);

        let response = reqwest::blocking::get(&url)
            .map_err(|e| SourceError::NetworkError(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            return Err(SourceError::NetworkError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let text = response
            .text()
            .map_err(|e| SourceError::NetworkError(format!("Read error: {}", e)))?;

        // Parse JSON response
        self.parse_acled_json(&text)
    }

    /// Parse ACLED JSON response
    fn parse_acled_json(&self, json_text: &str) -> Result<Vec<AcledEvent>, SourceError> {
        // ACLED returns: {"status": 200, "count": N, "data": [...]}
        let json: serde_json::Value = serde_json::from_str(json_text)
            .map_err(|e| SourceError::ParseError(format!("JSON parse error: {}", e)))?;

        let data = json
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| SourceError::ParseError("No 'data' array in response".into()))?;

        let mut events = Vec::new();
        for item in data {
            events.push(AcledEvent {
                event_date: item
                    .get("event_date")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                event_type: item
                    .get("event_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                sub_event_type: item
                    .get("sub_event_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                actor1: item
                    .get("actor1")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                actor2: item
                    .get("actor2")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                fatalities: item.get("fatalities").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                notes: item
                    .get("notes")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            });
        }

        Ok(events)
    }

    /// Classify actor as RED or BLUE based on ACLED actor names
    fn classify_actor(actor: &str, notes: &str) -> Option<Faction> {
        let text = format!("{} {}", actor.to_lowercase(), notes.to_lowercase());

        // Conservative/Right-wing indicators
        let red_keywords = [
            "republican",
            "gop",
            "conservative",
            "trump",
            "maga",
            "proud boys",
            "patriot front",
            "oath keepers",
            "militia",
            "three percenter",
            "boogaloo",
            "groyper",
            "right-wing",
            "pro-life",
            "anti-abortion",
            "pro-gun",
            "second amendment",
        ];

        // Liberal/Left-wing indicators
        let blue_keywords = [
            "democrat",
            "liberal",
            "progressive",
            "antifa",
            "blm",
            "black lives matter",
            "left-wing",
            "socialist",
            "dsa",
            "pro-choice",
            "abortion rights",
            "lgbtq",
            "trans rights",
            "climate activist",
            "environmental",
            "labor union",
        ];

        for kw in &red_keywords {
            if text.contains(kw) {
                return Some(Faction::Red);
            }
        }

        for kw in &blue_keywords {
            if text.contains(kw) {
                return Some(Faction::Blue);
            }
        }

        None
    }

    /// Compute faction metrics from ACLED events
    fn compute_faction_metrics(&self, events: &[AcledEvent]) -> (AcledMetrics, AcledMetrics) {
        let mut red = AcledMetrics::default();
        let mut blue = AcledMetrics::default();

        for event in events {
            let faction = Self::classify_actor(&event.actor1, &event.notes)
                .or_else(|| Self::classify_actor(&event.actor2, &event.notes));

            if let Some(f) = faction {
                let metrics = match f {
                    Faction::Red => &mut red,
                    Faction::Blue => &mut blue,
                };

                metrics.event_count += 1;
                metrics.fatalities += event.fatalities;

                // Classify event type
                match event.event_type.as_str() {
                    "Protests" => {
                        if event.sub_event_type.contains("Violent") {
                            metrics.violent_protests += 1;
                        } else {
                            metrics.peaceful_protests += 1;
                        }
                    }
                    "Riots" => metrics.riots += 1,
                    "Violence against civilians" => metrics.violence_against_civilians += 1,
                    "Battles" => metrics.battles += 1,
                    _ => {}
                }
            }
        }

        (red, blue)
    }
}

/// Aggregated metrics from ACLED
#[derive(Debug, Clone, Default)]
struct AcledMetrics {
    event_count: u32,
    peaceful_protests: u32,
    violent_protests: u32,
    riots: u32,
    violence_against_civilians: u32,
    battles: u32,
    fatalities: u32,
}

impl DataSource for AcledSource {
    fn name(&self) -> &str {
        "ACLED"
    }

    fn fetch(&self) -> Result<SourceUpdate, SourceError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let events = self.fetch_events()?;

        if events.is_empty() {
            return Err(SourceError::NotAvailable("No ACLED events fetched".into()));
        }

        let (red_metrics, blue_metrics) = self.compute_faction_metrics(&events);

        eprintln!(
            "[ACLED] {} events, RED:{} BLUE:{} tagged, {} fatalities",
            events.len(),
            red_metrics.event_count,
            blue_metrics.event_count,
            red_metrics.fatalities + blue_metrics.fatalities
        );

        // Map to FactionBehavior
        let max_events = red_metrics.event_count.max(blue_metrics.event_count).max(1) as f64;

        let mut red_deltas = FactionBehavior::default();
        red_deltas.protest_frequency = red_metrics.peaceful_protests as f64 / max_events;
        red_deltas.violence_incidents =
            (red_metrics.riots + red_metrics.violence_against_civilians) as f64 / max_events;
        red_deltas.counter_protest_frequency = red_metrics.violent_protests as f64 / max_events;

        let mut blue_deltas = FactionBehavior::default();
        blue_deltas.protest_frequency = blue_metrics.peaceful_protests as f64 / max_events;
        blue_deltas.violence_incidents =
            (blue_metrics.riots + blue_metrics.violence_against_civilians) as f64 / max_events;
        blue_deltas.counter_protest_frequency = blue_metrics.violent_protests as f64 / max_events;

        Ok(SourceUpdate {
            timestamp,
            red_deltas,
            blue_deltas,
            confidence: 0.9, // ACLED is expert-curated, high confidence
        })
    }

    fn max_staleness(&self) -> u64 {
        86400 * 7
    } // Weekly updates
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLUESKY SOURCE — Social Media Sentiment
// ═══════════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub struct BlueskySource {
    /// Sample size per faction query
    sample_size: usize,
}

#[allow(dead_code)]
impl BlueskySource {
    pub fn new() -> Self {
        Self { sample_size: 100 }
    }

    /// Search Bluesky for posts matching a query
    fn search_posts(&self, query: &str) -> Result<Vec<BlueskyPost>, SourceError> {
        // Public API endpoint (no auth required for search)
        let url = format!(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts?q={}&limit={}",
            urlencoding::encode(query),
            self.sample_size
        );

        let response = reqwest::blocking::get(&url)
            .map_err(|e| SourceError::NetworkError(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            return Err(SourceError::NetworkError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let text = response
            .text()
            .map_err(|e| SourceError::NetworkError(format!("Read error: {}", e)))?;

        self.parse_search_response(&text)
    }

    /// Parse Bluesky search response
    fn parse_search_response(&self, json_text: &str) -> Result<Vec<BlueskyPost>, SourceError> {
        let json: serde_json::Value = serde_json::from_str(json_text)
            .map_err(|e| SourceError::ParseError(format!("JSON parse error: {}", e)))?;

        let posts = json
            .get("posts")
            .and_then(|p| p.as_array())
            .ok_or_else(|| SourceError::ParseError("No 'posts' array in response".into()))?;

        let mut results = Vec::new();
        for post in posts {
            let record = post.get("record").unwrap_or(post);
            results.push(BlueskyPost {
                text: record
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                created_at: record
                    .get("createdAt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                like_count: post.get("likeCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                reply_count: post.get("replyCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                repost_count: post
                    .get("repostCount")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
            });
        }

        Ok(results)
    }

    /// Compute sentiment metrics from posts
    fn analyze_posts(&self, posts: &[BlueskyPost]) -> SentimentMetrics {
        let mut metrics = SentimentMetrics::default();
        metrics.post_count = posts.len() as u32;

        for post in posts {
            let text_lower = post.text.to_lowercase();

            // Simple sentiment analysis via keyword counting
            let hostile_words = [
                "hate", "evil", "destroy", "enemy", "traitor", "scum", "nazi", "fascist",
            ];
            let dehumanizing = ["vermin", "roach", "animal", "subhuman", "trash", "filth"];
            let violent = ["kill", "shoot", "hang", "execute", "attack", "war"];

            for word in &hostile_words {
                if text_lower.contains(word) {
                    metrics.hostility_signals += 1;
                }
            }

            for word in &dehumanizing {
                if text_lower.contains(word) {
                    metrics.dehumanization_signals += 1;
                }
            }

            for word in &violent {
                if text_lower.contains(word) {
                    metrics.violence_signals += 1;
                }
            }

            // Engagement metrics
            metrics.total_likes += post.like_count;
            metrics.total_reposts += post.repost_count;
        }

        metrics
    }
}

/// Bluesky post data
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BlueskyPost {
    text: String,
    created_at: String,
    like_count: u32,
    reply_count: u32,
    repost_count: u32,
}

/// Sentiment metrics from Bluesky posts
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct SentimentMetrics {
    post_count: u32,
    hostility_signals: u32,
    dehumanization_signals: u32,
    violence_signals: u32,
    total_likes: u32,
    total_reposts: u32,
}

impl DataSource for BlueskySource {
    fn name(&self) -> &str {
        "BLUESKY"
    }

    fn fetch(&self) -> Result<SourceUpdate, SourceError> {
        // NOTE: Bluesky public searchPosts API was disabled in Feb 2025
        // https://github.com/bluesky-social/bsky-docs/issues/332
        // Would need authenticated access via PDS or JetStream WebSocket

        Err(SourceError::NotAvailable(
            "Bluesky public search API disabled (Feb 2025). Requires auth via PDS.".into(),
        ))

        // FUTURE: Implement authenticated access or JetStream WebSocket sampling
        // See: https://docs.bsky.app/docs/advanced-guides/jetstream
    }

    fn max_staleness(&self) -> u64 {
        3600
    } // Hourly updates ideal
}

// ═══════════════════════════════════════════════════════════════════════════════
// KALSHI SOURCE — Prediction Market Implied Probabilities
// ═══════════════════════════════════════════════════════════════════════════════
// Public API: https://api.elections.kalshi.com/trade-api/v2 (no auth required)

#[allow(dead_code)]
pub struct KalshiSource {
    /// Base API URL
    base_url: String,
    /// Series tickers to track for conflict signals
    tracked_series: Vec<String>,
}

/// Kalshi market data
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct KalshiMarket {
    ticker: String,
    title: String,
    yes_bid: f64, // Current YES price (0-100)
    yes_ask: f64,
    volume: u64,
    open_interest: u64,
}

#[allow(dead_code)]
impl KalshiSource {
    pub fn new(_api_key: Option<String>) -> Self {
        Self {
            // Public endpoint (no auth required)
            base_url: "https://api.elections.kalshi.com/trade-api/v2".into(),
            tracked_series: vec![
                // Civil unrest / political violence indicators
                "CIVILUNR".into(), // Civil unrest
                "SHUTDOWN".into(), // Government shutdown
                "MARSHALL".into(), // Martial law
                "INSURREC".into(), // Insurrection
                // Political tension proxies
                "IMPEACH".into(), // Impeachment
                "SCOTUS".into(),  // Supreme Court crises
            ],
        }
    }

    /// Fetch markets for a series
    fn fetch_series(&self, series: &str) -> Result<Vec<KalshiMarket>, SourceError> {
        let url = format!("{}/markets?series_ticker={}", self.base_url, series);

        let response = reqwest::blocking::Client::new()
            .get(&url)
            .header("User-Agent", "Fractal/1.0")
            .send()
            .map_err(|e| SourceError::NetworkError(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            if response.status().as_u16() == 404 {
                return Ok(Vec::new()); // Series not found, not an error
            }
            return Err(SourceError::NetworkError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let text = response
            .text()
            .map_err(|e| SourceError::NetworkError(format!("Read error: {}", e)))?;

        self.parse_markets_response(&text)
    }

    /// Fetch a single market by ticker
    fn fetch_market(&self, ticker: &str) -> Result<Option<KalshiMarket>, SourceError> {
        let url = format!("{}/markets/{}", self.base_url, ticker);

        let response = reqwest::blocking::Client::new()
            .get(&url)
            .header("User-Agent", "Fractal/1.0")
            .send()
            .map_err(|e| SourceError::NetworkError(format!("HTTP error: {}", e)))?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(SourceError::NetworkError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let text = response
            .text()
            .map_err(|e| SourceError::NetworkError(format!("Read error: {}", e)))?;

        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| SourceError::ParseError(format!("JSON parse error: {}", e)))?;

        let market = json.get("market").unwrap_or(&json);

        Ok(Some(KalshiMarket {
            ticker: market
                .get("ticker")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            title: market
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            yes_bid: market
                .get("yes_bid")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            yes_ask: market
                .get("yes_ask")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            volume: market.get("volume").and_then(|v| v.as_u64()).unwrap_or(0),
            open_interest: market
                .get("open_interest")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        }))
    }

    /// Parse markets array response
    fn parse_markets_response(&self, json_text: &str) -> Result<Vec<KalshiMarket>, SourceError> {
        let json: serde_json::Value = serde_json::from_str(json_text)
            .map_err(|e| SourceError::ParseError(format!("JSON parse error: {}", e)))?;

        let markets = json
            .get("markets")
            .and_then(|m| m.as_array())
            .ok_or_else(|| SourceError::ParseError("No 'markets' array".into()))?;

        let mut results = Vec::new();
        for market in markets {
            results.push(KalshiMarket {
                ticker: market
                    .get("ticker")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                title: market
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                yes_bid: market
                    .get("yes_bid")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                yes_ask: market
                    .get("yes_ask")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                volume: market.get("volume").and_then(|v| v.as_u64()).unwrap_or(0),
                open_interest: market
                    .get("open_interest")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            });
        }

        Ok(results)
    }

    /// Get market-implied probability for an event (public method for validation)
    pub fn get_market_prob(&self, ticker: &str) -> Result<f64, SourceError> {
        match self.fetch_market(ticker)? {
            Some(m) => {
                // Midpoint of bid/ask as probability estimate
                let prob = (m.yes_bid + m.yes_ask) / 200.0; // Kalshi uses 0-100 scale
                Ok(prob)
            }
            None => Err(SourceError::NotAvailable(format!(
                "Market {} not found",
                ticker
            ))),
        }
    }

    /// Search for conflict-related markets
    fn search_conflict_markets(&self) -> Result<Vec<KalshiMarket>, SourceError> {
        // Fetch open markets with higher limit
        let url = format!("{}/markets?status=open&limit=500", self.base_url);

        let response = reqwest::blocking::Client::new()
            .get(&url)
            .header("User-Agent", "Fractal/1.0")
            .send()
            .map_err(|e| SourceError::NetworkError(format!("HTTP error: {}", e)))?;

        if !response.status().is_success() {
            return Err(SourceError::NetworkError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let text = response
            .text()
            .map_err(|e| SourceError::NetworkError(format!("Read error: {}", e)))?;

        let all_markets = self.parse_markets_response(&text)?;

        // Keywords that indicate POLITICAL conflict (not sports)
        let include_keywords = [
            "civil war",
            "civil unrest",
            "political violence",
            "riot",
            "protest",
            "martial law",
            "insurrection",
            "coup",
            "government shutdown",
            "impeach",
            "state of emergency",
            "national guard deploy",
            "secession",
            "secede",
            "election",
            "contested",
            "certification",
            "capitol",
            "congress attack",
            "domestic terrorism",
            "political assassination",
            "mass shooting",
        ];

        // Keywords that indicate sports/entertainment (exclude)
        let exclude_keywords = [
            "nba",
            "nfl",
            "mlb",
            "nhl",
            "mls",
            "esports",
            "game",
            "match",
            "super bowl",
            "world series",
            "playoffs",
            "championship",
            "mvp",
            "points",
            "rebounds",
            "assists",
            "touchdowns",
            "goals",
            "wins",
            "team",
            "player",
            "draft",
            "trade",
            "contract",
            "coach",
            "box office",
            "movie",
            "oscar",
            "grammy",
            "emmy",
            "album",
            "crypto",
            "bitcoin",
            "ethereum",
            "stock",
            "s&p",
            "nasdaq",
            "weather",
            "temperature",
            "hurricane",
            "earthquake",
        ];

        let filtered: Vec<KalshiMarket> = all_markets
            .into_iter()
            .filter(|m| {
                let title_lower = m.title.to_lowercase();
                let ticker_lower = m.ticker.to_lowercase();

                // Must contain at least one political conflict keyword
                let has_include = include_keywords.iter().any(|kw| title_lower.contains(kw));

                // Must NOT contain sports/entertainment keywords
                let has_exclude = exclude_keywords
                    .iter()
                    .any(|kw| title_lower.contains(kw) || ticker_lower.contains(kw));

                // Also exclude multi-game extended parlays (sports betting)
                let is_parlay = ticker_lower.contains("multigame")
                    || ticker_lower.contains("extended")
                    || ticker_lower.contains("esports");

                has_include && !has_exclude && !is_parlay
            })
            .collect();

        Ok(filtered)
    }
}

impl DataSource for KalshiSource {
    fn name(&self) -> &str {
        "KALSHI"
    }

    fn fetch(&self) -> Result<SourceUpdate, SourceError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        eprintln!("[KALSHI] Fetching prediction market data...");

        let conflict_markets = self.search_conflict_markets()?;

        if conflict_markets.is_empty() {
            return Err(SourceError::NotAvailable(
                "No conflict-related markets found".into(),
            ));
        }

        // Compute average conflict probability
        let mut total_prob = 0.0;
        let mut weighted_count = 0.0;

        for market in &conflict_markets {
            let prob = (market.yes_bid + market.yes_ask) / 200.0;
            let weight = (market.volume as f64).sqrt().max(1.0); // Weight by sqrt of volume
            total_prob += prob * weight;
            weighted_count += weight;

            eprintln!(
                "  [{}] {} - {:.1}% (vol:{})",
                market.ticker,
                market.title,
                prob * 100.0,
                market.volume
            );
        }

        let avg_conflict_prob = if weighted_count > 0.0 {
            total_prob / weighted_count
        } else {
            0.0
        };

        eprintln!(
            "[KALSHI] {} conflict markets, avg implied prob: {:.1}%",
            conflict_markets.len(),
            avg_conflict_prob * 100.0
        );

        // Kalshi provides validation signal, not faction behavior directly
        // Map conflict probability to a general tension indicator
        let mut red_deltas = FactionBehavior::default();
        let mut blue_deltas = FactionBehavior::default();

        // If market predicts conflict, increase both factions' tension indicators
        red_deltas.rhetoric_hostility = avg_conflict_prob;
        blue_deltas.rhetoric_hostility = avg_conflict_prob;

        Ok(SourceUpdate {
            timestamp,
            red_deltas,
            blue_deltas,
            confidence: 0.7, // Prediction markets are wisdom of crowds
        })
    }

    fn max_staleness(&self) -> u64 {
        3600
    } // Hourly
}

// ═══════════════════════════════════════════════════════════════════════════════
// FACTION & ORACLE CORE
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Faction {
    Red,
    Blue,
}

/// RedBlue Oracle - main entry point
pub struct RedBlueOracle {
    /// Shepherd dynamics engine
    shepherd: ShepherdDynamics,
    /// Current RED faction state
    red: FactionBehavior,
    /// Current BLUE faction state
    blue: FactionBehavior,
    /// Data sources
    sources: Vec<Box<dyn DataSource>>,
    /// Source weights for fusion
    source_weights: HashMap<String, f64>,
    /// Historical Φ values for trend analysis
    phi_history: Vec<(f64, f64)>, // (timestamp, phi)
    /// Alert history
    alerts: Vec<OracleAlert>,
}

/// Oracle alert
#[derive(Debug, Clone)]
pub struct OracleAlert {
    pub timestamp: f64,
    pub level: AlertLevel,
    pub phi: f64,
    pub trend: Trend,
    pub message: String,
}

#[derive(Debug, Clone, Copy)]
pub enum Trend {
    Stable,
    Increasing,
    Decreasing,
    Inflection, // Variance inflection detected
}

impl RedBlueOracle {
    /// Create new oracle with default sources
    pub fn new() -> Self {
        let n_bins = 20;
        let mut shepherd =
            ShepherdDynamics::new(n_bins).with_variance_config(VarianceConfig::sensitive());

        // Register factions as actors
        shepherd.register_actor("USA_RED", None);
        shepherd.register_actor("USA_BLUE", None);

        // Default source weights
        let mut source_weights = HashMap::new();
        source_weights.insert("VOTEVIEW".into(), 1.0); // High weight - ground truth
        source_weights.insert("GDELT".into(), 0.7); // Medium - volume
        source_weights.insert("ACLED".into(), 0.8); // High - kinetic
        source_weights.insert("BLUESKY".into(), 0.4); // Low - noisy
        source_weights.insert("KALSHI".into(), 0.5); // Medium - validation

        Self {
            shepherd,
            red: FactionBehavior::default(),
            blue: FactionBehavior::default(),
            sources: Vec::new(),
            source_weights,
            phi_history: Vec::new(),
            alerts: Vec::new(),
        }
    }

    /// Add a data source
    pub fn add_source(&mut self, source: Box<dyn DataSource>) {
        self.sources.push(source);
    }

    /// Add default sources
    pub fn with_default_sources(mut self) -> Self {
        self.add_source(Box::new(VoteviewSource::new()));
        self.add_source(Box::new(GdeltSource::new(7)));
        self.add_source(Box::new(AcledSource::new(None)));
        self.add_source(Box::new(BlueskySource::new()));
        self.add_source(Box::new(KalshiSource::new(None)));
        self
    }

    /// Update from all sources (daily batch)
    pub fn update(&mut self) -> Result<OracleStatus, String> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut red_updates = Vec::new();
        let mut blue_updates = Vec::new();
        let mut errors = Vec::new();

        // Fetch from all sources
        for source in &self.sources {
            match source.fetch() {
                Ok(update) => {
                    let weight = self
                        .source_weights
                        .get(source.name())
                        .copied()
                        .unwrap_or(0.5);

                    red_updates.push((update.red_deltas.clone(), weight * update.confidence));
                    blue_updates.push((update.blue_deltas.clone(), weight * update.confidence));
                }
                Err(e) => {
                    errors.push(format!("{}: {}", source.name(), e));
                }
            }
        }

        if red_updates.is_empty() {
            return Err(format!("All sources failed: {:?}", errors));
        }

        // Weighted fusion (simplified - just average for now)
        // TODO: Implement proper Kalman-style fusion
        self.red = self.fuse_behaviors(&red_updates);
        self.blue = self.fuse_behaviors(&blue_updates);

        // Update Shepherd
        let n_bins = 20;
        let red_dist = self.red.to_distribution(n_bins);
        let blue_dist = self.blue.to_distribution(n_bins);

        let _red_alerts = self.shepherd.update_actor("USA_RED", &red_dist, timestamp);
        let _blue_alerts = self
            .shepherd
            .update_actor("USA_BLUE", &blue_dist, timestamp);

        // Get current Φ
        let potentials = self.shepherd.all_potentials();
        let phi = potentials
            .iter()
            .find(|p| {
                (p.actor_a == "USA_RED" && p.actor_b == "USA_BLUE")
                    || (p.actor_a == "USA_BLUE" && p.actor_b == "USA_RED")
            })
            .map(|p| p.phi)
            .unwrap_or(0.0);

        self.phi_history.push((timestamp, phi));

        // Compute trend
        let trend = self.compute_trend();

        // Determine alert level
        let level = self.compute_alert_level(phi, &trend);

        // Record alert if significant
        if level >= AlertLevel::Yellow {
            self.alerts.push(OracleAlert {
                timestamp,
                level: level.clone(),
                phi,
                trend,
                message: format!("RED/BLUE Φ={:.2}, trend={:?}", phi, trend),
            });
        }

        Ok(OracleStatus {
            timestamp,
            phi,
            trend,
            alert_level: level,
            source_errors: errors,
            red_behavior: self.red.clone(),
            blue_behavior: self.blue.clone(),
        })
    }

    /// Fuse multiple behavior updates with weights
    fn fuse_behaviors(&self, updates: &[(FactionBehavior, f64)]) -> FactionBehavior {
        let total_weight: f64 = updates.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return FactionBehavior::default();
        }

        let mut fused = FactionBehavior::default();

        for (behavior, weight) in updates {
            let w = weight / total_weight;
            fused.rhetoric_hostility += behavior.rhetoric_hostility * w;
            fused.rhetoric_dehumanization += behavior.rhetoric_dehumanization * w;
            fused.rhetoric_violence_adjacent += behavior.rhetoric_violence_adjacent * w;
            fused.rhetoric_delegitimization += behavior.rhetoric_delegitimization * w;
            fused.protest_frequency += behavior.protest_frequency * w;
            fused.counter_protest_frequency += behavior.counter_protest_frequency * w;
            fused.violence_incidents += behavior.violence_incidents * w;
            fused.militia_signals += behavior.militia_signals * w;
            fused.legislative_cooperation += behavior.legislative_cooperation * w;
            fused.legal_challenges += behavior.legal_challenges * w;
            fused.norm_violations += behavior.norm_violations * w;
            fused.election_challenges += behavior.election_challenges * w;
            fused.geographic_concentration += behavior.geographic_concentration * w;
            fused.state_defiance += behavior.state_defiance * w;
        }

        fused
    }

    /// Compute trend from Φ history
    fn compute_trend(&self) -> Trend {
        if self.phi_history.len() < 7 {
            return Trend::Stable;
        }

        let recent: Vec<f64> = self
            .phi_history
            .iter()
            .rev()
            .take(7)
            .map(|(_, phi)| *phi)
            .collect();

        let older: Vec<f64> = self
            .phi_history
            .iter()
            .rev()
            .skip(7)
            .take(7)
            .map(|(_, phi)| *phi)
            .collect();

        if older.is_empty() {
            return Trend::Stable;
        }

        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;

        let delta = recent_avg - older_avg;

        // Check for variance inflection
        let recent_var = variance(&recent);
        let older_var = variance(&older);
        if recent_var < older_var * 0.5 && recent_avg > 1.0 {
            // Variance dropped while Φ elevated - potential nucleation
            return Trend::Inflection;
        }

        if delta > 0.1 {
            Trend::Increasing
        } else if delta < -0.1 {
            Trend::Decreasing
        } else {
            Trend::Stable
        }
    }

    /// Compute alert level from Φ and trend
    fn compute_alert_level(&self, phi: f64, trend: &Trend) -> AlertLevel {
        match trend {
            Trend::Inflection => AlertLevel::Red, // Nucleation signature
            _ if phi > 2.0 => AlertLevel::Red,
            _ if phi > 1.5 && matches!(trend, Trend::Increasing) => AlertLevel::Orange,
            _ if phi > 1.0 => AlertLevel::Yellow,
            _ => AlertLevel::Green,
        }
    }

    /// Get current status without updating
    pub fn status(&self) -> Option<OracleStatus> {
        self.phi_history.last().map(|(ts, phi)| OracleStatus {
            timestamp: *ts,
            phi: *phi,
            trend: self.compute_trend(),
            alert_level: self.compute_alert_level(*phi, &self.compute_trend()),
            source_errors: vec![],
            red_behavior: self.red.clone(),
            blue_behavior: self.blue.clone(),
        })
    }

    /// Get alert history
    pub fn alerts(&self) -> &[OracleAlert] {
        &self.alerts
    }
}

/// Current oracle status
#[derive(Debug, Clone)]
pub struct OracleStatus {
    pub timestamp: f64,
    pub phi: f64,
    pub trend: Trend,
    pub alert_level: AlertLevel,
    pub source_errors: Vec<String>,
    pub red_behavior: FactionBehavior,
    pub blue_behavior: FactionBehavior,
}

impl OracleStatus {
    pub fn display(&self) -> String {
        let level_color = match self.alert_level {
            AlertLevel::Red => "\x1b[31m",
            AlertLevel::Orange => "\x1b[33m",
            AlertLevel::Yellow => "\x1b[93m",
            AlertLevel::Green => "\x1b[32m",
        };

        format!(
            "{}[{:?}]\x1b[0m Φ(RED,BLUE)={:.3} trend={:?}",
            level_color, self.alert_level, self.phi, self.trend
        )
    }
}

/// Compute variance of a slice
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

/// Run the RedBlue oracle monitor
pub async fn run_monitor() -> anyhow::Result<()> {
    println!("\x1b[31m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[31m RED\x1b[0m/\x1b[34mBLUE\x1b[0m DOMESTIC CONFLICT ORACLE");
    println!("\x1b[31m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();

    let mut oracle = RedBlueOracle::new().with_default_sources();

    println!("Data sources configured:");
    for source in &oracle.sources {
        println!("  • {}", source.name());
    }
    println!();

    println!("Fetching data and computing Φ(RED,BLUE)...");
    println!();

    match oracle.update() {
        Ok(status) => {
            println!("{}", status.display());
            println!();

            if !status.source_errors.is_empty() {
                println!("\x1b[33mSource errors:\x1b[0m");
                for err in &status.source_errors {
                    println!("  • {}", err);
                }
                println!();
            }

            println!("\x1b[36mRED Faction Behavior:\x1b[0m");
            println!(
                "  rhetoric_hostility:      {:.2}",
                status.red_behavior.rhetoric_hostility
            );
            println!(
                "  rhetoric_delegitimize:   {:.2}",
                status.red_behavior.rhetoric_delegitimization
            );
            println!(
                "  protest_frequency:       {:.2}",
                status.red_behavior.protest_frequency
            );
            println!(
                "  violence_incidents:      {:.2}",
                status.red_behavior.violence_incidents
            );
            println!();

            println!("\x1b[34mBLUE Faction Behavior:\x1b[0m");
            println!(
                "  rhetoric_hostility:      {:.2}",
                status.blue_behavior.rhetoric_hostility
            );
            println!(
                "  rhetoric_delegitimize:   {:.2}",
                status.blue_behavior.rhetoric_delegitimization
            );
            println!(
                "  protest_frequency:       {:.2}",
                status.blue_behavior.protest_frequency
            );
            println!(
                "  violence_incidents:      {:.2}",
                status.blue_behavior.violence_incidents
            );
        }
        Err(e) => {
            println!("\x1b[31mOracle update failed: {}\x1b[0m", e);
            println!();
            println!(
                "Data sources not yet implemented. Run with real data sources to see results."
            );
        }
    }

    println!();
    println!("\x1b[31m═══════════════════════════════════════════════════════════════\x1b[0m");

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// BACKTEST — Validate oracle against historical data
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a single backtest window
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub cutoff_date: String,
    pub phi_before: f64,
    pub conflict_events_after: u64,
    pub total_events_after: u64,
    pub red_violence_after: u64,
    pub blue_violence_after: u64,
}

/// Country codes for GDELT filtering
pub struct CountryCode {
    pub geo: &'static str,   // 2-letter ActionGeo code
    pub actor: &'static str, // 3-letter Actor country code
    pub name: &'static str,  // Display name
}

pub const COUNTRY_US: CountryCode = CountryCode {
    geo: "US",
    actor: "USA",
    name: "United States",
};
pub const COUNTRY_CA: CountryCode = CountryCode {
    geo: "CA",
    actor: "CAN",
    name: "Canada",
};
pub const COUNTRY_UK: CountryCode = CountryCode {
    geo: "UK",
    actor: "GBR",
    name: "United Kingdom",
};
pub const COUNTRY_DE: CountryCode = CountryCode {
    geo: "GM",
    actor: "DEU",
    name: "Germany",
};
pub const COUNTRY_FR: CountryCode = CountryCode {
    geo: "FR",
    actor: "FRA",
    name: "France",
};

/// Run backtest on historical GDELT data
pub async fn run_backtest(
    cutoff_date: &str, // Format: YYYYMMDD
    days_before: usize,
    days_after: usize,
) -> anyhow::Result<BacktestResult> {
    run_backtest_country(cutoff_date, days_before, days_after, &COUNTRY_US).await
}

/// Run backtest on historical GDELT data for a specific country
pub async fn run_backtest_country(
    cutoff_date: &str, // Format: YYYYMMDD
    days_before: usize,
    days_after: usize,
    country: &CountryCode,
) -> anyhow::Result<BacktestResult> {
    use chrono::{Duration, NaiveDate};

    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!(
        "\x1b[36m BACKTEST — {} Oracle Validation\x1b[0m",
        country.name
    );
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();
    println!(
        "Country: {} (geo={}, actor={})",
        country.name, country.geo, country.actor
    );
    println!("Cutoff date: {}", cutoff_date);
    println!("Training window: {} days before cutoff", days_before);
    println!("Validation window: {} days after cutoff", days_after);
    println!();

    let cutoff = NaiveDate::parse_from_str(cutoff_date, "%Y%m%d")
        .map_err(|e| anyhow::anyhow!("Invalid date format: {}", e))?;

    // Generate date lists
    let mut before_dates = Vec::new();
    let mut after_dates = Vec::new();

    for i in 1..=days_before {
        let d = cutoff - Duration::days(i as i64);
        before_dates.push(d.format("%Y%m%d").to_string());
    }

    for i in 1..=days_after {
        let d = cutoff + Duration::days(i as i64);
        after_dates.push(d.format("%Y%m%d").to_string());
    }

    println!(
        "\x1b[33mPhase 1: Fetching training data ({} days before cutoff)...\x1b[0m",
        days_before
    );

    // Fetch BEFORE data and compute Φ
    let gdelt_before = GdeltSource::new(1);
    let mut all_events_before = Vec::new();
    let mut fetch_errors_before = 0;

    for date in &before_dates {
        match gdelt_before.fetch_day_for_country(date, country.geo, country.actor) {
            Ok(events) => {
                all_events_before.extend(events);
                print!(".");
            }
            Err(_) => {
                fetch_errors_before += 1;
                print!("x");
            }
        }
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    println!(
        " done ({} events, {} errors)",
        all_events_before.len(),
        fetch_errors_before
    );

    // Compute Φ from before data
    let (red_before, blue_before) = gdelt_before.compute_faction_metrics(&all_events_before);

    let phi_before = compute_phi_from_metrics(&red_before, &blue_before);

    println!();
    println!("\x1b[32mTraining period results:\x1b[0m");
    println!(
        "  LEFT events:  {} (violence: {})",
        red_before.event_count, red_before.violence_events
    );
    println!(
        "  RIGHT events: {} (violence: {})",
        blue_before.event_count, blue_before.violence_events
    );
    println!(
        "  \x1b[33mMCSI = {:.4}\x1b[0m (Media Conflict Salience Index)",
        phi_before
    );
    println!();

    println!(
        "\x1b[33mPhase 2: Fetching validation data ({} days after cutoff)...\x1b[0m",
        days_after
    );

    // Fetch AFTER data to validate
    let gdelt_after = GdeltSource::new(1);
    let mut all_events_after = Vec::new();
    let mut fetch_errors_after = 0;

    for date in &after_dates {
        match gdelt_after.fetch_day_for_country(date, country.geo, country.actor) {
            Ok(events) => {
                all_events_after.extend(events);
                print!(".");
            }
            Err(_) => {
                fetch_errors_after += 1;
                print!("x");
            }
        }
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    println!(
        " done ({} events, {} errors)",
        all_events_after.len(),
        fetch_errors_after
    );

    // Compute actual conflict events in validation period
    let (red_after, blue_after) = gdelt_after.compute_faction_metrics(&all_events_after);

    let conflict_events = red_after.violence_events
        + blue_after.violence_events
        + red_after.coercion_events
        + blue_after.coercion_events;

    println!();
    println!("\x1b[32mValidation period results:\x1b[0m");
    println!(
        "  RED events:  {} (violence: {}, coercion: {})",
        red_after.event_count, red_after.violence_events, red_after.coercion_events
    );
    println!(
        "  BLUE events: {} (violence: {}, coercion: {})",
        blue_after.event_count, blue_after.violence_events, blue_after.coercion_events
    );
    println!("  Total conflict events: {}", conflict_events);
    println!();

    // Prediction assessment
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m PREDICTION ASSESSMENT\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");

    // Volume-adjusted thresholds
    let prediction = if phi_before > 3.5 {
        "HIGH conflict potential"
    } else if phi_before > 2.5 {
        "MEDIUM conflict potential"
    } else {
        "LOW conflict potential"
    };

    let per_day_conflict = conflict_events as f64 / days_after as f64;
    let actual = if per_day_conflict > 50.0 {
        "HIGH actual conflict"
    } else if per_day_conflict > 10.0 {
        "MEDIUM actual conflict"
    } else {
        "LOW actual conflict"
    };

    println!("  Oracle predicted: {}", prediction);
    println!(
        "  Actual outcome:   {} ({:.1} events/day)",
        actual, per_day_conflict
    );

    let match_result = (phi_before > 3.5 && per_day_conflict > 50.0)
        || (phi_before > 2.5
            && phi_before <= 3.5
            && per_day_conflict > 10.0
            && per_day_conflict <= 50.0)
        || (phi_before <= 2.5 && per_day_conflict <= 10.0);

    if match_result {
        println!("  \x1b[32m✓ PREDICTION MATCHED\x1b[0m");
    } else {
        println!("  \x1b[31m✗ PREDICTION MISSED\x1b[0m");
    }
    println!();

    Ok(BacktestResult {
        cutoff_date: cutoff_date.to_string(),
        phi_before,
        conflict_events_after: conflict_events,
        total_events_after: all_events_after.len() as u64,
        red_violence_after: red_after.violence_events,
        blue_violence_after: blue_after.violence_events,
    })
}

/// Compute MCSI (Media Conflict Salience Index) from faction metrics
///
/// Note: Despite taking RED/BLUE metrics, the null test showed faction labels
/// add no signal. This is effectively measuring total conflict salience.
/// Faction decomposition is retained for backward compatibility but is not meaningful.
fn compute_phi_from_metrics(red: &FactionMetrics, blue: &FactionMetrics) -> f64 {
    let red_total = (red.event_count + 1) as f64;
    let blue_total = (blue.event_count + 1) as f64;
    let total_events = red_total + blue_total;

    // === INTENSITY METRICS (ratio-based) ===

    // Violence/coercion rates
    let red_violence_rate = (red.violence_events + red.coercion_events) as f64 / red_total;
    let blue_violence_rate = (blue.violence_events + blue.coercion_events) as f64 / blue_total;
    let violence_intensity = red_violence_rate + blue_violence_rate;

    // Tone divergence
    let red_tone = if red.event_count > 0 {
        red.tone_sum / red.event_count as f64
    } else {
        0.0
    };
    let blue_tone = if blue.event_count > 0 {
        blue.tone_sum / blue.event_count as f64
    } else {
        0.0
    };
    let tone_divergence = (red_tone - blue_tone).abs() / 10.0;

    // Protest asymmetry
    let red_protest_rate = red.protest_events as f64 / red_total;
    let blue_protest_rate = blue.protest_events as f64 / blue_total;
    let protest_asymmetry = (red_protest_rate - blue_protest_rate).abs();

    // Faction size asymmetry
    let faction_asymmetry = (red_total - blue_total).abs() / (red_total + blue_total);

    // === VOLUME SCALING ===
    // log10 scaling: 100 events = 2, 1000 = 3, 10000 = 4, 100000 = 5
    let volume_scale = (total_events.max(1.0)).log10();

    // Normalize volume to ~0-1 range (assuming 100k events is "full scale")
    let volume_factor = (volume_scale / 5.0).min(1.0);

    // === ABSOLUTE CONFLICT COUNT ===
    let absolute_violence = (red.violence_events + blue.violence_events) as f64;
    let absolute_coercion = (red.coercion_events + blue.coercion_events) as f64;
    let absolute_conflict = absolute_violence + absolute_coercion;

    // Log-scaled absolute conflict (1 = 10 events, 2 = 100, 3 = 1000)
    let conflict_magnitude = (absolute_conflict.max(1.0)).log10();

    // === COMBINED Φ (volume-adjusted) ===
    // Base intensity score
    let intensity = violence_intensity * 3.0
        + tone_divergence * 1.5
        + protest_asymmetry * 2.0
        + faction_asymmetry * 0.5;

    // Volume-adjusted Φ: intensity × volume_factor + conflict_magnitude
    let phi = intensity * volume_factor + conflict_magnitude;

    phi
}

/// Run multi-window backtest to build correlation (US default)
pub async fn run_backtest_series(
    start_date: &str,
    num_windows: usize,
    window_gap_days: usize,
    days_before: usize,
    days_after: usize,
) -> anyhow::Result<Vec<BacktestResult>> {
    run_backtest_series_country(
        start_date,
        num_windows,
        window_gap_days,
        days_before,
        days_after,
        &COUNTRY_US,
    )
    .await
}

/// Run multi-window backtest for a specific country
pub async fn run_backtest_series_country(
    start_date: &str, // Format: YYYYMMDD
    num_windows: usize,
    window_gap_days: usize,
    days_before: usize,
    days_after: usize,
    country: &CountryCode,
) -> anyhow::Result<Vec<BacktestResult>> {
    use chrono::{Duration, NaiveDate};

    println!("\x1b[35m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!(
        "\x1b[35m BACKTEST SERIES — {} ({} windows)\x1b[0m",
        country.name, num_windows
    );
    println!("\x1b[35m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();

    let start = NaiveDate::parse_from_str(start_date, "%Y%m%d")
        .map_err(|e| anyhow::anyhow!("Invalid date format: {}", e))?;

    let mut results = Vec::new();

    for i in 0..num_windows {
        let cutoff = start + Duration::days((i * window_gap_days) as i64);
        let cutoff_str = cutoff.format("%Y%m%d").to_string();

        println!(
            "\n\x1b[33m[Window {}/{}] Cutoff: {}\x1b[0m",
            i + 1,
            num_windows,
            cutoff_str
        );

        match run_backtest_country(&cutoff_str, days_before, days_after, country).await {
            Ok(result) => results.push(result),
            Err(e) => println!("  Error: {}", e),
        }
    }

    // Summary statistics
    if !results.is_empty() {
        println!(
            "\n\x1b[35m═══════════════════════════════════════════════════════════════\x1b[0m"
        );
        println!("\x1b[35m SERIES SUMMARY\x1b[0m");
        println!("\x1b[35m═══════════════════════════════════════════════════════════════\x1b[0m");

        println!("\n  Cutoff          MCSI       Conflict Events");
        println!("  ─────────────────────────────────────────────");
        for r in &results {
            println!(
                "  {}    {:.4}      {}",
                r.cutoff_date, r.phi_before, r.conflict_events_after
            );
        }

        // Compute correlation
        let n = results.len() as f64;
        let phi_mean: f64 = results.iter().map(|r| r.phi_before).sum::<f64>() / n;
        let conflict_mean: f64 = results
            .iter()
            .map(|r| r.conflict_events_after as f64)
            .sum::<f64>()
            / n;

        let mut cov = 0.0;
        let mut var_phi = 0.0;
        let mut var_conflict = 0.0;

        for r in &results {
            let phi_dev = r.phi_before - phi_mean;
            let conflict_dev = r.conflict_events_after as f64 - conflict_mean;
            cov += phi_dev * conflict_dev;
            var_phi += phi_dev * phi_dev;
            var_conflict += conflict_dev * conflict_dev;
        }

        let correlation = if var_phi > 0.0 && var_conflict > 0.0 {
            cov / (var_phi.sqrt() * var_conflict.sqrt())
        } else {
            0.0
        };

        println!(
            "\n  \x1b[36mCorrelation(MCSI, conflict_events) = {:.4}\x1b[0m",
            correlation
        );

        if correlation > 0.5 {
            println!("  \x1b[32m✓ STRONG POSITIVE — Oracle has predictive power\x1b[0m");
        } else if correlation > 0.2 {
            println!("  \x1b[33m~ WEAK POSITIVE — Some signal, needs refinement\x1b[0m");
        } else if correlation > -0.2 {
            println!("  \x1b[31m✗ NO CORRELATION — Oracle not predictive\x1b[0m");
        } else {
            println!("  \x1b[31m✗ NEGATIVE — Oracle inversely predictive (wrong)\x1b[0m");
        }
    }

    Ok(results)
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERMUTATION NULL TEST (L6 Mitigation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of permutation null test
#[derive(Debug, Clone)]
pub struct NullTestResult {
    pub observed_phi: f64,
    pub null_mean: f64,
    pub null_std: f64,
    pub z_score: f64,
    pub p_value: f64,
    pub permutations: usize,
    pub null_distribution: Vec<f64>,
}

/// Run permutation null test on GDELT data
///
/// This shuffles faction labels (which events are RED vs BLUE) and recomputes Φ
/// to establish whether observed Φ is significantly above random noise.
pub async fn run_null_test(days: usize, permutations: usize) -> anyhow::Result<NullTestResult> {
    run_null_test_country(days, permutations, &COUNTRY_US).await
}

/// Run permutation null test for a specific country
pub async fn run_null_test_country(
    days: usize,
    permutations: usize,
    country: &CountryCode,
) -> anyhow::Result<NullTestResult> {
    println!("\x1b[35m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!(
        "\x1b[35m PERMUTATION NULL TEST — {} ({} permutations)\x1b[0m",
        country.name, permutations
    );
    println!("\x1b[35m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();
    println!("Testing H0: Observed MCSI is indistinguishable from random faction assignment");
    println!("(Only events from classifiable sources are shuffled)");
    println!();

    // Fetch GDELT data
    let gdelt = GdeltSource::new(days);
    let today = chrono::Utc::now().date_naive();

    println!(
        "[1/5] Fetching {} days of GDELT data for {}...",
        days, country.name
    );

    let mut all_events = Vec::new();
    for i in (1..=days).rev() {
        let date = today - chrono::Duration::days(i as i64);
        let date_str = date.format("%Y%m%d").to_string();

        match gdelt.fetch_day_for_country(&date_str, country.geo, country.actor) {
            Ok(events) => {
                all_events.extend(events);
                print!(".");
            }
            Err(_) => print!("x"),
        }
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    println!(" {} total events", all_events.len());

    if all_events.is_empty() {
        anyhow::bail!("No events fetched");
    }

    // Filter to only events that ARE classifiable (have a RED or BLUE source)
    println!("[2/5] Filtering to classifiable events only...");

    // Tag each event with its real faction (if any)
    #[derive(Clone)]
    #[allow(dead_code)]
    struct TaggedEvent {
        faction: Option<Faction>,
        goldstein: f64,
        avg_tone: f64,
        num_mentions: u32,
        event_code: String,
    }

    let mut classified_events: Vec<TaggedEvent> = Vec::new();
    let mut red_count = 0usize;
    let mut blue_count = 0usize;

    for event in &all_events {
        let source_faction = GdeltSource::classify_source(&event.source_url);
        let actor_faction = GdeltSource::classify_actor(&event.actor1_name, &event.actor1_type)
            .or_else(|| GdeltSource::classify_actor(&event.actor2_name, &event.actor2_type));
        let faction = source_faction.or(actor_faction);

        if faction.is_some() {
            match faction {
                Some(Faction::Red) => red_count += 1,
                Some(Faction::Blue) => blue_count += 1,
                _ => {}
            }
            classified_events.push(TaggedEvent {
                faction,
                goldstein: event.goldstein,
                avg_tone: event.avg_tone,
                num_mentions: event.num_mentions,
                event_code: event.event_code.clone(),
            });
        }
    }

    println!(
        "  {} classifiable events ({} RED, {} BLUE)",
        classified_events.len(),
        red_count,
        blue_count
    );

    if classified_events.len() < 100 {
        anyhow::bail!(
            "Too few classifiable events ({}) for meaningful null test",
            classified_events.len()
        );
    }

    // Compute observed MCSI with real faction labels
    println!("[3/5] Computing observed MCSI with real faction labels...");
    let (red_real, blue_real) = gdelt.compute_faction_metrics(&all_events);
    let observed_phi = compute_phi_from_metrics(&red_real, &blue_real);
    println!("  Observed MCSI = {:.4}", observed_phi);
    println!(
        "  RED events: {} (violence: {}, coercion: {})",
        red_real.event_count, red_real.violence_events, red_real.coercion_events
    );
    println!(
        "  BLUE events: {} (violence: {}, coercion: {})",
        blue_real.event_count, blue_real.violence_events, blue_real.coercion_events
    );
    println!();

    // Run permutation test - shuffle faction labels among classified events only
    println!(
        "[4/5] Running {} permutations (shuffling faction labels among classified events)...",
        permutations
    );

    let mut null_distribution = Vec::with_capacity(permutations);

    // Use system time as base seed
    let base_seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Pre-compute the fraction that should be RED (to maintain marginal distribution)
    let red_fraction = red_count as f64 / classified_events.len() as f64;

    for p in 0..permutations {
        // xorshift64 PRNG with unique seed per permutation
        let mut state = base_seed
            .wrapping_add(p as u64)
            .wrapping_mul(2685821657736338717);
        let mut random_float = || -> f64 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state = state.wrapping_mul(0x2545F4914F6CDD1D);
            (state as f64) / (u64::MAX as f64)
        };

        // Compute metrics with shuffled faction assignment
        // Maintain the same RED/BLUE ratio as observed (stratified permutation)
        let mut red_perm = FactionMetrics::default();
        let mut blue_perm = FactionMetrics::default();

        for event in &classified_events {
            // Randomly assign to RED with probability = observed RED fraction
            let is_red = random_float() < red_fraction;
            let metrics = if is_red {
                &mut red_perm
            } else {
                &mut blue_perm
            };

            metrics.event_count += 1;
            metrics.total_mentions += event.num_mentions as u64;
            metrics.goldstein_sum += event.goldstein;
            metrics.tone_sum += event.avg_tone;

            // Categorize by CAMEO event code
            let code_prefix: u32 = event
                .event_code
                .chars()
                .take(2)
                .collect::<String>()
                .parse()
                .unwrap_or(0);

            match code_prefix {
                14 => metrics.protest_events += 1,
                17 | 18 => metrics.coercion_events += 1,
                19 | 20 => metrics.violence_events += 1,
                _ => {}
            }
        }

        let phi_perm = compute_phi_from_metrics(&red_perm, &blue_perm);
        null_distribution.push(phi_perm);

        // Progress indicator
        if (p + 1) % (permutations / 10).max(1) == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    println!(" done");
    println!();

    // Compute statistics
    println!("[5/5] Computing null distribution statistics...");

    let n = null_distribution.len() as f64;
    let null_mean: f64 = null_distribution.iter().sum::<f64>() / n;

    let variance: f64 = null_distribution
        .iter()
        .map(|x| (x - null_mean).powi(2))
        .sum::<f64>()
        / n;
    let null_std = variance.sqrt();

    let z_score = if null_std > 0.0 {
        (observed_phi - null_mean) / null_std
    } else {
        0.0
    };

    // Compute empirical p-value (fraction of null samples >= observed)
    let exceeds_count = null_distribution
        .iter()
        .filter(|&&x| x >= observed_phi)
        .count();
    let p_value = exceeds_count as f64 / n;

    // Sort for percentile reporting
    let mut sorted_null = null_distribution.clone();
    sorted_null.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p5 = sorted_null[(n * 0.05) as usize];
    let p25 = sorted_null[(n * 0.25) as usize];
    let p50 = sorted_null[(n * 0.50) as usize];
    let p75 = sorted_null[(n * 0.75) as usize];
    let p95 = sorted_null[(n * 0.95) as usize];

    println!();
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!("\x1b[36m NULL TEST RESULTS\x1b[0m");
    println!("\x1b[36m═══════════════════════════════════════════════════════════════\x1b[0m");
    println!();
    println!("  Observed MCSI:  {:.4}", observed_phi);
    println!("  Null mean:      {:.4}", null_mean);
    println!("  Null std:       {:.4}", null_std);
    println!();
    println!("  \x1b[33mZ-score:         {:.2}\x1b[0m", z_score);
    println!(
        "  \x1b[33mP-value:         {:.4}\x1b[0m (empirical, one-tailed)",
        p_value
    );
    println!();
    println!("  Null distribution percentiles:");
    println!("    5th:  {:.4}", p5);
    println!("    25th: {:.4}", p25);
    println!("    50th: {:.4} (median)", p50);
    println!("    75th: {:.4}", p75);
    println!("    95th: {:.4}", p95);
    println!();

    // Interpretation
    if z_score > 2.0 {
        println!("  \x1b[32m✓ SIGNIFICANT (z > 2.0)\x1b[0m");
        println!("    Observed MCSI is significantly above null distribution.");
        println!("    Faction labels carry real signal beyond random noise.");
    } else if z_score > 1.5 {
        println!("  \x1b[33m~ MARGINAL (1.5 < z < 2.0)\x1b[0m");
        println!("    Observed MCSI is marginally above null distribution.");
        println!("    Some signal present but not strongly significant.");
    } else {
        println!("  \x1b[31m✗ NOT SIGNIFICANT (z < 1.5)\x1b[0m");
        println!("    Observed MCSI overlaps null distribution.");
        println!("    Cannot reject H0: faction labels may not carry meaningful signal.");
    }
    println!();

    // L6 falsification check
    println!("  \x1b[36mL6 Falsification criterion: z > 2.0\x1b[0m");
    if z_score > 2.0 {
        println!("  \x1b[32m✓ PASSED — Faction decomposition validated\x1b[0m");
    } else {
        println!("  \x1b[31m✗ FAILED — Faction labels no better than random\x1b[0m");
        println!("    MCSI is valid as total conflict salience index.");
        println!("    RED/BLUE decomposition adds no signal.");
    }
    println!();

    Ok(NullTestResult {
        observed_phi,
        null_mean,
        null_std,
        z_score,
        p_value,
        permutations,
        null_distribution,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faction_behavior_to_distribution() {
        let mut behavior = FactionBehavior::default();
        behavior.rhetoric_hostility = 0.8;
        behavior.violence_incidents = 0.5;

        let dist = behavior.to_distribution(10);
        assert_eq!(dist.len(), 10);

        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Distribution should sum to 1.0");
    }

    #[test]
    fn test_oracle_creation() {
        let oracle = RedBlueOracle::new();
        assert!(oracle.phi_history.is_empty());
        assert!(oracle.alerts.is_empty());
    }

    #[test]
    fn test_trend_computation() {
        let mut oracle = RedBlueOracle::new();

        // Add stable history
        for i in 0..14 {
            oracle.phi_history.push((i as f64, 0.5));
        }

        assert!(matches!(oracle.compute_trend(), Trend::Stable));

        // Add increasing history
        oracle.phi_history.clear();
        for i in 0..14 {
            oracle.phi_history.push((i as f64, 0.5 + (i as f64 * 0.1)));
        }

        assert!(matches!(oracle.compute_trend(), Trend::Increasing));
    }
}
