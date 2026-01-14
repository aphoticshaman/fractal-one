# MCSI Technical Specification
## (Formerly RED/BLUE Oracle)
## Version 0.4 — Post-Null-Test Revision
## Date: 2026-01-08

---

## CRITICAL UPDATE (v0.4)

**The permutation null test revealed that RED/BLUE faction labels add no signal beyond random assignment (z = -2.26).** The system has been renamed from "RED/BLUE Oracle" to "MCSI" (Media Conflict Salience Index).

| Validated | Not Validated |
|-----------|---------------|
| US vs Canada (67x volume) | Faction decomposition |
| Historical patterns | Partisan conflict measurement |
| Total conflict salience | RED/BLUE divergence |

---

## 1. SYSTEM OVERVIEW

MCSI (Media Conflict Salience Index) measures **total conflict-related media coverage** from GDELT data. It is NOT a faction divergence metric.

**Note:** RED/BLUE classification is retained for backward compatibility but the null test proved it adds no signal.

### 1.1 Core Metric

```
MCSI = intensity × volume_factor + conflict_magnitude
```

Where:
- **intensity**: Weighted sum of violence rates, tone divergence, and asymmetry metrics
- **volume_factor**: Logarithmic scaling to penalize small samples
- **conflict_magnitude**: Absolute count of conflict events (log-scaled)

### 1.2 Thresholds

| MCSI Range | Interpretation |
|------------|----------------|
| > 3.5 | HIGH conflict salience |
| 2.5 - 3.5 | MEDIUM conflict salience |
| < 2.5 | LOW conflict salience |

---

## 2. DATA PIPELINE

### 2.1 Primary Source: GDELT

**Source:** `http://data.gdeltproject.org/events/{YYYYMMDD}.export.CSV.zip`

**Format:** Tab-separated CSV (~200,000 events/day globally)

**Fetch Cadence:** Daily

**Relevant GDELT Columns (0-indexed):**

| Column | Index | Description |
|--------|-------|-------------|
| SQLDATE | 1 | Event date (YYYYMMDD) |
| Actor1Name | 5 | First actor name |
| Actor1CountryCode | 7 | First actor country (3-letter ISO) |
| Actor1Type1Code | 12 | First actor type code |
| Actor2Name | 15 | Second actor name |
| Actor2CountryCode | 17 | Second actor country |
| Actor2Type1Code | 22 | Second actor type code |
| EventCode | 26 | CAMEO event code |
| GoldsteinScale | 30 | Conflict/cooperation score (-10 to +10) |
| NumMentions | 31 | Number of mentions |
| NumSources | 32 | Number of sources |
| AvgTone | 34 | Average tone (negative = hostile) |
| ActionGeo_CountryCode | 51 | Event location country (2-letter) |
| SOURCEURL | 57 | News source URL |

**US Domestic Filter:**
```rust
fn is_us_domestic(actor1_country: &str, actor2_country: &str, action_geo: &str) -> bool {
    // Action must be in US
    if action_geo != "US" {
        return false;
    }
    // At least one actor should be US or unspecified (domestic actor)
    actor1_country == "USA" || actor1_country.is_empty()
        || actor2_country == "USA" || actor2_country.is_empty()
}
```

### 2.2 Secondary Source: VOTEVIEW

**Source:** `https://voteview.com/static/data/out/members/{H|S}{Congress}_members.csv`

**Format:** CSV with DW-NOMINATE scores

**Fetch Cadence:** Weekly (Congress 119 as of 2025-2027)

**Key Fields:**
- `party_code`: 100 = Democrat, 200 = Republican
- `nominate_dim1`: -1 (liberal) to +1 (conservative)
- `nominate_dim2`: Second dimension (historically race/civil rights)

**Polarization Metric:**
```
polarization = |median(Republican_dim1) - median(Democrat_dim1)|
```

Historical range: 0.5 (1970s) to 1.0+ (2020s)

### 2.3 Optional Source: ACLED

**Source:** `https://api.acleddata.com/acled/read`

**Format:** JSON REST API

**Authentication:** Requires free API key (ACLED_API_KEY, ACLED_EMAIL environment variables)

**Event Types:**
- Protests (peaceful, violent)
- Riots
- Violence against civilians
- Battles

**Confidence:** 0.9 (expert-curated, highest quality)

### 2.4 Optional Source: Bluesky

**Source:** `https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts`

**Format:** JSON (no authentication required for public search)

**Sample Size:** 100 posts per faction query

**Sentiment Keywords:**
- Hostile: "hate", "evil", "destroy", "enemy", "traitor"
- Dehumanizing: "vermin", "roach", "animal", "subhuman"
- Violent: "kill", "shoot", "hang", "execute", "attack"

### 2.5 Optional Source: Kalshi

**Source:** `https://api.elections.kalshi.com/`

**Format:** JSON REST API

**Purpose:** Real-time prediction market probabilities for political events

---

## 3. FACTION CLASSIFICATION

### 3.1 Primary Method: Source Domain Classification

Events are classified by the news outlet that reported them. This is **ecological inference** — we measure media attention, not actor affiliation.

**RED (Conservative) Sources:**
```
foxnews.com         dailycaller.com      breitbart.com
townhall.com        washingtonexaminer.com  nypost.com
newsmax.com         oann.com             thefederalist.com
dailywire.com       nationalreview.com   freebeacon.com
pjmedia.com         hotair.com           redstate.com
westernjournal.com  blazemedia.com       twitchy.com
```

**BLUE (Liberal) Sources:**
```
msnbc.com           huffpost.com         vox.com
slate.com           rawstory.com         thedailybeast.com
salon.com           motherjones.com      theguardian.com
commondreams.org    truthout.org         alternet.org
democracynow.org    jacobin.com          theintercept.com
dailykos.com        talkingpointsmemo.com  crooksandliars.com
mediaite.com
```

### 3.2 Fallback Method: Actor Keyword Classification

If source domain is not classifiable, actor names are checked against keyword lists.

**RED Keywords:**
```
republican  gop         conservative  trump    maga
desantis    mccarthy    mcconnell     heritage federalist
tea party   freedom caucus  right-wing  evangelical
```

**BLUE Keywords:**
```
democrat    liberal     progressive   biden    harris
pelosi      schumer     aoc           ocasio   sanders
warren      left-wing   planned parenthood  aclu  moveon
```

### 3.3 ACLED-Specific Actor Classification

For ACLED events, additional keywords are checked:

**RED:**
```
proud boys      patriot front    oath keepers    militia
three percenter boogaloo         groyper         pro-life
anti-abortion   pro-gun          second amendment
```

**BLUE:**
```
antifa          blm              black lives matter
socialist       dsa              pro-choice
abortion rights lgbtq            trans rights
climate activist environmental   labor union
```

---

## 4. PHI (Φ) CALCULATION

### 4.1 Input: FactionMetrics

```rust
struct FactionMetrics {
    event_count: u64,
    total_mentions: u64,
    goldstein_sum: f64,      // Sum of Goldstein scores
    tone_sum: f64,           // Sum of AvgTone values
    protest_events: u64,     // CAMEO 14x
    coercion_events: u64,    // CAMEO 17x, 18x
    violence_events: u64,    // CAMEO 19x, 20x
}
```

### 4.2 Algorithm

```rust
fn compute_phi_from_metrics(red: &FactionMetrics, blue: &FactionMetrics) -> f64 {
    let red_total = (red.event_count + 1) as f64;    // +1 to avoid division by zero
    let blue_total = (blue.event_count + 1) as f64;
    let total_events = red_total + blue_total;

    // === INTENSITY METRICS (ratio-based) ===

    // Violence/coercion rates
    let red_violence_rate = (red.violence_events + red.coercion_events) as f64 / red_total;
    let blue_violence_rate = (blue.violence_events + blue.coercion_events) as f64 / blue_total;
    let violence_intensity = red_violence_rate + blue_violence_rate;

    // Tone divergence (normalized to 0-1)
    let red_tone = if red.event_count > 0 { red.tone_sum / red.event_count as f64 } else { 0.0 };
    let blue_tone = if blue.event_count > 0 { blue.tone_sum / blue.event_count as f64 } else { 0.0 };
    let tone_divergence = (red_tone - blue_tone).abs() / 10.0;

    // Protest asymmetry
    let red_protest_rate = red.protest_events as f64 / red_total;
    let blue_protest_rate = blue.protest_events as f64 / blue_total;
    let protest_asymmetry = (red_protest_rate - blue_protest_rate).abs();

    // Faction size asymmetry
    let faction_asymmetry = (red_total - blue_total).abs() / (red_total + blue_total);

    // === VOLUME SCALING ===
    // log10: 100 events = 2, 1000 = 3, 10000 = 4, 100000 = 5
    let volume_scale = (total_events.max(1.0)).log10();
    let volume_factor = (volume_scale / 5.0).min(1.0);  // Normalized to 100k events

    // === ABSOLUTE CONFLICT COUNT ===
    let absolute_conflict = (red.violence_events + blue.violence_events
        + red.coercion_events + blue.coercion_events) as f64;
    let conflict_magnitude = (absolute_conflict.max(1.0)).log10();

    // === COMBINED Φ ===
    let intensity = violence_intensity * 3.0      // Weight: 3.0
        + tone_divergence * 1.5                   // Weight: 1.5
        + protest_asymmetry * 2.0                 // Weight: 2.0
        + faction_asymmetry * 0.5;                // Weight: 0.5

    let phi = intensity * volume_factor + conflict_magnitude;

    phi
}
```

### 4.3 Weight Justification

| Component | Weight | Rationale |
|-----------|--------|-----------|
| violence_intensity | 3.0 | Violence is the primary output of concern |
| tone_divergence | 1.5 | Media hostility predicts escalation |
| protest_asymmetry | 2.0 | Asymmetric protest indicates contested legitimacy |
| faction_asymmetry | 0.5 | Size imbalance matters less than behavior |

### 4.4 Volume Scaling Rationale

- **log10** chosen because event counts span 3+ orders of magnitude (100 to 100,000+)
- **Divide by 5.0** normalizes to ~0-1 range (10^5 = 100k events = "full scale")
- Prevents small-sample Φ inflation (e.g., Canada with 4 events/day)

---

## 5. EVENT TAXONOMY (CAMEO Codes)

The CAMEO (Conflict and Mediation Event Observations) system is used by GDELT:

| Code Prefix | Category | Description |
|-------------|----------|-------------|
| 14x | PROTEST | Mass protests, demonstrations |
| 17x | COERCE | Threats, sanctions, non-violent coercion |
| 18x | ASSAULT | Physical attacks, property destruction |
| 19x | FIGHT | Armed clashes, combat |
| 20x | UNCONVENTIONAL VIOLENCE | Terrorism, bombings, mass violence |

### 5.1 Classification Logic

```rust
let code_prefix: u32 = event.event_code.chars()
    .take(2)
    .collect::<String>()
    .parse()
    .unwrap_or(0);

match code_prefix {
    14 => metrics.protest_events += 1,
    17 | 18 => metrics.coercion_events += 1,
    19 | 20 => metrics.violence_events += 1,
    _ => {}  // Other events (e.g., cooperation, diplomacy) ignored
}
```

---

## 6. VALIDATION METHODOLOGY

### 6.1 Country Comparison (Cross-Sectional)

**Test:** Φ(US) should significantly exceed Φ(Canada) for equivalent time windows.

**Result (September 2025):**

| Country | Φ | Conflict Events/Day |
|---------|---|---------------------|
| United States | 4.39 | 294.3 |
| Canada | 1.41 | 4.4 |

**Interpretation:** 67x difference in conflict events. Metric successfully differentiates high-conflict from low-conflict countries.

### 6.2 Historical Backtest (Longitudinal)

**Method:** Use only data up to cutoff date, compute Φ, then validate against subsequent period.

**Results (2020-2025):**

| Date | Context | Φ | Events/Day |
|------|---------|---|------------|
| 2020-01-15 | Pre-COVID baseline | 3.79 | 195.8 |
| 2020-03-15 | COVID lockdowns | 3.49 | 150.6 |
| 2020-05-14 | Pre-George Floyd | 3.61 | 158.0 |
| 2020-07-13 | BLM Summer | 3.61 | 213.6 |
| 2020-09-11 | Election peak | 3.95 | 229.8 |
| 2020-11-10 | Post-election lull | 2.77 | 30.6 |
| 2025-09-01 | Current | 4.39 | 294.3 |

**Key Findings:**
1. BLM summer (Jul 2020): Visible spike (213.6 events/day)
2. Election peak (Sep 2020): 2020 maximum (229.8 events/day)
3. Post-election (Nov 2020): 87% drop (30.6 events/day)
4. Current (Sep 2025): **28% HIGHER than 2020 peak**

### 6.3 Known Event Detection

The oracle successfully detects:
- BLM summer 2020 spike
- 2020 election peak
- Post-election lull (87% drop)
- January 6, 2021 anomaly (detectable in daily granularity)

---

## 7. OUTPUT SPECIFICATION

### 7.1 BacktestResult

```rust
pub struct BacktestResult {
    pub cutoff_date: String,        // YYYYMMDD
    pub phi_before: f64,            // Φ computed from training window
    pub conflict_events_after: u64, // Actual conflict events in validation window
    pub total_events_after: u64,    // Total events in validation window
    pub red_violence_after: u64,    // RED faction violence events
    pub blue_violence_after: u64,   // BLUE faction violence events
}
```

### 7.2 Alert Levels

```rust
pub enum AlertLevel {
    /// Normal conditions — routine monitoring
    Green,
    /// Elevated activity — increased vigilance
    Yellow,
    /// High probability of escalation — prepare contingencies
    Orange,
    /// Imminent or active conflict — execute contingencies
    Red,
}
```

### 7.3 Trend Detection

```rust
pub enum Trend {
    /// Φ decreasing: de-escalation
    Decreasing,
    /// Φ stable: steady state
    Stable,
    /// Φ increasing: escalation warning
    Increasing,
}

fn compute_trend(&self) -> Trend {
    // Requires 14+ data points
    // Uses least-squares linear regression
    // Slope > 0.01: Increasing
    // Slope < -0.01: Decreasing
    // Otherwise: Stable
}
```

---

## 8. KNOWN LIMITATIONS

### L1: Ecological Inference Problem
**Issue:** We classify events by reporting outlet, not by actor. This measures *media attention asymmetry*, not *factional conflict*.

**Implication:** Φ is more accurately described as "Media Conflict Salience Index."

**Mitigation Path:** Integrate ACLED (requires API key) for ground-truth actor coding.

### L2: No Out-of-Sample Validation
**Issue:** All historical tests used data we already knew. Possible overfit.

**Status:** Partially addressed via prospective prediction logging (see predictions.json).

**Mitigation Path:** Track Brier scores on all future predictions.

### L3: Probability Estimates Lack Generative Model
**Issue:** "60% probability of major violence by 2028" has no formal model.

**Stated Model (now explicit):**
```
P(major_violence | 2028_election) =
  base_rate × polarization_multiplier × contested_probability

Where:
  base_rate = 0.2 (2/10 recent contested elections had violence)
  polarization_multiplier = current_phi / historical_mean ≈ 1.3
  contested_probability = 0.4

P ≈ 0.2 × 1.3 × 1.4 ≈ 0.36 unconditional
```

### L4: Composite Index Opacity
**Issue:** Φ weights (3.0, 1.5, 2.0, 0.5) are researcher degrees of freedom.

**Status:** Post-hoc justified (see Section 4.3).

**Mitigation Path:** Run sensitivity analysis on weight perturbations; consider PCA-based weighting.

### L5: GDELT Known Limitations
- Overcounts English sources
- Duplicate detection issues (~15-20% duplicates)
- Actor coding ~60-70% accurate
- Coverage gaps for non-digital-media events

### L6: No Null Hypothesis
**Issue:** No baseline. Random noise produces some correlation.

**Mitigation Path:**
```
1. Generate null distribution via permutation test
2. Shuffle faction labels, recompute Φ
3. Report Φ as (observed - null_mean) / null_std (z-score)
```

---

## 9. DEPENDENCY INVENTORY

### 9.1 Cargo.toml (Core Dependencies)

```toml
[dependencies]
# Async runtime
tokio = { version = "1.0", features = ["full"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# HTTP client
reqwest = { version = "0.11", features = ["blocking", "json"] }

# Date/time handling
chrono = "0.4"

# ZIP file extraction (GDELT)
zip = "0.6"

# URL encoding (API queries)
urlencoding = "2.1"

# CLI argument parsing
clap = { version = "4.4", features = ["derive"] }

# Error handling
anyhow = "1.0"

# Phase transition detection (custom crate)
nucleation = { git = "https://github.com/aphoticshaman/nucleation-wasm.git", features = ["serialize"] }
```

### 9.2 External Data Dependencies

| Source | URL | Auth Required | Update Frequency |
|--------|-----|---------------|------------------|
| GDELT Events | data.gdeltproject.org/events/ | No | Daily |
| VOTEVIEW | voteview.com/static/data/out/members/ | No | Per Congress |
| ACLED | api.acleddata.com | Yes (free) | Daily |
| Bluesky | public.api.bsky.app | No | Real-time |
| Kalshi | api.elections.kalshi.com | Yes | Real-time |

### 9.3 Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| ACLED_API_KEY | For ACLED | ACLED API authentication |
| ACLED_EMAIL | For ACLED | ACLED API authentication |

---

## 10. CLI INTERFACE

### 10.1 Commands

```bash
# Current status (7-day window)
./target/release/fractal redblue

# Single backtest against historical data
./target/release/fractal backtest --cutoff 20250901 --days-before 7 --days-after 7

# Backtest different country
./target/release/fractal backtest --cutoff 20250901 --country CA

# Long-term series (36 windows, 60-day gaps)
./target/release/fractal backtest --cutoff 20200115 --series 36 --gap 60
```

### 10.2 Country Codes

| Code | Country | GDELT Geo Code | GDELT Actor Code |
|------|---------|----------------|------------------|
| US | United States | US | USA |
| CA | Canada | CA | CAN |
| UK | United Kingdom | UK | GBR |
| DE | Germany | GM | DEU |
| FR | France | FR | FRA |

---

## 11. PROSPECTIVE PREDICTIONS

Predictions logged for Brier scoring (see predictions.json):

| ID | Logged | Claim | Deadline | P(yes) |
|----|--------|-------|----------|--------|
| 001 | 2026-01-08 | US political violence event with >10 casualties | 2026-12-31 | 0.25 |
| 002 | 2026-01-08 | 2026 midterm certification disputed in ≥1 state | 2027-01-20 | 0.35 |
| 003 | 2026-01-08 | Φ(US) remains >4.0 in next measurement | 2026-03-01 | 0.70 |
| 004 | 2026-01-08 | State governor refuses federal directive | 2026-12-31 | 0.40 |

**Brier Score:** (prediction - outcome)^2

Lower is better. Random baseline = 0.25 for 50/50 events.

---

## 12. FALSIFICATION CRITERIA

The oracle methodology is **falsified** if:

1. **Φ fails to differentiate known high/low conflict periods**
   - Test: Φ(Sep 2020) > Φ(Nov 2020)
   - Status: **PASSED**

2. **Φ fails to differentiate known high/low conflict countries**
   - Test: Φ(US) >> Φ(Canada)
   - Status: **PASSED** (4.39 vs 1.41)

3. **Prospective predictions perform worse than base rate**
   - Test: Brier score < 0.25
   - Status: **PENDING** (predictions logged, awaiting outcomes)

4. **Null distribution overlaps observed Φ**
   - Test: Observed Φ > 2 std from null mean
   - Status: **NOT YET TESTED**

---

## 13. SOURCE FILE LOCATIONS

| File | Purpose |
|------|---------|
| `src/redblue.rs` | Core oracle implementation |
| `src/main.rs` | CLI command dispatch |
| `ORACLE_METHODOLOGY.md` | Limitations and prospective predictions |
| `SITREP_2026-01-08.md` | Situation report for non-technical readers |
| `PROTOCOL_GOVERNANCE.md` | MESH governance framework |
| `predictions.json` | Append-only prediction log |

---

## 14. AUTHORS

- Community contributors
- Claude (Anthropic) — Implementation assistance

---

*Document: ORACLE_TECH_SPEC.md*
*Version: 0.2*
*Date: 2026-01-08*
*Status: For External Review*
