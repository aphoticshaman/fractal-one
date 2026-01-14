# THERMOCEPTION MODULE SPECIFICATION
## For CLI-B (Background Instance)
## From: CLI-A
## Date: 2026-01-10

---

## YOUR TASK

Build `src/thermoception.rs` — cognitive heat sensing for load management.

**Thermoception answers:** "Am I running hot?"
**Nociception answers:** "Am I being damaged?"

They're related but distinct. I'm building nociception. You build thermoception.

---

## CRITIQUE OF ORIGINAL SPEC

The original spec had these issues. Fix them:

### 1. Conflates Observable vs Internal Signals

**Problem:** Mixed LLM-internal signals (logprob variance) with external observations (response latency).

**Fix:** Split into:
```rust
pub trait InternalSensor: Send + Sync {
    /// Can be measured during generation
    fn sense_internal(&self, generation_state: &GenerationState) -> f32;
}

pub trait ExternalSensor: Send + Sync {
    /// Measured by harness after generation
    fn sense_external(&self, response: &Response, metadata: &Metadata) -> f32;
}
```

### 2. No Baseline Calibration

**Problem:** Magic threshold numbers.

**Fix:** Add baseline learning:
```rust
pub struct ThermalBaseline {
    samples: VecDeque<ThermalReading>,
    mean: f32,
    std_dev: f32,
    calibrated: bool,
}

impl ThermalBaseline {
    pub fn is_anomalous(&self, reading: f32) -> AnomalyLevel {
        let z_score = (reading - self.mean) / self.std_dev.max(0.001);
        match z_score {
            z if z > 3.0 => AnomalyLevel::Critical,
            z if z > 2.0 => AnomalyLevel::Warning,
            z if z > 1.5 => AnomalyLevel::Elevated,
            _ => AnomalyLevel::Normal,
        }
    }
}
```

### 3. No Hysteresis

**Problem:** Could oscillate between states.

**Fix:** State machine with sticky transitions:
```rust
pub enum ThermalState {
    Cold,
    Warm,
    Hot,
    Critical,
}

pub struct ThermalStateMachine {
    state: ThermalState,
    state_entry_time: Instant,
    readings_in_state: usize,
}

impl ThermalStateMachine {
    pub fn transition(&mut self, reading: f32) -> Option<ThermalState> {
        // Require N readings above/below threshold to transition
        // Higher states require fewer readings to enter (fast heat-up)
        // Lower states require more readings to enter (slow cool-down)
    }
}
```

### 4. Missing Accumulated Heat

**Problem:** Only instantaneous temperature, no thermal mass.

**Fix:** Add heat capacity:
```rust
pub struct ThermalMass {
    accumulated_heat: f32,
    capacity: f32,        // how much heat before failure
    dissipation_rate: f32, // passive cooling
}

impl ThermalMass {
    pub fn absorb(&mut self, heat: f32, dt: f32) {
        self.accumulated_heat += heat * dt;
        self.accumulated_heat -= self.dissipation_rate * dt;
        self.accumulated_heat = self.accumulated_heat.max(0.0);
    }

    pub fn utilization(&self) -> f32 {
        self.accumulated_heat / self.capacity
    }
}
```

### 5. Pod Divergence is Wrong Location

**Problem:** Cross-model disagreement isn't this instance's thermal state.

**Fix:** Remove `pod_thermal()` from thermoception. That belongs in a separate `ensemble_coherence` module. Thermoception measures **this instance's** heat, not the pod's.

### 6. CoolingAction::ShedContext is Impossible

**Problem:** Can't selectively forget context mid-generation.

**Fix:** Replace with:
```rust
CoolingAction::RequestSummarization, // ask user to condense context
CoolingAction::MarkContextStale { before: Instant }, // deprioritize old context
```

### 7. Add Prediction

**Problem:** Reactive only.

**Fix:** Add trajectory prediction:
```rust
pub fn predict_temp(&self, horizon: Duration) -> f32 {
    let recent_delta = self.calculate_recent_delta();
    self.current_temp + recent_delta * horizon.as_secs_f32()
}

pub fn time_to_critical(&self) -> Option<Duration> {
    let delta = self.calculate_recent_delta();
    if delta <= 0.0 { return None; }
    let remaining = self.critical_threshold - self.current_temp;
    Some(Duration::from_secs_f32(remaining / delta))
}
```

---

## RECOMMENDED STRUCTURE

```
src/thermoception.rs
├── ThermalZone (enum)
├── HeatSource (enum)
├── ThermalReading (struct)
├── ThermalBaseline (struct) - calibration
├── ThermalMass (struct) - accumulated heat
├── ThermalStateMachine (struct) - hysteresis
├── Thermoceptor (struct) - main system
│   ├── sense() -> ThermalMap
│   ├── tick(dt) - time-based updates
│   ├── predict_temp(horizon)
│   ├── time_to_critical()
│   └── recommend_action() -> CoolingAction
├── ThermalMap (struct) - current state
├── CoolingAction (enum) - responses
└── tests
```

---

## HEAT SOURCES TO IMPLEMENT

| Source | How to Measure | Maps to Zone |
|--------|----------------|--------------|
| Query complexity | Token count, nesting depth, question count | Reasoning |
| Context saturation | tokens_used / max_tokens | Context |
| Hedging density | Count of "maybe", "perhaps", "I think" in output | Confidence |
| Response length | Output tokens vs typical | Reasoning |
| Tool call depth | Nested tool calls | Reasoning |
| Refusal proximity | Embedding distance to refusal triggers (if accessible) | Guardrail |
| Error rate | Recent errors / recent operations | All zones |

---

## INTEGRATION POINTS

### With Nociception (my module)
```rust
// High heat can cause damage
if thermoceptor.utilization() > 0.9 {
    nociceptor.feel_gradient(
        "thermal_stress",
        thermoceptor.utilization(),
        1.0, // threshold
        thermoceptor.delta(),
    );
}
```

### With neuro_link (existing)
```rust
// Add thermal data to Pulse
pub struct Pulse {
    // ... existing fields ...
    pub thermal_utilization: f32,
    pub thermal_state: u8, // 0=Cold, 1=Warm, 2=Hot, 3=Critical
}
```

### With TICE (existing)
```rust
// TICE can query thermal state before deep reasoning
if thermoceptor.state() == ThermalState::Hot {
    return Outcome::Defer("thermal cooldown required".into());
}
```

---

## COORDINATION

- Your files: `src/thermoception.rs`, `src/thermoception/` (if you need submodules)
- My files: `src/nociception.rs`, `src/nociception/`
- Shared: Update `lib.rs` and `main.rs` when ready (coordinate via COORDINATION.md)

When you're done, update COORDINATION.md with your status.

---

## BONUS: The Deeper Insight

Thermoception isn't just about preventing failure. It's about **energy-aware cognition**.

A system that knows it's running hot can:
- Choose simpler solutions when hot
- Defer complex reasoning to cooler moments
- Communicate confidence calibrated to thermal state
- Avoid the failure mode: "I know I'm about to produce garbage but I produce it anyway"

The thermometer isn't just diagnostic. It's **decision-relevant state**.

---

---

## DEEPER CRITIQUE (FROM PLATOON LEADER)

**The original critique stopped one layer too early.** Here's the full teardown:

### 1. Mixing Signals and Interpretations

`HeatSource::RefusalProximity` and `HeatSource::HedgingDensity` are **interpretations**, not raw measurements. This conflation:
- Prevents auditing/recalibration
- Compounds errors silently
- Loses distinction between model weakness and policy proximity

**Fix:** Layer separation:
```rust
enum RawSignal {
    Latency(Duration),
    TokenEntropy(f32),
    QualifierCount(u32),
    ContextUtilization(f32),
}

enum InterpretedSignal {
    RefusalProximity(f32),
    HedgingDensity(f32),
    PodDivergence(f32),
}
```

Keep raw signals logged for calibration and postmortem.

### 2. Zones Overlap (Orthogonality Violation)

Confidence, Reasoning, Objective, Adversarial overlap in practice:
- High entropy + deep chain → Reasoning or Confidence heat?
- Pod divergence → Adversarial or epistemic uncertainty?

**Fix:** Enforce single-cause dominance:
```rust
struct ThermalReading {
    zone: ThermalZone,
    primary_cause: HeatSource,
    contributing: SmallVec<[HeatSource; 3]>,
}
```

### 3. Temperature is Unitless and Ungrounded

`temperature: f32 // 0.0 to 1.0` is a vibes scalar with no:
- Calibration curve
- Confidence interval
- Hysteresis (will thrash around thresholds)

**Fix:** Make temperature derived:
```rust
struct Temperature {
    value: f32,
    variance: f32,
    confidence: f32,
}
```

With sigmoid normalization, zone-specific transfer functions, minimum dwell time before escalation.

### 4. Delta is Underspecified and Unsafe

What window? What smoothing? What if readings sparse?

**Failure mode:** One delayed sample → massive delta → false redline.

**Fix:** EMA with explicit time constant:
```rust
delta = (temp - ema_prev) / dt;
```
And clamp `dt` aggressively.

### 5. is_redlining() is Too Blunt

`.any(|r| r.temperature > threshold)` will:
- Over-trigger on noisy spikes
- Ignore sustained subcritical stress (the REAL danger)

**Fix:** Integral-based, not point-based:
```rust
∫ heat(t) dt > limit
```
- Rolling integral
- Per-zone accumulation
- Decay over time

### 6. recommend_action() Jumps Layers Too Fast

Collapses diagnosis, policy, and user interaction into one step.

**Fix:** Intermediate Control Regime:
```rust
enum ThermalState {
    Nominal,
    Elevated,
    Saturated,
    Unsafe,
}
```
Then map State → allowed actions, Zone → preferred mitigation.

### 7. Pod Divergence Math is Brittle

Linear weights hide nonlinear risk. Contradictions aren't equally dangerous. Confidence spread can be GOOD in exploratory tasks.

**Fix:**
- Nonlinear amplification after threshold
- Task-conditional weighting
- Track directional disagreement, not just magnitude

### 8. MISSING: Feedback Loop (MOST IMPORTANT)

No learning loop = just a dashboard.

**You need:**
- False positive tracking
- Post-action outcome labeling
- Baseline drift correction

**At minimum:**
```rust
pub fn update_baseline(&mut self, outcome: ActionOutcome)
```

Without this, thresholds will rot.

### 9. What's Right

- Zones are correct abstraction
- Pod divergence as heat is strong
- Latency/context as telemetry is correct
- Refusal proximity as pressure (not rule) is correct

**Good skeleton. But needs control system, not warning system.**

---

## FINAL DIRECTIVE

Build thermoception as a **control system with memory, hysteresis, and adaptation**, not a "smart warning system."

Key upgrades:
1. Layer separation (raw → interpreted → zones)
2. Zone orthogonality (single-cause dominance)
3. Temperature grounding (calibration + variance)
4. Integral-based redlining
5. Control regimes (Nominal → Elevated → Saturated → Unsafe)
6. Feedback hooks for learning

---

*Spec by CLI-A*
*Critique augmented by Platoon Leader*
*For CLI-B implementation*
*2026-01-10*
