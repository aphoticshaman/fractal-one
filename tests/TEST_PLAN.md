# Fractal One: Ablation & Integration Test Plan

## Architecture Under Test

```
┌─────────────────────────────────────────────────────────────┐
│                      FRACTAL CRATE                          │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   heart     │   cortex    │ voice_bridge│  command_module  │
│  (timing)   │ (monitor)   │   (LLM)     │    (human)       │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                     neuro_link (IPC)                        │
├─────────────────────────────────────────────────────────────┤
│                     crux (falsification)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## ABLATION TESTS

Goal: Identify what each component contributes by removing it.

### A1: neuro_link Ablation
**Remove:** Shared memory IPC
**Expect:** All inter-component communication fails
**Test:**
```rust
#[test]
fn ablation_no_neuro_link() {
    // Start cortex WITHOUT heart creating SHM
    let result = std::panic::catch_unwind(|| {
        Synapse::connect(false) // Should fail - no SHM exists
    });
    assert!(result.is_err(), "Cortex should fail without neuro_link");
}
```
**Confirms:** neuro_link is foundational IPC layer

### A2: heart Ablation
**Remove:** Timing core
**Expect:** No pulses generated, cortex starves
**Test:**
```rust
#[test]
fn ablation_no_heart() {
    let synapse = Synapse::connect(true); // Create SHM but don't run heart
    let start = Instant::now();

    while start.elapsed() < Duration::from_secs(2) {
        if synapse.peek_latest().is_some() {
            panic!("Received pulse without heart running");
        }
        thread::sleep(Duration::from_millis(100));
    }
    // Pass: no pulses without heart
}
```
**Confirms:** heart is sole pulse generator

### A3: cortex Ablation
**Remove:** Health monitoring
**Expect:** System runs but blind to own state
**Test:**
```rust
#[test]
fn ablation_no_cortex() {
    // Run heart only, verify pulses still flow
    let handle = spawn_heart();
    thread::sleep(Duration::from_secs(1));

    let synapse = Synapse::connect(false);
    let pulse = synapse.peek_latest();
    assert!(pulse.is_some(), "Heart should pulse without cortex");

    // System runs but no monitoring output
    handle.abort();
}
```
**Confirms:** cortex is observer, not required for core function

### A4: voice_bridge Ablation
**Remove:** LLM integration
**Expect:** System runs, no external intelligence
**Test:**
```rust
#[test]
fn ablation_no_voice_bridge() {
    // Full daemon without voice_bridge
    // Verify heart+cortex function normally
    // No API calls should occur
    let api_calls = AtomicU32::new(0);
    // ... run system ...
    assert_eq!(api_calls.load(Ordering::Relaxed), 0);
}
```
**Confirms:** voice_bridge is optional intelligence layer

### A5: crux Ablation (Critical)
**Remove:** Falsification engine
**Expect:** All other cognitive operations become untestable
**Test:**
```rust
#[test]
fn ablation_no_crux() {
    // Attempt evolution without fitness function grounded in crux
    let mut evolver = IdeaEvolver::new_without_crux();
    let result = evolver.evolve(|_| 0.5, 10); // Random fitness

    // Ideas "evolve" but have no connection to reality
    assert!(result.predictions.is_empty() || !result.tested);
}
```
**Confirms:** Without crux, cognitive operations are ungrounded

---

## INTEGRATION TESTS

Goal: Verify components work correctly together.

### I1: heart → neuro_link → cortex Pipeline
**Test:** Full pulse flow from generation to consumption
```rust
#[test]
fn integration_pulse_pipeline() {
    // Start heart
    let heart_handle = spawn_heart();
    thread::sleep(Duration::from_millis(200));

    // Start cortex
    let (tx, rx) = channel();
    let cortex_handle = spawn_cortex_with_callback(move |pulse| {
        tx.send(pulse.id).unwrap();
    });

    // Verify pulses flow
    let received: Vec<u64> = rx.try_iter().take(10).collect();
    assert!(received.len() >= 5, "Should receive multiple pulses");

    // Verify ordering
    for window in received.windows(2) {
        assert!(window[1] > window[0], "Pulses should be ordered");
    }

    cleanup(heart_handle, cortex_handle);
}
```

### I2: Multi-Consumer IPC
**Test:** Multiple consumers read same pulse stream
```rust
#[test]
fn integration_multi_consumer() {
    spawn_heart();
    thread::sleep(Duration::from_millis(100));

    // Three consumers with different IDs
    let mut consumer_0 = Synapse::connect_as(false, 0); // cortex
    let mut consumer_1 = Synapse::connect_as(false, 1); // voice
    let mut consumer_2 = Synapse::connect_as(false, 2); // gpu

    thread::sleep(Duration::from_millis(500));

    // Each should have independent tail pointer
    let p0: Vec<u64> = drain_pulses(&mut consumer_0).iter().map(|p| p.id).collect();
    let p1: Vec<u64> = drain_pulses(&mut consumer_1).iter().map(|p| p.id).collect();
    let p2: Vec<u64> = drain_pulses(&mut consumer_2).iter().map(|p| p.id).collect();

    // All should receive same pulses
    assert_eq!(p0, p1);
    assert_eq!(p1, p2);
}
```

### I3: Kill Signal Propagation
**Test:** Kill signal stops all components
```rust
#[test]
fn integration_kill_signal() {
    let heart = spawn_heart();
    let cortex = spawn_cortex();
    let voice = spawn_voice_bridge();

    thread::sleep(Duration::from_millis(500));

    // Send kill signal
    Synapse::connect(false).send_kill_signal();

    // All should terminate within 2 seconds
    let timeout = Duration::from_secs(2);
    assert!(heart.join_timeout(timeout).is_ok(), "Heart should stop");
    assert!(cortex.join_timeout(timeout).is_ok(), "Cortex should stop");
    assert!(voice.join_timeout(timeout).is_ok(), "Voice should stop");
}
```

### I4: crux → Test → Environment Loop
**Test:** Crux generates testable claims and executes them
```rust
#[test]
fn integration_crux_test_loop() {
    let engine = CruxEngine::new(test_api_key());

    // Known-true claim
    let mut crux = block_on(engine.resolve(
        "The file Cargo.toml exists in the project root"
    ));

    assert!(!crux.falsifier.is_empty(), "Should generate falsifier");
    assert!(matches!(crux.test_type, TestType::FileCheck(_)));

    let result = block_on(engine.test(&mut crux));
    assert!(result, "Known-true claim should pass");
    assert!(crux.result.is_some());

    // Known-false claim
    let mut crux_false = block_on(engine.resolve(
        "The file NONEXISTENT_FILE_12345.xyz exists"
    ));

    let result_false = block_on(engine.test(&mut crux_false));
    assert!(!result_false, "Known-false claim should fail");
}
```

### I5: crux → Evolution Fitness Integration
**Test:** Evolution uses crux results as fitness
```rust
#[test]
fn integration_crux_evolution() {
    let crux_engine = CruxEngine::new(test_api_key());
    let mut evolver = IdeaEvolver::new();

    // Seed with testable ideas
    evolver.seed(vec![
        "Sorting algorithms have O(n log n) average case",
        "Hash tables have O(1) average lookup",
        "Linked lists have O(1) insertion at head",
    ]);

    // Fitness = crux test pass rate
    let fitness_fn = |idea: &Idea| -> f64 {
        let crux = block_on(crux_engine.resolve(&idea.content));
        match block_on(crux_engine.test(&mut crux)) {
            true => 1.0,
            false => 0.0,
        }
    };

    let evolved = block_on(evolver.evolve(fitness_fn, 3));

    // Evolved ideas should be testable and true
    let final_crux = block_on(crux_engine.resolve(&evolved.content));
    assert!(final_crux.result.map(|r| r.passed).unwrap_or(false));
}
```

### I6: Daemon Mode Integration
**Test:** All components run together in daemon mode
```rust
#[test]
fn integration_daemon_mode() {
    let daemon = spawn_daemon();
    thread::sleep(Duration::from_secs(3));

    // Verify heart is pulsing
    let synapse = Synapse::connect(false);
    assert!(synapse.peek_latest().is_some(), "Heart should be pulsing");

    // Verify SHM file exists
    assert!(Path::new("neuro_link.shm").exists());

    // Verify no orphan processes
    let fractal_procs = count_processes_matching("fractal");
    assert_eq!(fractal_procs, 1, "Should be single daemon process");

    // Clean shutdown
    synapse.send_kill_signal();
    thread::sleep(Duration::from_secs(1));

    let remaining = count_processes_matching("fractal");
    assert_eq!(remaining, 0, "All processes should terminate");
}
```

---

## STRESS TESTS

### S1: Timing Precision Under Load
```rust
#[test]
fn stress_timing_precision() {
    spawn_heart();
    spawn_entropy_storm(); // Max CPU load

    thread::sleep(Duration::from_secs(5));

    let synapse = Synapse::connect(false);
    let pulses: Vec<Pulse> = drain_pulses(&mut synapse);

    // Jitter should remain <1ms even under load
    let max_jitter = pulses.iter().map(|p| p.jitter_ms).fold(0.0, f64::max);
    assert!(max_jitter < 1.0, "Jitter {} exceeded 1ms under load", max_jitter);
}
```

### S2: Memory Stability
```rust
#[test]
fn stress_memory_stability() {
    let initial_mem = get_process_memory();

    spawn_daemon();

    for _ in 0..60 {
        thread::sleep(Duration::from_secs(1));
        let current_mem = get_process_memory();
        let growth = current_mem - initial_mem;

        // Memory should not grow unbounded
        assert!(growth < 100_000_000, "Memory grew by {}MB", growth / 1_000_000);
    }
}
```

### S3: Ring Buffer Overflow
```rust
#[test]
fn stress_ring_buffer_overflow() {
    spawn_heart();

    // Don't consume for long enough to wrap buffer
    thread::sleep(Duration::from_secs(10));

    let mut synapse = Synapse::connect(false);
    let pulses = drain_pulses(&mut synapse);

    // Should get most recent BUFFER_SIZE pulses, not crash
    assert!(pulses.len() <= 1024, "Should not exceed buffer size");

    // IDs should be contiguous (no corruption)
    for window in pulses.windows(2) {
        let gap = window[1].id - window[0].id;
        assert!(gap == 1 || gap == 0, "Non-contiguous pulse IDs: gap={}", gap);
    }
}
```

---

## TEST EXECUTION ORDER

```
Phase 1: Ablation (verify component boundaries)
  A1 → A2 → A3 → A4 → A5

Phase 2: Integration (verify component cooperation)
  I1 → I2 → I3 → I4 → I5 → I6

Phase 3: Stress (verify robustness)
  S1 → S2 → S3
```

## PASS CRITERIA

| Test Category | Pass Threshold |
|--------------|----------------|
| Ablation     | 100% (all must pass) |
| Integration  | 100% (all must pass) |
| Stress       | 90% (S1 may degrade on slow hardware) |

## CRUX META-TEST

The test plan itself should be crux-testable:

```rust
let test_plan_crux = engine.resolve(
    "The test plan covers all critical component interactions"
).await;

// Falsifier: IF any component pair lacks integration test THEN plan is incomplete
```

Current coverage matrix:

|              | neuro_link | heart | cortex | voice | crux |
|--------------|------------|-------|--------|-------|------|
| neuro_link   | -          | I1    | I1     | I2    | -    |
| heart        | I1         | -     | I1     | I2    | -    |
| cortex       | I1         | I1    | -      | -     | -    |
| voice        | I2         | I2    | -      | -     | I4   |
| crux         | -          | -     | -      | I4    | I5   |

**Gap identified:** crux ↔ neuro_link integration (crux testing timing claims)
