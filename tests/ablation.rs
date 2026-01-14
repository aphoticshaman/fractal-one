//! Ablation Tests - What breaks when you remove X?
//!
//! These tests require a clean SHM environment and must run serially.
//! Run with: cargo test --test ablation -- --test-threads=1 --ignored

use std::panic;
use std::thread;
use std::time::Duration;

use fractal::neuro_link::{Pulse, Synapse};

const SHM_PATH: &str = "neuro_link.shm";

/// Helper: ensure clean state before each test
fn setup() {
    let _ = std::fs::remove_file(SHM_PATH);
    thread::sleep(Duration::from_millis(10));
}

/// Helper: cleanup after test
fn teardown() {
    let _ = std::fs::remove_file(SHM_PATH);
}

/// A1: connect(false) requires existing SHM file
#[test]
fn ablation_a1_shm_must_exist_for_consumer() {
    setup();

    let result = panic::catch_unwind(|| {
        Synapse::connect(false) // create=false means don't create
    });

    assert!(
        result.is_err(),
        "connect(false) should panic when SHM missing"
    );
    teardown();
}

/// A2: Fresh SHM with create=true zeros the buffer
#[test]
#[ignore] // Requires clean SHM state, run with --ignored
fn ablation_a2_fresh_shm_is_zeroed() {
    setup();

    // Create fresh SHM
    let _producer = Synapse::connect(true);
    let consumer = Synapse::connect(false);

    // peek_latest on fresh buffer: head=0, so index wraps to BUFFER_SIZE-1
    // That slot contains zeroed data (id=0)
    let peek = consumer.peek_latest();

    if let Some(p) = peek {
        // Zeroed pulse has id=0
        assert_eq!(p.id, 0, "Fresh SHM should have zeroed pulse data");
    }
    // None is also acceptable (empty buffer interpretation)

    teardown();
}

/// A3: Pulses persist in SHM without memory_sink
#[test]
#[ignore] // Requires clean SHM state, run with --ignored
fn ablation_a3_memory_sink_optional() {
    setup();

    let mut synapse = Synapse::connect(true);

    // Fire pulses
    for i in 1..=10 {
        synapse.fire(Pulse {
            id: i,
            telemetry_sequence: i,
            jitter_ms: 0.001,
            cpu_load_percent: 25.0,
            current_interval_ms: 80,
            bad_actor_id: 0,
            entropy_damping: 0.0,
            payload: [0.0; 32],
            scheduler_override: 0,
        });
    }

    // Pulses exist in SHM
    let consumer = Synapse::connect(false);
    let latest = consumer.peek_latest().expect("Should have pulses");
    assert_eq!(latest.id, 10, "Latest pulse should be id=10");

    // Journal file should NOT grow (memory_sink not running)
    let journal_before = std::fs::metadata("archon_journal.jsonl")
        .map(|m| m.len())
        .unwrap_or(0);

    thread::sleep(Duration::from_millis(100));

    let journal_after = std::fs::metadata("archon_journal.jsonl")
        .map(|m| m.len())
        .unwrap_or(0);

    assert_eq!(
        journal_before, journal_after,
        "Journal unchanged without memory_sink"
    );

    teardown();
}

/// A4: Kill signal propagates through SHM
#[test]
#[ignore] // Requires clean SHM state, run with --ignored
fn ablation_a4_kill_signal_works() {
    setup();

    let synapse = Synapse::connect(true);

    assert!(!synapse.check_kill_signal(), "Fresh SHM has no kill signal");

    synapse.send_kill_signal();

    assert!(synapse.check_kill_signal(), "Kill signal should be set");

    // Other consumers see it too
    let other = Synapse::connect(false);
    assert!(
        other.check_kill_signal(),
        "Kill signal visible to all consumers"
    );

    teardown();
}

/// A5: Target interval is controllable
#[test]
#[ignore] // Requires clean SHM state, run with --ignored
fn ablation_a5_interval_control() {
    setup();

    let synapse = Synapse::connect(true);

    assert_eq!(synapse.get_target_interval(), 80, "Default is 80ms");

    synapse.set_target_interval(100);
    assert_eq!(synapse.get_target_interval(), 100);

    // Other consumers see the change
    let other = Synapse::connect(false);
    assert_eq!(
        other.get_target_interval(),
        100,
        "Interval visible to consumers"
    );

    synapse.set_target_interval(50);
    assert_eq!(other.get_target_interval(), 50, "Updates propagate");

    teardown();
}

/// A6: Without any component, SHM is just inert file
#[test]
#[ignore] // Requires clean SHM state, run with --ignored
fn ablation_a6_shm_is_passive() {
    setup();

    // Create SHM
    {
        let _syn = Synapse::connect(true);
    }
    // Synapse dropped, but file persists

    assert!(std::path::Path::new(SHM_PATH).exists(), "SHM file persists");

    // Can reconnect
    let syn = Synapse::connect(false);
    assert_eq!(
        syn.get_target_interval(),
        80,
        "State persists across connections"
    );

    teardown();
}
