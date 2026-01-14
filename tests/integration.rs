//! Integration Tests - Do components work together?
//!
//! These tests require shared memory and must run serially.
//! Run with: cargo test --test integration -- --test-threads=1 --ignored

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use fractal::neuro_link::{Pulse, Synapse};

/// I1: Pulse flow from producer to consumer
#[test]
#[ignore] // Requires SHM, run with --ignored
fn integration_pulse_flow() {
    // Producer
    let mut producer = Synapse::connect(true);

    // Consumer
    let mut consumer = Synapse::connect(false);

    // Fire pulses
    for i in 1..=5 {
        producer.fire(Pulse {
            id: i,
            telemetry_sequence: i,
            jitter_ms: 0.001 * i as f64,
            cpu_load_percent: 10.0 * i as f64,
            current_interval_ms: 80,
            bad_actor_id: 0,
            entropy_damping: 0.0,
            payload: [0.0; 32],
            scheduler_override: 0,
        });
    }

    // Read back
    let mut received = Vec::new();
    while let Some(pulse) = consumer.sense() {
        received.push(pulse);
    }

    assert_eq!(received.len(), 5, "Should receive all 5 pulses");

    // Verify order
    for (i, pulse) in received.iter().enumerate() {
        assert_eq!(pulse.id, (i + 1) as u64);
    }

    let _ = std::fs::remove_file("neuro_link.shm");
}

/// I2: Multiple consumers with independent tails
#[test]
#[ignore] // Requires SHM, run with --ignored
fn integration_multi_consumer() {
    let mut producer = Synapse::connect(true);

    // Three independent consumers
    let mut consumer_0 = Synapse::connect_as(false, 0);
    let mut consumer_1 = Synapse::connect_as(false, 1);
    let mut consumer_2 = Synapse::connect_as(false, 2);

    // Fire pulses
    for i in 1..=10 {
        producer.fire(Pulse {
            id: i,
            telemetry_sequence: i,
            jitter_ms: 0.0,
            cpu_load_percent: 0.0,
            current_interval_ms: 80,
            bad_actor_id: 0,
            entropy_damping: 0.0,
            payload: [0.0; 32],
            scheduler_override: 0,
        });
    }

    // Each consumer reads independently
    let read_0: Vec<u64> = std::iter::from_fn(|| consumer_0.sense().map(|p| p.id)).collect();
    let read_1: Vec<u64> = std::iter::from_fn(|| consumer_1.sense().map(|p| p.id)).collect();
    let read_2: Vec<u64> = std::iter::from_fn(|| consumer_2.sense().map(|p| p.id)).collect();

    // All should have received same data
    assert_eq!(read_0.len(), 10);
    assert_eq!(read_0, read_1);
    assert_eq!(read_1, read_2);

    // Now fire more - each already consumed, should only see new
    for i in 11..=15 {
        producer.fire(Pulse {
            id: i,
            telemetry_sequence: i,
            jitter_ms: 0.0,
            cpu_load_percent: 0.0,
            current_interval_ms: 80,
            bad_actor_id: 0,
            entropy_damping: 0.0,
            payload: [0.0; 32],
            scheduler_override: 0,
        });
    }

    let read_0_new: Vec<u64> = std::iter::from_fn(|| consumer_0.sense().map(|p| p.id)).collect();
    assert_eq!(read_0_new, vec![11, 12, 13, 14, 15]);

    let _ = std::fs::remove_file("neuro_link.shm");
}

/// I3: Peek doesn't consume
#[test]
#[ignore] // Requires SHM, run with --ignored
fn integration_peek_vs_sense() {
    let mut producer = Synapse::connect(true);
    let mut consumer = Synapse::connect(false);

    producer.fire(Pulse {
        id: 42,
        telemetry_sequence: 42,
        jitter_ms: 0.0,
        cpu_load_percent: 0.0,
        current_interval_ms: 80,
        bad_actor_id: 0,
        entropy_damping: 0.0,
        payload: [0.0; 32],
        scheduler_override: 0,
    });

    // Peek multiple times - should always return same
    let peek1 = consumer.peek_latest().unwrap().id;
    let peek2 = consumer.peek_latest().unwrap().id;
    let peek3 = consumer.peek_latest().unwrap().id;

    assert_eq!(peek1, 42);
    assert_eq!(peek2, 42);
    assert_eq!(peek3, 42);

    // Sense consumes
    let sensed = consumer.sense().unwrap().id;
    assert_eq!(sensed, 42);

    // Now sense returns None (consumed)
    assert!(consumer.sense().is_none());

    // But peek still works (returns last written)
    assert_eq!(consumer.peek_latest().unwrap().id, 42);

    let _ = std::fs::remove_file("neuro_link.shm");
}

/// I4: Ring buffer wraps correctly
#[test]
#[ignore] // Requires SHM, run with --ignored
fn integration_ring_buffer_wrap() {
    let mut producer = Synapse::connect(true);
    let consumer = Synapse::connect(false);

    // Fire more than buffer size (1024)
    for i in 1..=1500 {
        producer.fire(Pulse {
            id: i,
            telemetry_sequence: i,
            jitter_ms: 0.0,
            cpu_load_percent: 0.0,
            current_interval_ms: 80,
            bad_actor_id: 0,
            entropy_damping: 0.0,
            payload: [0.0; 32],
            scheduler_override: 0,
        });
    }

    // Should be able to peek latest
    let latest = consumer.peek_latest().unwrap();
    assert_eq!(latest.id, 1500, "Should see most recent pulse");

    let _ = std::fs::remove_file("neuro_link.shm");
}

/// I5: Payload data integrity
#[test]
#[ignore] // Requires SHM, run with --ignored
fn integration_payload_integrity() {
    let mut producer = Synapse::connect(true);
    let mut consumer = Synapse::connect(false);

    // Create distinctive payload
    let mut payload = [0.0f32; 32];
    for i in 0..32 {
        payload[i] = (i as f32) * 3.14159;
    }

    producer.fire(Pulse {
        id: 1,
        telemetry_sequence: 1,
        jitter_ms: 1.23456,
        cpu_load_percent: 78.9,
        current_interval_ms: 80,
        bad_actor_id: 12345,
        entropy_damping: 0.5678,
        payload,
        scheduler_override: 999,
    });

    let received = consumer.sense().unwrap();

    assert_eq!(received.id, 1);
    assert!((received.jitter_ms - 1.23456).abs() < 0.0001);
    assert!((received.cpu_load_percent - 78.9).abs() < 0.1);
    assert_eq!(received.bad_actor_id, 12345);
    assert!((received.entropy_damping - 0.5678).abs() < 0.0001);
    assert_eq!(received.scheduler_override, 999);

    // Check payload
    for i in 0..32 {
        let expected = (i as f32) * 3.14159;
        assert!(
            (received.payload[i] - expected).abs() < 0.0001,
            "Payload[{}] mismatch: {} vs {}",
            i,
            received.payload[i],
            expected
        );
    }

    let _ = std::fs::remove_file("neuro_link.shm");
}

/// I6: Concurrent access safety
#[test]
#[ignore] // Requires SHM, run with --ignored
fn integration_concurrent_access() {
    let _ = std::fs::remove_file("neuro_link.shm");

    let pulse_count = Arc::new(AtomicU64::new(0));

    // Producer thread
    let producer_count = pulse_count.clone();
    let producer_handle = thread::spawn(move || {
        let mut synapse = Synapse::connect(true);
        for i in 1..=100 {
            synapse.fire(Pulse {
                id: i,
                telemetry_sequence: i,
                jitter_ms: 0.0,
                cpu_load_percent: 0.0,
                current_interval_ms: 80,
                bad_actor_id: 0,
                entropy_damping: 0.0,
                payload: [0.0; 32],
                scheduler_override: 0,
            });
            producer_count.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_micros(100));
        }
    });

    // Give producer head start
    thread::sleep(Duration::from_millis(5));

    // Consumer thread
    let consumer_handle = thread::spawn(move || {
        let mut synapse = Synapse::connect(false);
        let mut received = 0u64;
        let start = Instant::now();

        while start.elapsed() < Duration::from_secs(2) {
            if let Some(_) = synapse.sense() {
                received += 1;
            }
            thread::sleep(Duration::from_micros(50));
        }
        received
    });

    producer_handle.join().unwrap();
    let received = consumer_handle.join().unwrap();

    assert!(
        received >= 90,
        "Should receive most pulses, got {}",
        received
    );

    let _ = std::fs::remove_file("neuro_link.shm");
}
