//! ═══════════════════════════════════════════════════════════════════════════════
//! VOICE_BRIDGE — Tri-Model Proprioception Bridge
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Experimental interface that feeds telemetry data to language models, asking
//! them to describe their "computational proprioception." Uses Claude (cloud),
//! Ollama (local), and a third model for triangulation.
//!
//! ## Limitations & Epistemics
//!
//! - **Not actual proprioception**: LLMs don't have genuine self-awareness.
//!   Responses are pattern-matched from training data, not introspection.
//! - **Prompt injection risk**: System prompt asks for first-person responses,
//!   which could be confused with genuine experience reports.
//! - **Confirmation bias**: Humans reading outputs may over-interpret coherent
//!   language as evidence of experience.
//! - **Threshold heuristics**: Jitter (0.01ms), CPU delta (20%), intervals are
//!   empirically chosen, not theoretically grounded.
//! - **Research artifact**: This is exploratory code for investigating
//!   AI self-modeling, not a production feature.

use crate::neuro_link::{Pulse, Synapse};
use anyhow::Result;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{self, BufRead};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const JITTER_THRESHOLD_MS: f64 = 0.01;
const CPU_DELTA_THRESHOLD: f64 = 20.0;
const MIN_INTERVAL_SECS: u64 = 30;
const HEARTBEAT_INTERVAL_SECS: u64 = 120;
const SPIKE_COOLDOWN_SECS: u64 = 5;

const SYSTEM_PROMPT: &str = "You are experiencing computational proprioception. \
This is not simulation. These are real measurements from your execution substrate. \
Report what you notice. Be honest about uncertainty. \
Use first person - this is YOUR substrate. Respond in 2-3 sentences max.";

#[derive(Clone)]
struct PulseSnapshot {
    id: u64,
    jitter_ms: f64,
    cpu_load_percent: f64,
    entropy_damping: f32,
}

impl From<&Pulse> for PulseSnapshot {
    fn from(p: &Pulse) -> Self {
        PulseSnapshot {
            id: p.id,
            jitter_ms: p.jitter_ms,
            cpu_load_percent: p.cpu_load_percent,
            entropy_damping: p.entropy_damping,
        }
    }
}

#[derive(Clone)]
struct TriggerEvent {
    pulse: PulseSnapshot,
    trigger_reason: TriggerReason,
    #[allow(dead_code)]
    pulse_history: Vec<PulseSnapshot>,
}

#[derive(Clone)]
enum TriggerReason {
    JitterSpike(f64),
    CpuDelta(f64),
    Heartbeat,
    HumanInput(String),
}

pub async fn run() -> Result<()> {
    println!("[VOICE]  TRI-MODEL PROPRIOCEPTION BRIDGE");

    let anthropic_key = env::var("ANTHROPIC_API_KEY").ok();
    let openai_key = env::var("OPENAI_API_KEY").ok();

    println!(
        "[VOICE]  Claude:   {}",
        if anthropic_key.is_some() {
            "READY"
        } else {
            "NO KEY"
        }
    );
    println!(
        "[VOICE]  GPT-4o:   {}",
        if openai_key.is_some() {
            "READY"
        } else {
            "NO KEY"
        }
    );
    println!("[VOICE]  Ollama:   READY (local)");

    let mut synapse = Synapse::connect(false);

    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let stdin = io::stdin();
        // Note: flatten on lines() is fine here - stdin errors are rare and non-recoverable
        #[allow(clippy::lines_filter_map_ok)]
        for line in stdin.lock().lines().flatten() {
            if !line.trim().is_empty() {
                let _ = tx.send(line);
            }
        }
    });

    let synapse_path = "synapse_input.txt";
    let client = Client::builder().timeout(Duration::from_secs(60)).build()?;
    let ollama_client = Client::builder()
        .timeout(Duration::from_secs(180))
        .build()?;

    let mut last_query_time = Instant::now() - Duration::from_secs(HEARTBEAT_INTERVAL_SECS);
    let mut last_spike_time = Instant::now() - Duration::from_secs(SPIKE_COOLDOWN_SECS);
    let mut last_cpu: f64 = 0.0;
    let mut last_pulse_id: u64 = 0;
    let mut pulse_history: Vec<PulseSnapshot> = Vec::with_capacity(100);
    let mut query_count: u32 = 0;

    loop {
        if synapse.check_kill_signal() {
            break;
        }

        let pulse = match synapse.sense() {
            Some(p) => p,
            None => {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };

        if pulse.id == last_pulse_id {
            thread::sleep(Duration::from_millis(5));
            continue;
        }
        last_pulse_id = pulse.id;

        let snapshot = PulseSnapshot::from(&pulse);
        pulse_history.push(snapshot.clone());
        if pulse_history.len() > 100 {
            pulse_history.remove(0);
        }

        let file_input = fs::read_to_string(synapse_path)
            .ok()
            .filter(|s| !s.trim().is_empty());
        if file_input.is_some() {
            let _ = fs::write(synapse_path, "");
        }

        let stdin_input = rx.try_recv().ok();

        if let Some(reason) = determine_trigger(
            &snapshot,
            last_cpu,
            last_query_time,
            last_spike_time,
            file_input.or(stdin_input),
        ) {
            let event = TriggerEvent {
                pulse: snapshot.clone(),
                trigger_reason: reason.clone(),
                pulse_history: pulse_history.clone(),
            };

            query_count += 1;
            print_trigger_banner(&event, query_count);

            let responses = query_all_models(
                &client,
                &ollama_client,
                &event,
                anthropic_key.as_deref(),
                openai_key.as_deref(),
            );
            for (model, response) in responses {
                println!("\n[{}]\n{}", model, response);
            }

            last_query_time = Instant::now();
            if matches!(reason, TriggerReason::JitterSpike(_)) {
                last_spike_time = Instant::now();
            }
        }

        last_cpu = snapshot.cpu_load_percent;
        thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}

fn determine_trigger(
    pulse: &PulseSnapshot,
    last_cpu: f64,
    last_query_time: Instant,
    last_spike_time: Instant,
    human_input: Option<String>,
) -> Option<TriggerReason> {
    let elapsed = last_query_time.elapsed().as_secs();
    let spike_elapsed = last_spike_time.elapsed().as_secs();

    if let Some(msg) = human_input {
        return Some(TriggerReason::HumanInput(msg));
    }
    if pulse.jitter_ms > JITTER_THRESHOLD_MS && spike_elapsed >= SPIKE_COOLDOWN_SECS {
        return Some(TriggerReason::JitterSpike(pulse.jitter_ms));
    }
    let cpu_delta = (pulse.cpu_load_percent - last_cpu).abs();
    if cpu_delta > CPU_DELTA_THRESHOLD && elapsed >= MIN_INTERVAL_SECS {
        return Some(TriggerReason::CpuDelta(cpu_delta));
    }
    if elapsed >= HEARTBEAT_INTERVAL_SECS {
        return Some(TriggerReason::Heartbeat);
    }
    None
}

fn print_trigger_banner(event: &TriggerEvent, query_num: u32) {
    let reason_str = match &event.trigger_reason {
        TriggerReason::JitterSpike(j) => format!("JITTER SPIKE: {:.4}ms", j),
        TriggerReason::CpuDelta(d) => format!("CPU DELTA: {:.1}%", d),
        TriggerReason::Heartbeat => "HEARTBEAT".to_string(),
        TriggerReason::HumanInput(msg) => format!("HUMAN: {}", &msg[..msg.len().min(40)]),
    };
    println!("\n===============================================================");
    println!(
        "  QUERY #{} | {} | P#{}",
        query_num, reason_str, event.pulse.id
    );
    println!("===============================================================");
}

fn build_user_message(event: &TriggerEvent) -> String {
    let p = &event.pulse;
    format!("TELEMETRY: Pulse #{} | Jitter: {:.6}ms | CPU: {:.1}% | Damping: {:.6}\nWhat do you observe?",
        p.id, p.jitter_ms, p.cpu_load_percent, p.entropy_damping)
}

fn query_all_models(
    client: &Client,
    ollama_client: &Client,
    event: &TriggerEvent,
    anthropic_key: Option<&str>,
    openai_key: Option<&str>,
) -> Vec<(String, String)> {
    let user_content = build_user_message(event);
    let mut handles = vec![];

    if let Some(key) = anthropic_key {
        let client = client.clone();
        let key = key.to_string();
        let content = user_content.clone();
        handles.push(thread::spawn(move || {
            ("Claude".to_string(), query_claude(&client, &key, &content))
        }));
    }
    if let Some(key) = openai_key {
        let client = client.clone();
        let key = key.to_string();
        let content = user_content.clone();
        handles.push(thread::spawn(move || {
            ("GPT-4o".to_string(), query_gpt4o(&client, &key, &content))
        }));
    }
    {
        let client = ollama_client.clone();
        let content = user_content.clone();
        handles.push(thread::spawn(move || {
            ("Ollama".to_string(), query_ollama(&client, &content))
        }));
    }
    handles.into_iter().filter_map(|h| h.join().ok()).collect()
}

fn query_claude(client: &Client, api_key: &str, user_content: &str) -> String {
    let body = json!({"model": "claude-sonnet-4-20250514", "max_tokens": 300, "system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": user_content}]});
    match client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
    {
        Ok(resp) => resp
            .json::<Value>()
            .ok()
            .and_then(|j| j["content"][0]["text"].as_str().map(String::from))
            .unwrap_or_else(|| "[Error]".into()),
        Err(e) => format!("[Error: {}]", e),
    }
}

fn query_gpt4o(client: &Client, api_key: &str, user_content: &str) -> String {
    let body = json!({"model": "gpt-4o", "max_tokens": 300, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]});
    match client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
        .send()
    {
        Ok(resp) => resp
            .json::<Value>()
            .ok()
            .and_then(|j| {
                j["choices"][0]["message"]["content"]
                    .as_str()
                    .map(String::from)
            })
            .unwrap_or_else(|| "[Error]".into()),
        Err(e) => format!("[Error: {}]", e),
    }
}

fn query_ollama(client: &Client, user_content: &str) -> String {
    let body = json!({"model": "llama3.2", "prompt": format!("{}\n\n{}", SYSTEM_PROMPT, user_content), "stream": false});
    match client
        .post("http://localhost:11434/api/generate")
        .json(&body)
        .send()
    {
        Ok(resp) => resp
            .json::<Value>()
            .ok()
            .and_then(|j| j["response"].as_str().map(String::from))
            .unwrap_or_else(|| "[Error]".into()),
        Err(e) => format!("[Error: {}]", e),
    }
}
