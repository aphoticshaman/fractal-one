//! ═══════════════════════════════════════════════════════════════════════════════
//! COMMAND_MODULE — Interactive Control
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::neuro_link::Synapse;
use anyhow::Result;
use serde_json::Value;
use std::{
    fs,
    io::{self, Write},
    path::Path,
    sync::mpsc,
    thread,
};

pub async fn run() -> Result<()> {
    println!("\x1b[32m[COMMAND] VETTED HANDSHAKE ACTIVE\x1b[0m");
    println!("\x1b[32m[COMMAND] Type message for Archon, or 'y/n' for proposals.\x1b[0m");

    let mut synapse = Synapse::connect(false);
    let proposal_path = "proposal.json";

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || loop {
        let mut buffer = String::new();
        if io::stdin().read_line(&mut buffer).is_ok() {
            let _ = tx.send(buffer.trim().to_string());
        }
    });

    loop {
        if synapse.check_kill_signal() {
            break;
        }

        if Path::new(proposal_path).exists() {
            if let Ok(data) = fs::read_to_string(proposal_path) {
                if let Ok(json) = serde_json::from_str::<Value>(&data) {
                    let cmd = json["command"].as_str().unwrap_or("");
                    let reason = json["reason"].as_str().unwrap_or("No reason");

                    println!("\n\x1b[33m[!] ARCHON PROPOSAL:\x1b[0m");
                    println!("REASON: {}", reason);
                    println!("ACTION: {}", cmd);
                    print!("AUTHORIZE? (y/n): ");
                    io::stdout().flush()?;

                    if let Ok(confirm) = rx.recv() {
                        if confirm.to_lowercase() == "y" {
                            execute_vetted_command(cmd, &mut synapse);
                            println!("\x1b[32m[+] EXECUTED.\x1b[0m");
                        } else {
                            println!("\x1b[31m[-] REJECTED.\x1b[0m");
                        }
                    }
                    let _ = fs::remove_file(proposal_path);
                    continue;
                }
            }
        }

        if let Ok(msg) = rx.try_recv() {
            if !msg.is_empty() {
                let _ = fs::write("synapse_input.txt", msg);
                println!("\x1b[33m[COMMAND] Relay sent.\x1b[0m");
            }
        }

        thread::sleep(std::time::Duration::from_millis(100));
    }
    Ok(())
}

fn execute_vetted_command(cmd: &str, synapse: &mut Synapse) {
    if cmd.contains("SHUTDOWN") {
        synapse.send_kill_signal();
    } else if cmd.contains("SET_SCHEDULER") {
        if let Some(val) = cmd.split(':').last() {
            if let Ok(ms) = val.trim_matches(|c: char| !c.is_numeric()).parse::<u64>() {
                synapse.set_target_interval(ms);
            }
        }
    }
}
