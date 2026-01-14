use cpal::traits::{DeviceTrait, HostTrait};

fn main() {
    let host = cpal::default_host();

    println!(
        "Default input: {:?}",
        host.default_input_device().map(|d| d.name())
    );
    println!("\nAll input devices:");

    if let Ok(devices) = host.input_devices() {
        for (i, device) in devices.enumerate() {
            let name = device.name().unwrap_or_else(|_| "???".into());
            let config = device.default_input_config();
            println!("  [{}] {} {:?}", i, name, config.map(|c| c.sample_rate()));
        }
    }
}
