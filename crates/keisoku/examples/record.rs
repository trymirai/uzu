//! Records a telemetry session and prints a live readout + a JSON export.
//!
//! Usage: `cargo run -p keisoku --example record -- [seconds] [out.json]`

use std::time::{Duration, Instant};

use keisoku::{Config, Session};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let seconds: u64 = arguments.next().and_then(|argument| argument.parse().ok()).unwrap_or(5);
    let output_path = arguments.next().unwrap_or_else(|| "keisoku-session.json".to_string());

    println!("recording {seconds}s (sensors available: {})...", keisoku::sensors_available());
    let handle = keisoku::start(Config {
        interval: Duration::from_millis(1000),
    });
    handle.mark("start");

    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut tick = 0;
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1000));
        tick += 1;
        if tick == seconds / 2 {
            handle.mark("midpoint");
        }
    }
    handle.mark("stop");

    let session = handle.stop();
    print_session(&session);

    match session.write_json(&output_path) {
        Ok(()) => println!("\nwrote {} snapshots to {output_path}", session.snapshots.len()),
        Err(error) => eprintln!("\nfailed to write {output_path}: {error}"),
    }
}

fn print_session(session: &Session) {
    let device = &session.device;
    println!(
        "\n{} — {} ({}E/{}P cores, {} GPU cores, {:.1} GB RAM)",
        device.chip,
        device.os,
        device.efficiency_cores,
        device.performance_cores,
        device.gpu_cores,
        device.ram_total as f64 / 1e9,
    );
    println!(
        "\n{:>6}  {:>5}  {:>5}  {:>5}  {:>6}  {:>9}  {:>5}  {:>5}",
        "t(ms)", "cpu%", "gpu%", "ane%", "pwr(W)", "DRAM GB/s", "cpu°", "gpu°",
    );
    for snapshot in &session.snapshots {
        let cpu_percent = snapshot.cpu.as_ref().map(|cpu| cpu.usage_percent).unwrap_or(0.0);
        let gpu_percent = snapshot.gpu.as_ref().map(|gpu| gpu.usage_percent).unwrap_or(0.0);
        let ane_percent = snapshot.neural_engine.as_ref().map(|ane| ane.active_percent).unwrap_or(0.0);
        let total_watts = snapshot.power.as_ref().map(|power| power.total_watts).unwrap_or(0.0);
        let (dram_read, dram_write) = snapshot
            .bandwidth
            .as_ref()
            .map(|bandwidth| (bandwidth.dram_read_gbps, bandwidth.dram_write_gbps))
            .unwrap_or((0.0, 0.0));
        let (cpu_celsius, gpu_celsius) = snapshot
            .temperatures
            .as_ref()
            .map(|temperatures| (temperatures.cpu_average, temperatures.gpu_average))
            .unwrap_or((0.0, 0.0));
        println!(
            "{:>6}  {:>5.1}  {:>5.1}  {:>5.1}  {:>6.2}  {:>4.0}/{:<4.0}  {:>5.1}  {:>5.1}",
            snapshot.elapsed_milliseconds,
            cpu_percent,
            gpu_percent,
            ane_percent,
            total_watts,
            dram_read,
            dram_write,
            cpu_celsius,
            gpu_celsius,
        );
    }
    if let Some(snapshot) = session.snapshots.last() {
        if let Some(power) = &snapshot.power {
            println!(
                "\nlast power: cpu {:.2}W  gpu {:.2}W (sram {:.2}W)  ane {:.2}W  ram {:.2}W  total {:.2}W",
                power.cpu_watts,
                power.gpu_watts,
                power.gpu_sram_watts,
                power.ane_watts,
                power.ram_watts,
                power.total_watts,
            );
        }
        if let Some(ane) = &snapshot.neural_engine {
            println!(
                "last ANE:   {:.1}% active, {:.2}W, bandwidth {:.1}/{:.1} GB/s (r/w)",
                ane.active_percent, ane.power_watts, ane.read_bandwidth_gbps, ane.write_bandwidth_gbps,
            );
        }
        if let Some(thermal_pressure) = snapshot.thermal_pressure {
            println!("thermal pressure: {thermal_pressure:?}");
        }
    }
    for marker in &session.markers {
        println!("  marker @ {:>6}ms: {}", marker.elapsed_milliseconds, marker.label);
    }
}
