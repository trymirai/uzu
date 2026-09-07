#[cfg(all(target_os = "macos", feature = "hardware-control"))]
fn main() -> std::process::ExitCode {
    let device = keisoku::Device::new();
    println!("Device: {}", device.chip());
    match device.fans() {
        Some(metrics) => {
            println!("Fans: {}", metrics.fans.len());
            for (index, fan) in metrics.fans.iter().enumerate() {
                println!("Fan {index}: actual {}, maximum {}, target {}", fan.actual, fan.maximum, fan.target);
            }
        },
        None => println!("Fan telemetry unavailable"),
    }
    match keisoku::PowerModeControl::new() {
        Ok(_) => println!("High Power Mode: supported on this Mac's power sources"),
        Err(error) => println!("High Power Mode unavailable: {error}"),
    }
    match keisoku::GpuPowerControl::new()
        .and_then(|gpu| Ok((gpu.power_limit_milliwatts()?, gpu.maximum_power_milliwatts()?)))
    {
        Ok((limit, maximum)) => {
            println!("GPU power ceiling: {limit} mW");
            println!("GPU calibrated maximum power budget: {maximum} mW");
            std::process::ExitCode::SUCCESS
        },
        Err(error) => {
            eprintln!("GPU power properties unavailable: {error}");
            std::process::ExitCode::FAILURE
        },
    }
}

#[cfg(not(all(target_os = "macos", feature = "hardware-control")))]
fn main() -> std::process::ExitCode {
    eprintln!("This diagnostic requires macOS and --features hardware-control");
    std::process::ExitCode::FAILURE
}
