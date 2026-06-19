use std::time::Duration;

use keisoku::Collector;

#[test]
fn available_telemetry() {
    let mut collector = Collector::new();
    let device = collector.device();
    let snapshot = collector.sample(Duration::from_millis(300));
    let flag = |present: bool| {
        if present {
            "yes"
        } else {
            "--"
        }
    };

    println!("--- keisoku available telemetry ---");
    println!(
        "device     {}  {}E+{}P  {} GPU  {:.1} GB",
        device.chip,
        device.efficiency_cores,
        device.performance_cores,
        device.gpu_cores,
        device.ram_total.value() as f64 / 1e9,
    );
    println!("cpu        {}", flag(snapshot.cpu.is_some()));
    println!("gpu        {}", flag(snapshot.gpu.is_some()));
    println!("neural     {}", flag(snapshot.neural_engine.is_some()));
    println!("power      {}", flag(snapshot.power.is_some()));
    println!("bandwidth  {}", flag(snapshot.bandwidth.is_some()));
    println!("memory     {}", flag(snapshot.memory.is_some()));
    println!("fans       {}", flag(snapshot.fans.is_some()));
    println!("battery    {}", flag(snapshot.battery.is_some()));
    println!("temps      {}", flag(snapshot.temperatures.is_some()));
    println!("sensors    {}", snapshot.sensors.len());

    if let Some(memory) = &snapshot.memory {
        println!(
            "  ram      {:.2} / {:.2} GB",
            memory.ram_usage.value() as f64 / 1e9,
            memory.ram_total.value() as f64 / 1e9,
        );
    }
    if let Some(temperatures) = &snapshot.temperatures {
        println!("  cpu {:.1} C   gpu {:.1} C", temperatures.cpu_average.value(), temperatures.gpu_average.value());
    }
    for sensor in snapshot.sensors.iter().take(24) {
        println!("  {:<26} {:>8.2}  [{}]", sensor.name, sensor.value, sensor.component);
    }
}
