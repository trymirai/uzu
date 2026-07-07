#[cfg(target_os = "macos")]
#[test]
fn available_telemetry() {
    use std::time::Duration;

    use keisoku::{
        Bandwidth, Battery, Chip, CpuUsage, CurrentSensors, EfficiencyCores, Fans, GpuCores, GpuUsage, Instant,
        Interval, Memory, NeuralEngine, PerformanceCores, Power, RailPower, RamTotal, Static, TemperatureSensors,
        Temps, VoltageSensors,
    };

    let (chip, efficiency_cores, performance_cores, gpu_cores, ram_total) =
        Static::<(Chip, EfficiencyCores, PerformanceCores, GpuCores, RamTotal)>::new().into_inner();

    let mut soc = Interval::<(CpuUsage, GpuUsage, NeuralEngine, Power, Bandwidth)>::new();
    let session = soc.begin();
    std::thread::sleep(Duration::from_millis(300));
    let (cpu, gpu, neural_engine, power, bandwidth) = soc.end(session);

    let mut gauges =
        Instant::<(Memory, Fans, Battery, Temps, TemperatureSensors, VoltageSensors, CurrentSensors, RailPower)>::new();
    let (memory, fans, battery, temperatures, sensors, voltage, current, rail_power) = gauges.read();

    println!("--- keisoku available telemetry ---");
    println!(
        "device     {}  {}E+{}P  {} GPU  {:.1} GB",
        chip,
        efficiency_cores,
        performance_cores,
        gpu_cores,
        ram_total.value() as f64 / 1e9,
    );
    println!(
        "cpu        {:.2}% @ E{}/P{} MHz",
        cpu.usage.value(),
        cpu.ecpu_frequency.value(),
        cpu.pcpu_frequency.value(),
    );
    println!("gpu        {:.0}% @ {} MHz", gpu.usage.value(), gpu.frequency.value());
    println!("neural     {:.2}%", neural_engine.active.value());
    println!("power      {:.2} W package  {:.2} W total", power.package.value(), power.total().value());
    println!("bandwidth  R {:.1} / W {:.1} GB/s", bandwidth.dram_read.value(), bandwidth.dram_write.value());
    println!("fans       {}", fans.map(|fans| fans.fans.len()).unwrap_or(0));
    println!(
        "battery    {}",
        battery.map(|battery| format!("{:.0}%", battery.percent.value())).unwrap_or_else(|| "--".to_string()),
    );
    println!("sensors    {}", sensors.len());
    println!("voltage    {}", voltage.len());
    println!("current    {}", current.len());
    println!("rail_power {}", rail_power.map_or("--".into(), |watts| format!("{:.2} W", watts.value())));

    if let Some(memory) = &memory {
        println!(
            "  ram      {:.2} / {:.2} GB",
            memory.ram_usage.value() as f64 / 1e9,
            memory.ram_total.value() as f64 / 1e9,
        );
    }
    if let Some(temperatures) = &temperatures {
        if let Some(cpu) = temperatures.cpu_average {
            println!("  cpu temp {:.1} C", cpu.value());
        }
        if let Some(gpu) = temperatures.gpu_average {
            println!("  gpu temp {:.1} C", gpu.value());
        }
    }
    for sensor in sensors.iter().take(24) {
        println!("  {:<26} {:>8.2}  [{}]", sensor.name, sensor.value, sensor.component);
    }
    for volts in voltage.iter() {
        println!("  {:<26} {:>8.3} V  [{}]", volts.name, volts.value, volts.component);
    }
    for amps in current.iter() {
        println!("  {:<26} {:>8.3} A  [{}]", amps.name, amps.value, amps.component);
    }
}
