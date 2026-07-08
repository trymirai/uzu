#[cfg(target_os = "macos")]
#[test]
fn available_telemetry() {
    use std::time::Duration;

    use keisoku::{
        Bandwidth, Battery, Chip, CpuUsage, CurrentSensors, EfficiencyCores, Fans, GpuCores, GpuUsage, Instant,
        Interval, Memory, NeuralEngine, PerformanceCores, Power, RailPower, Select, TemperatureSensors, VoltageSensors,
    };

    let instant_values = Instant::<
        Select![
            Chip,
            EfficiencyCores,
            PerformanceCores,
            GpuCores,
            Memory,
            Fans,
            Battery,
            TemperatureSensors,
            VoltageSensors,
            CurrentSensors,
            RailPower
        ],
    >::new()
    .read();
    let chip = instant_values.get::<Chip>();
    let efficiency_cores = instant_values.get::<EfficiencyCores>();
    let performance_cores = instant_values.get::<PerformanceCores>();
    let gpu_cores = instant_values.get::<GpuCores>();
    let memory = instant_values.get::<Memory>();
    let fans = instant_values.get::<Fans>();
    let battery = instant_values.get::<Battery>();
    let sensors = instant_values.get::<TemperatureSensors>();
    let voltage = instant_values.get::<VoltageSensors>();
    let current = instant_values.get::<CurrentSensors>();
    let rail_power = instant_values.get::<RailPower>();

    let mut interval = Interval::<Select![CpuUsage, GpuUsage, NeuralEngine, Power, Bandwidth]>::new();
    let session = interval.start();
    std::thread::sleep(Duration::from_millis(300));
    let interval_values = interval.stop(session);
    let cpu = interval_values.get::<CpuUsage>();
    let gpu = interval_values.get::<GpuUsage>();
    let neural_engine = interval_values.get::<NeuralEngine>();
    let power = interval_values.get::<Power>();
    let bandwidth = interval_values.get::<Bandwidth>();

    println!("--- keisoku available telemetry ---");
    println!("device     {}  {}E+{}P  {} GPU", chip, efficiency_cores, performance_cores, gpu_cores);
    println!(
        "cpu        {:.2}% @ E{}/P{} MHz",
        cpu.usage.value(),
        cpu.ecpu_frequency.value(),
        cpu.pcpu_frequency.value(),
    );
    println!("gpu        {:.0}% @ {} MHz", gpu.usage.value(), gpu.frequency.value());
    println!("neural     {:.2}%", neural_engine.active.value());
    println!("power      {:.2} W total", power.total().value());
    println!("bandwidth  R {:.1} / W {:.1} GB/s", bandwidth.dram_read.value(), bandwidth.dram_write.value());
    println!("fans       {}", fans.as_ref().map(|fans| fans.fans.len()).unwrap_or(0));
    println!(
        "battery    {}",
        battery.as_ref().map(|battery| format!("{:.0}%", battery.percent.value())).unwrap_or_else(|| "--".to_string()),
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
