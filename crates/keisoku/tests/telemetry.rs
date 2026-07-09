#[test]
fn available_telemetry() {
    use keisoku::Device;

    let mut device = Device::new();
    let chip = device.chip();
    let efficiency_cores = device.efficiency_cores();
    let performance_cores = device.performance_cores();
    let gpu_cores = device.gpu_cores();
    let memory = device.memory();
    let fans = device.fans();
    let battery = device.battery();
    let thermal = device.thermal();
    let sensors = device.temperature_sensors();
    let voltage = device.voltage_sensors();
    let current = device.current_sensors();
    let rail_power = device.rail_power();

    println!("--- keisoku available telemetry ---");
    println!("device     {}  {}E+{}P  {} GPU", chip, efficiency_cores, performance_cores, gpu_cores);
    println!("fans       {}", fans.as_ref().map(|fans| fans.fans.len()).unwrap_or(0));
    println!(
        "battery    {}",
        battery.as_ref().map(|battery| format!("{:.0}%", battery.percent.value())).unwrap_or_else(|| "--".to_string()),
    );
    println!("thermal    {}", thermal.map_or_else(|| "--".to_string(), |level| format!("{level:?}")));
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

    #[cfg(target_os = "macos")]
    {
        use std::time::Duration;

        use keisoku::{
            Ane, AneBandwidth, Cpu, Device, DramBytes, DramHistogram, DramRead, DramWrite, EnergyRail, Gpu, Ram, Select,
        };

        let mut handle = Device::interval_measurement::<
            Select![
                EnergyRail<Cpu>,
                EnergyRail<Gpu>,
                EnergyRail<Ane>,
                EnergyRail<Ram>,
                AneBandwidth,
                DramBytes<DramRead>,
                DramBytes<DramWrite>,
                DramHistogram<DramRead>,
                DramHistogram<DramWrite>,
            ],
        >();
        handle.start();
        std::thread::sleep(Duration::from_millis(300));
        let sample = handle.stop().expect("interval sample");

        let cpu_energy = sample.get::<EnergyRail<Cpu>>();
        let gpu_energy = sample.get::<EnergyRail<Gpu>>();
        let ane_energy = sample.get::<EnergyRail<Ane>>();
        let ram_energy = sample.get::<EnergyRail<Ram>>();
        let ane = sample.get::<AneBandwidth>();
        let dram_read_bytes = sample.get::<DramBytes<DramRead>>();
        let dram_write_bytes = sample.get::<DramBytes<DramWrite>>();
        let dram_read = sample.get::<DramHistogram<DramRead>>();
        let dram_write = sample.get::<DramHistogram<DramWrite>>();

        println!(
            "energy     CPU {:.3} J  GPU {:.3} J  ANE {:.3} J  RAM {:.3} J",
            cpu_energy.value(),
            gpu_energy.value(),
            ane_energy.value(),
            ram_energy.value(),
        );
        println!("neural     {:.2}%", ane.value());
        println!(
            "bandwidth  R {:.1} GB/s ({} B)  W {:.1} GB/s ({} B)",
            dram_read.value(),
            dram_read_bytes.value(),
            dram_write.value(),
            dram_write_bytes.value(),
        );
    }
}
