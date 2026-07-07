#[cfg(not(target_vendor = "apple"))]
compile_error!("kanshi supports Apple platforms only (macOS and iOS)");

mod chart;
mod color;
mod disk_row;
mod format;
mod host_info;
mod info;
mod net_interface;
mod panels;
mod process_row;
mod sampler;
mod state;
mod telemetry;
mod widgets;

use std::{
    io::{self, IsTerminal},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use keisoku::{
    Bandwidth, Battery, Chip, CpuUsage, EfficiencyCores, Fans, GpuCores, GpuUsage, Instant as Gauges, Interval, Memory,
    NeuralEngine, PerformanceCores, Power, RamTotal, SensorKind, Static, Temps,
};
use ratatui::DefaultTerminal;

use crate::{
    panels::draw,
    sampler::sample_loop,
    state::{cycle_theme, data_version, slow_down, speed_up, toggle_background, toggle_info},
    telemetry::Telemetry,
};

fn main() -> io::Result<()> {
    if std::env::args().any(|arg| arg == "--once") || !io::stdout().is_terminal() {
        return print_once();
    }

    let telemetry = Arc::new(Mutex::new(Telemetry::default()));
    let stop = Arc::new(AtomicBool::new(false));

    let sampler = {
        let telemetry = Arc::clone(&telemetry);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || sample_loop(&telemetry, &stop))
    };

    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &telemetry);
    ratatui::restore();

    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    result
}

fn print_once() -> io::Result<()> {
    let (chip, efficiency_cores, performance_cores, gpu_cores, ram_total) =
        Static::<(Chip, EfficiencyCores, PerformanceCores, GpuCores, RamTotal)>::new().into_inner();

    let mut soc = Interval::<(CpuUsage, GpuUsage, NeuralEngine, Power, Bandwidth)>::new();
    let session = soc.begin();
    std::thread::sleep(Duration::from_millis(300));
    let (cpu, gpu, neural_engine, power, bandwidth) = soc.end(session);

    let mut gauges = Gauges::<(Memory, Fans, Battery, Temps)>::new();
    let (memory, fans, battery, temperatures) = gauges.read();

    let sensors = keisoku::sensors(SensorKind::Temperature);

    println!("kanshi — available telemetry");
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
    println!("power      {:.2} W package", power.package.value());
    println!("bandwidth  R {:.1} / W {:.1} GB/s", bandwidth.dram_read.value(), bandwidth.dram_write.value());
    println!("fans       {}", fans.map(|fans| fans.fans.len()).unwrap_or(0));
    println!(
        "battery    {}",
        battery.map(|battery| format!("{:.0}%", battery.percent.value())).unwrap_or_else(|| "--".to_string()),
    );
    println!("sensors    {}", sensors.len());

    if let Some(memory) = &memory {
        println!(
            "  ram      {:.2} / {:.2} GB",
            memory.ram_usage.value() as f64 / 1e9,
            memory.ram_total.value() as f64 / 1e9,
        );
    }
    if let Some(temperatures) = &temperatures {
        if let Some(cpu) = temperatures.cpu_average {
            println!("  cpu temp   {:.1}°C", cpu.value());
        }
        if let Some(gpu) = temperatures.gpu_average {
            println!("  gpu temp   {:.1}°C", gpu.value());
        }
    }
    for sensor in sensors.iter().take(24) {
        println!("  {:<26} {:>8.2}  [{}]", sensor.name, sensor.value, sensor.component);
    }
    Ok(())
}

fn run(
    terminal: &mut DefaultTerminal,
    telemetry: &Arc<Mutex<Telemetry>>,
) -> io::Result<()> {
    let mut shown_version = u64::MAX;
    let mut redraw = true;
    loop {
        let version = data_version();
        if redraw || version != shown_version {
            shown_version = version;
            redraw = false;
            let state = telemetry.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            terminal.draw(|frame| draw(frame, &state))?;
        }
        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char('c') if ctrl => return Ok(()),
                        KeyCode::Char('c') => cycle_theme(),
                        KeyCode::Char('b') => toggle_background(),
                        KeyCode::Char('i') => toggle_info(),
                        KeyCode::Char('+') | KeyCode::Char('=') => speed_up(),
                        KeyCode::Char('-') | KeyCode::Char('_') => slow_down(),
                        _ => {},
                    }
                    redraw = true;
                },
                Event::Resize(_, _) => redraw = true,
                _ => {},
            }
        }
    }
}
