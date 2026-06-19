mod chart;
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
use keisoku::Collector;
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

    println!("kanshi — available telemetry");
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
        println!("  cpu {:.1}°C   gpu {:.1}°C", temperatures.cpu_average.value(), temperatures.gpu_average.value(),);
    }
    for sensor in snapshot.sensors.iter().take(24) {
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
