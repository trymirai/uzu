//! A mactop-style live system monitor built on `keisoku`, using the same
//! `ratatui`/`crossterm` stack as the rest of the workspace.
//!
//! `cargo run -p keisoku --example monitor`  —  press `q`/`Esc`/`Ctrl-C` to quit.

use std::{
    collections::VecDeque,
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use keisoku::{Collector, Device, Snapshot};
use ratatui::{
    Frame, Terminal,
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
};

const HISTORY: usize = 240;
const SAMPLE_INTERVAL: Duration = Duration::from_millis(1000);

/// Live state shared between the sampler thread and the render loop.
#[derive(Default)]
struct Monitor {
    device: Option<Device>,
    latest: Option<Snapshot>,
    power_history: VecDeque<f64>,
}

fn main() -> io::Result<()> {
    let monitor = Arc::new(Mutex::new(Monitor::default()));
    let stop = Arc::new(AtomicBool::new(false));

    // The Collector holds raw IOReport pointers (`!Send`) and `sample` blocks
    // for the window, so it lives in its own thread and publishes the latest
    // snapshot; the render loop stays responsive.
    let sampler = {
        let monitor = Arc::clone(&monitor);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut collector = Collector::new();
            let device = collector.device();
            if let Ok(mut state) = monitor.lock() {
                state.device = Some(device);
            }
            while !stop.load(Ordering::Relaxed) {
                let snapshot = collector.sample(SAMPLE_INTERVAL);
                if let Ok(mut state) = monitor.lock() {
                    if let Some(power) = &snapshot.power {
                        state.power_history.push_back(power.total.value() as f64);
                        while state.power_history.len() > HISTORY {
                            state.power_history.pop_front();
                        }
                    }
                    state.latest = Some(snapshot);
                }
            }
        })
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = run(&mut terminal, &monitor);

    stop.store(true, Ordering::Relaxed);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    let _ = sampler.join();
    result
}

fn run<B: Backend>(
    terminal: &mut Terminal<B>,
    monitor: &Arc<Mutex<Monitor>>,
) -> io::Result<()> {
    loop {
        // Copy the data out under the lock, then draw without holding it.
        let (device, latest, power_history) = {
            let state = monitor.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            (state.device.clone(), state.latest.clone(), state.power_history.iter().copied().collect::<Vec<_>>())
        };
        terminal
            .draw(|frame| draw(frame, device.as_ref(), latest.as_ref(), &power_history))
            .map_err(|error| io::Error::other(error.to_string()))?;

        if event::poll(Duration::from_millis(200))?
            && let Event::Key(key) = event::read()?
        {
            let quit = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL));
            if quit {
                return Ok(());
            }
        }
    }
}

fn draw(
    frame: &mut Frame,
    device: Option<&Device>,
    latest: Option<&Snapshot>,
    power_history: &[f64],
) {
    let title = device.map(device_title).unwrap_or_else(|| " keisoku monitor ".to_string());
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(title, Style::default().add_modifier(Modifier::BOLD).fg(Color::White)));
    let inner = outer.inner(frame.area());
    frame.render_widget(outer, frame.area());

    let Some(snapshot) = latest else {
        frame.render_widget(Paragraph::new("\n  sampling…").style(Style::default().fg(Color::DarkGray)), inner);
        return;
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // CPU
            Constraint::Length(3), // GPU
            Constraint::Length(3), // ANE
            Constraint::Length(3), // RAM
            Constraint::Min(4),    // power graph
            Constraint::Length(5), // stats (DRAM / temps / thermal)
            Constraint::Length(1), // footer
        ])
        .split(inner);

    if let Some(cpu) = &snapshot.cpu {
        let label = format!(
            "CPU {:>5.1}%    E {:.2} GHz    P {:.2} GHz",
            cpu.usage.value(),
            cpu.ecpu_frequency.value() as f64 / 1000.0,
            cpu.pcpu_frequency.value() as f64 / 1000.0,
        );
        render_gauge(frame, rows[0], Color::Cyan, ratio(cpu.usage.value(), 100.0), label);
    } else {
        render_unavailable(frame, rows[0], "CPU");
    }

    if let Some(gpu) = &snapshot.gpu {
        let label = format!("GPU {:>5.1}%    {} MHz", gpu.usage.value(), gpu.frequency.value());
        render_gauge(frame, rows[1], Color::Green, ratio(gpu.usage.value(), 100.0), label);
    } else {
        render_unavailable(frame, rows[1], "GPU");
    }

    if let Some(ane) = &snapshot.neural_engine {
        let label = format!("ANE {:>5.1}%    {:.2} W", ane.active.value(), ane.power.value());
        render_gauge(frame, rows[2], Color::Magenta, ratio(ane.active.value(), 100.0), label);
    } else {
        render_unavailable(frame, rows[2], "ANE");
    }

    if let Some(memory) = &snapshot.memory {
        let used = memory.ram_usage.value() as f64;
        let total = memory.ram_total.value() as f64;
        let label = format!("RAM   {:.1} / {:.1} GB", used / 1e9, total / 1e9);
        render_gauge(
            frame,
            rows[3],
            Color::Blue,
            if total > 0.0 {
                used / total
            } else {
                0.0
            },
            label,
        );
    } else {
        render_unavailable(frame, rows[3], "RAM");
    }

    render_power(frame, rows[4], snapshot, power_history);
    render_stats(frame, rows[5], snapshot);
    frame.render_widget(
        Paragraph::new(" q/Esc quit · 1s sampling").style(Style::default().fg(Color::DarkGray)),
        rows[6],
    );
}

fn render_power(
    frame: &mut Frame,
    area: Rect,
    snapshot: &Snapshot,
    power_history: &[f64],
) {
    let block = Block::default().borders(Borders::ALL).title(" Power ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = Vec::new();
    if let Some(power) = &snapshot.power {
        lines.push(Line::from(format!(
            "total {:>6.2} W    cpu {:.2}  gpu {:.2} (sram {:.2})  ane {:.2}  ram {:.2}",
            power.total.value(),
            power.cpu.value(),
            power.gpu.value(),
            power.gpu_sram.value(),
            power.ane.value(),
            power.ram.value(),
        )));
    } else {
        lines.push(Line::from(Span::styled("power unavailable", Style::default().fg(Color::DarkGray))));
    }
    let graph = sparkline(power_history, inner.width as usize);
    lines.push(Line::from(Span::styled(graph, Style::default().fg(Color::Yellow))));
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_stats(
    frame: &mut Frame,
    area: Rect,
    snapshot: &Snapshot,
) {
    let mut lines = Vec::new();
    if let Some(bandwidth) = &snapshot.bandwidth {
        lines.push(Line::from(format!(
            "DRAM    read {:>6.1}    write {:>6.1} GB/s",
            bandwidth.dram_read.value(),
            bandwidth.dram_write.value(),
        )));
    }
    if let Some(temperatures) = &snapshot.temperatures {
        lines.push(Line::from(format!(
            "Temp    CPU {:>5.1}°C    GPU {:>5.1}°C    sensors {}",
            temperatures.cpu_average.value(),
            temperatures.gpu_average.value(),
            snapshot.sensors.len(),
        )));
    }
    let thermal = snapshot.thermal_pressure.map(|state| format!("{state:?}")).unwrap_or_else(|| "—".to_string());
    let thermal_color = if matches!(thermal.as_str(), "Nominal" | "—") {
        Color::Green
    } else {
        Color::Red
    };
    lines.push(Line::from(vec![Span::raw("Thermal "), Span::styled(thermal, Style::default().fg(thermal_color))]));

    frame.render_widget(Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" System ")), area);
}

fn render_gauge(
    frame: &mut Frame,
    area: Rect,
    color: Color,
    ratio: f64,
    label: String,
) {
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL))
        .gauge_style(Style::default().fg(color).bg(Color::Black))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(label);
    frame.render_widget(gauge, area);
}

fn render_unavailable(
    frame: &mut Frame,
    area: Rect,
    name: &str,
) {
    let text = format!("{name}   unavailable on this platform");
    frame.render_widget(
        Paragraph::new(Span::styled(text, Style::default().fg(Color::DarkGray)))
            .block(Block::default().borders(Borders::ALL)),
        area,
    );
}

fn ratio(
    value: f32,
    full: f32,
) -> f64 {
    if full > 0.0 {
        (value / full) as f64
    } else {
        0.0
    }
}

fn device_title(device: &Device) -> String {
    format!(
        " {} · {} · {}E/{}P · {} GPU · {:.0} GB ",
        device.chip,
        device.os,
        device.efficiency_cores,
        device.performance_cores,
        device.gpu_cores,
        device.ram_total.value() as f64 / 1e9,
    )
}

/// A block-glyph history graph (`▁▂▃▄▅▆▇█`), auto-scaled to the window's peak.
fn sparkline(
    history: &[f64],
    width: usize,
) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if history.is_empty() || width == 0 {
        return String::new();
    }
    let window = &history[history.len().saturating_sub(width)..];
    let peak = window.iter().copied().fold(0.0_f64, f64::max).max(1e-9);
    window
        .iter()
        .map(|&value| {
            let index = ((value / peak) * (BARS.len() - 1) as f64).round().clamp(0.0, 7.0) as usize;
            BARS[index]
        })
        .collect()
}
