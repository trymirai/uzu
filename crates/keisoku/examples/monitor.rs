//! A faithful recreation of the `mactop` default layout, built on `keisoku`
//! for the SoC telemetry (CPU/GPU/ANE/power/DRAM/temps) and `sysinfo` for the
//! OS-level panels (processes, network, disks), using the same
//! `ratatui`/`crossterm` stack as the rest of the workspace.
//!
//! `cargo run -p keisoku --example monitor`  —  press `q`/`Esc`/`Ctrl-C` to quit.
//!
//! Layout (mactop "default", all green, rounded outer frame):
//! ```text
//! ┌ keisoku · <chip> · <cores> · <gpu> · <ram> ───────────────── 0.5 ┐
//! │ CPU gauge                       │ GPU gauge                      │
//! ├─────────────────────────────────┼────────────────────────────────┤
//! │ ANE gauge                       │ Memory gauge                   │
//! │ Power Usage   │ <power spark>   │ Apple Silicon │ Network & Disk │
//! ├─────────────────────────────────┴────────────────────────────────┤
//! │ Process List                                                      │
//! └ 1/1 layout (green) ───────────── Exit: q ──────────────── 1000ms ┘
//! ```

use std::{
    collections::VecDeque,
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
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
    style::{Color, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        Block, BorderType, Borders, Gauge, List, ListItem, Paragraph,
        canvas::{Canvas, Line as CanvasLine},
    },
};
use sysinfo::{Disks, Networks, ProcessesToUpdate, System};

const GREEN: Color = Color::Green;
const LABEL: Color = Color::Indexed(245); // grey, like mactop's gauge label
const HISTORY: usize = 256;
const SAMPLE_INTERVAL: Duration = Duration::from_millis(1000);
const PROCESS_ROWS: usize = 8;

/// A process row for the bottom list.
struct ProcessRow {
    cpu: f32,
    memory: u64,
    name: String,
}

/// A disk row for the network/disk panel.
struct DiskRow {
    name: String,
    used: u64,
    total: u64,
}

/// Live state shared between the sampler thread and the render loop.
#[derive(Default)]
struct Telemetry {
    device: Option<Device>,
    snapshot: Option<Snapshot>,
    power_history: VecDeque<f64>,
    max_power: f64,
    uptime_seconds: u64,
    network_down: f64,
    network_up: f64,
    disks: Vec<DiskRow>,
    processes: Vec<ProcessRow>,
}

fn main() -> io::Result<()> {
    let telemetry = Arc::new(Mutex::new(Telemetry::default()));
    let stop = Arc::new(AtomicBool::new(false));

    let sampler = {
        let telemetry = Arc::clone(&telemetry);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || sample_loop(&telemetry, &stop))
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = run(&mut terminal, &telemetry);

    stop.store(true, Ordering::Relaxed);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    let _ = sampler.join();
    result
}

/// Background sampler: `keisoku` for SoC telemetry (blocks the window),
/// `sysinfo` for processes / network / disks, published into shared state.
fn sample_loop(
    telemetry: &Arc<Mutex<Telemetry>>,
    stop: &Arc<AtomicBool>,
) {
    let mut collector = Collector::new();
    let device = collector.device();
    if let Ok(mut state) = telemetry.lock() {
        state.device = Some(device);
    }

    let mut system = System::new();
    let mut networks = Networks::new_with_refreshed_list();
    let mut last_refresh = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        let snapshot = collector.sample(SAMPLE_INTERVAL);

        system.refresh_processes(ProcessesToUpdate::All, true);
        networks.refresh(true);
        let elapsed = last_refresh.elapsed().as_secs_f64().max(0.001);
        last_refresh = Instant::now();

        let (mut received, mut transmitted) = (0u64, 0u64);
        for data in networks.values() {
            received += data.received();
            transmitted += data.transmitted();
        }

        let disks = Disks::new_with_refreshed_list()
            .list()
            .iter()
            .map(|disk| DiskRow {
                name: disk.name().to_string_lossy().into_owned(),
                used: disk.total_space().saturating_sub(disk.available_space()),
                total: disk.total_space(),
            })
            .collect::<Vec<_>>();

        let mut processes = system
            .processes()
            .values()
            .map(|process| ProcessRow {
                cpu: process.cpu_usage(),
                memory: process.memory(),
                name: process.name().to_string_lossy().into_owned(),
            })
            .collect::<Vec<_>>();
        processes.sort_by(|a, b| b.cpu.partial_cmp(&a.cpu).unwrap_or(std::cmp::Ordering::Equal));
        processes.truncate(PROCESS_ROWS);

        if let Ok(mut state) = telemetry.lock() {
            if let Some(power) = &snapshot.power {
                let total = power.total.value() as f64;
                state.power_history.push_back(total);
                while state.power_history.len() > HISTORY {
                    state.power_history.pop_front();
                }
                state.max_power = state.max_power.max(total);
            }
            state.snapshot = Some(snapshot);
            state.uptime_seconds = System::uptime();
            state.network_down = received as f64 / elapsed;
            state.network_up = transmitted as f64 / elapsed;
            state.disks = disks;
            state.processes = processes;
        }
    }
}

fn run<B: Backend>(
    terminal: &mut Terminal<B>,
    telemetry: &Arc<Mutex<Telemetry>>,
) -> io::Result<()> {
    loop {
        {
            let state = telemetry.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            terminal.draw(|frame| draw(frame, &state)).map_err(|error| io::Error::other(error.to_string()))?;
        }
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
    state: &Telemetry,
) {
    let outer = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(GREEN))
        .title_style(Style::default().fg(GREEN))
        .title_top(Line::from(header_title(state.device.as_ref())))
        .title_top(Line::from(format!(" {} ", env!("CARGO_PKG_VERSION"))).right_aligned())
        .title_bottom(Line::from(" 1/1 layout (green) "))
        .title_bottom(Line::from(" Exit: q ").centered())
        .title_bottom(Line::from(" 1000ms ").right_aligned());
    let area = outer.inner(frame.area());
    frame.render_widget(outer, frame.area());

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(50), Constraint::Percentage(25)])
        .split(area);

    // Row 1: CPU | GPU gauges.
    let top = split_horizontal(rows[0], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_cpu(frame, top[0], state);
    render_gpu(frame, top[1], state);

    // Row 2: left (ANE + power) | right (memory + model/network).
    let middle = split_horizontal(rows[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);

    let left = split_vertical(middle[0], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_ane(frame, left[0], state);
    let left_bottom = split_horizontal(left[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_power(frame, left_bottom[0], state);
    render_power_chart(frame, left_bottom[1], state);

    let right = split_vertical(middle[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_memory(frame, right[0], state);
    let right_bottom = split_horizontal(right[1], [Constraint::Ratio(1, 3), Constraint::Ratio(2, 3)]);
    render_model(frame, right_bottom[0], state);
    render_network(frame, right_bottom[1], state);

    // Row 3: process list.
    render_processes(frame, rows[2], state);
}

fn render_cpu(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let cpu = state.snapshot.as_ref().and_then(|s| s.cpu.as_ref());
    let device = state.device.as_ref();
    let temperature = state.snapshot.as_ref().and_then(|s| s.temperatures.as_ref()).map(|t| t.cpu_average.value());
    let (title, ratio) = match (cpu, device) {
        (Some(cpu), Some(device)) => {
            let cores = device.efficiency_cores + device.performance_cores;
            let title = format!(
                "{} Cores ({}E/{}P) {:.2}% @ E{:.1}/P{:.1} GHz ({:.0}°C)",
                cores,
                device.efficiency_cores,
                device.performance_cores,
                cpu.usage.value(),
                cpu.ecpu_frequency.value() as f64 / 1000.0,
                cpu.pcpu_frequency.value() as f64 / 1000.0,
                temperature.unwrap_or(0.0),
            );
            (title, cpu.usage.value() as f64 / 100.0)
        },
        _ => ("Loading…".to_string(), 0.0),
    };
    render_gauge(frame, area, &title, ratio);
}

fn render_gpu(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let gpu = state.snapshot.as_ref().and_then(|s| s.gpu.as_ref());
    let temperature = state.snapshot.as_ref().and_then(|s| s.temperatures.as_ref()).map(|t| t.gpu_average.value());
    let (title, ratio) = match gpu {
        Some(gpu) => (
            format!(
                "GPU Usage: {:.0}% @ {} MHz ({:.0}°C)",
                gpu.usage.value(),
                gpu.frequency.value(),
                temperature.unwrap_or(0.0),
            ),
            gpu.usage.value() as f64 / 100.0,
        ),
        None => ("GPU Usage".to_string(), 0.0),
    };
    render_gauge(frame, area, &title, ratio);
}

fn render_ane(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let ane = state.snapshot.as_ref().and_then(|s| s.neural_engine.as_ref());
    let (title, ratio) = match ane {
        Some(ane) => (
            format!("ANE Usage: {:.2}% @ {:.2} W", ane.active.value(), ane.power.value()),
            ane.active.value() as f64 / 100.0,
        ),
        None => ("ANE Usage".to_string(), 0.0),
    };
    render_gauge(frame, area, &title, ratio);
}

fn render_memory(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let memory = state.snapshot.as_ref().and_then(|s| s.memory.as_ref());
    let bandwidth = state
        .snapshot
        .as_ref()
        .and_then(|s| s.bandwidth.as_ref())
        .map(|b| b.dram_read.value() + b.dram_write.value())
        .unwrap_or(0.0);
    let (title, ratio) = match memory {
        Some(memory) => {
            let used = memory.ram_usage.value() as f64;
            let total = memory.ram_total.value() as f64;
            let title = format!(
                "Mem: {:.2} GB / {:.2} GB (Swap: {:.2}/{:.2} GB) BW: {:.1} GB/s",
                used / 1e9,
                total / 1e9,
                memory.swap_usage.value() as f64 / 1e9,
                memory.swap_total.value() as f64 / 1e9,
                bandwidth,
            );
            (
                title,
                if total > 0.0 {
                    used / total
                } else {
                    0.0
                },
            )
        },
        None => ("Memory Usage".to_string(), 0.0),
    };
    render_gauge(frame, area, &title, ratio);
}

fn render_gauge(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    ratio: f64,
) {
    let ratio = ratio.clamp(0.0, 1.0);
    let gauge = Gauge::default()
        .block(panel(title))
        .gauge_style(Style::default().fg(GREEN).bg(Color::Reset))
        .use_unicode(true) // sub-cell (1/8) bar resolution for a smooth edge
        .label(Span::styled(format!("{:.0}%", ratio * 100.0), Style::default().fg(LABEL)))
        .ratio(ratio);
    frame.render_widget(gauge, area);
}

fn render_power(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let mut lines = Vec::new();
    if let Some(power) = state.snapshot.as_ref().and_then(|s| s.power.as_ref()) {
        lines.push(Line::from(format!(
            "CPU: {:.2} W | GPU: {:.2} W",
            power.cpu.value(),
            power.gpu.value() + power.gpu_sram.value(),
        )));
        lines.push(Line::from(format!("ANE: {:.2} W | DRAM: {:.2} W", power.ane.value(), power.ram.value())));
        lines.push(Line::from(format!("Total: {:.2} W", power.total.value())));
    }
    let thermals =
        state.snapshot.as_ref().and_then(|s| s.thermal_pressure).map(|t| format!("{t:?}")).unwrap_or_default();
    lines.push(Line::from(format!("Thermals: {thermals}")));
    lines.push(Line::from(format!("Uptime: {}", format_uptime(state.uptime_seconds))));
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(GREEN)).block(panel("Power Usage")), area);
}

fn render_power_chart(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let current = state.power_history.back().copied().unwrap_or(0.0);
    let average = if state.power_history.is_empty() {
        0.0
    } else {
        state.power_history.iter().sum::<f64>() / state.power_history.len() as f64
    };
    let thermals =
        state.snapshot.as_ref().and_then(|s| s.thermal_pressure).map(|t| format!("{t:?}")).unwrap_or_default();

    let block = panel_owned(format!("{current:.2} W Total (Max: {:.2} W)", state.max_power));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    // Braille filled-area chart: each cell is 2×4 dots, so a vertical line per
    // braille column gives a smooth, high-resolution graph (btop-style).
    let columns = (parts[0].width as usize * 2).max(1);
    let shown: Vec<f64> = state.power_history.iter().rev().take(columns).rev().copied().collect();
    let offset = columns - shown.len(); // keep the newest samples flush to the right edge
    let ceiling = state.max_power.max(1e-9);
    let chart =
        Canvas::default().marker(Marker::Braille).x_bounds([0.0, columns as f64]).y_bounds([0.0, ceiling]).paint(
            move |context| {
                for (index, &watts) in shown.iter().enumerate() {
                    let x = (offset + index) as f64;
                    context.draw(&CanvasLine {
                        x1: x,
                        y1: 0.0,
                        x2: x,
                        y2: watts,
                        color: GREEN,
                    });
                }
            },
        );
    frame.render_widget(chart, parts[0]);

    frame.render_widget(
        Paragraph::new(Line::from(format!("Avg: {average:.2} W | {thermals}"))).style(Style::default().fg(GREEN)),
        parts[1],
    );
}

fn render_model(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let mut lines = Vec::new();
    if let Some(device) = &state.device {
        lines.push(Line::from(device.chip.clone()));
        lines.push(Line::from(format!("{} Cores", device.efficiency_cores + device.performance_cores)));
        lines.push(Line::from(format!("{} E-Cores", device.efficiency_cores)));
        lines.push(Line::from(format!("{} P-Cores", device.performance_cores)));
        lines.push(Line::from(format!("{} GPU Cores", device.gpu_cores)));
    }
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(GREEN)).block(panel("Apple Silicon")), area);
}

fn render_network(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let mut lines = vec![Line::from(format!(
        "Net: ↑ {}/s  ↓ {}/s",
        human_bytes(state.network_up as u64),
        human_bytes(state.network_down as u64),
    ))];
    for disk in state.disks.iter().take(3) {
        lines.push(Line::from(format!("{}: {} / {}", disk.name, human_bytes(disk.used), human_bytes(disk.total),)));
    }
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(GREEN)).block(panel("Network & Disk")), area);
}

fn render_processes(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let items: Vec<ListItem> = state
        .processes
        .iter()
        .map(|process| {
            ListItem::new(format!("{:>6.1}%  {:>9}  {}", process.cpu, human_bytes(process.memory), process.name))
        })
        .collect();
    frame.render_widget(List::new(items).style(Style::default().fg(GREEN)).block(panel("Process List")), area);
}

/// A square-bordered, green panel with a title (matches mactop's inner widgets).
fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(GREEN))
        .title_style(Style::default().fg(GREEN))
        .title(title)
}

fn panel_owned(title: String) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(GREEN))
        .title_style(Style::default().fg(GREEN))
        .title(title)
}

fn split_horizontal<const N: usize>(
    area: Rect,
    constraints: [Constraint; N],
) -> std::rc::Rc<[Rect]> {
    Layout::default().direction(Direction::Horizontal).constraints(constraints).split(area)
}

fn split_vertical<const N: usize>(
    area: Rect,
    constraints: [Constraint; N],
) -> std::rc::Rc<[Rect]> {
    Layout::default().direction(Direction::Vertical).constraints(constraints).split(area)
}

fn header_title(device: Option<&Device>) -> String {
    match device {
        Some(device) => format!(
            " keisoku  •  {}  •  {}C ({}E+{}P)  •  {} GPU  •  {:.0} GB ",
            device.chip,
            device.efficiency_cores + device.performance_cores,
            device.efficiency_cores,
            device.performance_cores,
            device.gpu_cores,
            device.ram_total.value() as f64 / 1e9,
        ),
        None => " keisoku ".to_string(),
    }
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_uptime(seconds: u64) -> String {
    let (hours, minutes) = (seconds / 3600, (seconds % 3600) / 60);
    if hours > 0 {
        format!("{hours}h {minutes}m")
    } else {
        format!("{minutes}m")
    }
}
