//! A faithful recreation of the `mactop` default layout, built on `keisoku`
//! for the SoC telemetry (CPU/GPU/ANE/power/DRAM/temps) and `sysinfo` for the
//! OS-level panels (processes, network, disks), using the same
//! `ratatui`/`crossterm` stack as the rest of the workspace.
//!
//! Keys: `c` cycles the accent color, `b` toggles the background, `q`/`Esc`/
//! `Ctrl-C` quits (mactop-style). Run: `cargo run -p keisoku --example monitor`.
//!
//! Layout (mactop "default", rounded outer frame). Every usage metric is a
//! braille filled-area history chart (`⡇⢸⣿⣤`):
//! ```text
//! ┌ keisoku · <chip> · <cores> · <gpu> · <ram> ───────────────── 0.5 ┐
//! │ CPU per-core bars               │ GPU chart                      │
//! ├─────────────────────────────────┼────────────────────────────────┤
//! │ ANE chart                       │ Memory chart                   │
//! │ Power Usage   │ Package chart   │ Apple Silicon │ Net·Disk·Fans  │
//! ├─────────────────────────────────┴────────────────────────────────┤
//! │ Process List                                                      │
//! └ green ──────────── Color: c · BG: b · Exit: q ──────────── 1000ms ┘
//! ```
//! Power Usage shows per-rail watts + `Pkg` (SMC package) + `Battery`.

use std::{
    collections::VecDeque,
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use keisoku::{Collector, Device, Percent, Snapshot};
use ratatui::{
    Frame, Terminal,
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    symbols::Marker,
    text::Line,
    widgets::{
        Block, BorderType, Borders, List, ListItem, Paragraph,
        canvas::{Canvas, Line as CanvasLine},
    },
};
use sysinfo::{Disks, Networks, ProcessesToUpdate, System};

const HISTORY: usize = 256;
const SAMPLE_INTERVAL: Duration = Duration::from_millis(1000);
const PROCESS_ROWS: usize = 8;

/// Accent colors cycled with `c` (mactop-style); first is the default.
const THEMES: [(&str, Color); 7] = [
    ("green", Color::Green),
    ("cyan", Color::Cyan),
    ("blue", Color::Blue),
    ("magenta", Color::Magenta),
    ("yellow", Color::Yellow),
    ("red", Color::Red),
    ("white", Color::White),
];

static THEME_INDEX: AtomicUsize = AtomicUsize::new(0);
static DARK_BACKGROUND: AtomicBool = AtomicBool::new(false);

/// Current accent (name, color), advanced by `c`.
fn theme() -> (&'static str, Color) {
    THEMES[THEME_INDEX.load(Ordering::Relaxed) % THEMES.len()]
}

fn accent() -> Color {
    theme().1
}

/// Current background, toggled by `b`.
fn background() -> Color {
    if DARK_BACKGROUND.load(Ordering::Relaxed) {
        Color::Black
    } else {
        Color::Reset
    }
}

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
    gpu_history: VecDeque<f64>,
    ane_history: VecDeque<f64>,
    memory_history: VecDeque<f64>,
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

        let gpu = snapshot.gpu.as_ref().map(|g| g.usage.value() as f64).unwrap_or(0.0);
        let ane = snapshot.neural_engine.as_ref().map(|a| a.active.value() as f64).unwrap_or(0.0);
        let memory = snapshot
            .memory
            .as_ref()
            .filter(|m| m.ram_total.value() > 0)
            .map(|m| m.ram_usage.value() as f64 / m.ram_total.value() as f64 * 100.0)
            .unwrap_or(0.0);
        // Track whole-package power (SMC PSTR) as the headline graph.
        let power = snapshot.power.as_ref().map(|p| p.package.value() as f64).unwrap_or(0.0);

        if let Ok(mut state) = telemetry.lock() {
            push_history(&mut state.gpu_history, gpu);
            push_history(&mut state.ane_history, ane);
            push_history(&mut state.memory_history, memory);
            push_history(&mut state.power_history, power);
            state.max_power = state.max_power.max(power);
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
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                KeyCode::Char('c') if ctrl => return Ok(()),
                KeyCode::Char('c') => {
                    THEME_INDEX.fetch_add(1, Ordering::Relaxed);
                },
                KeyCode::Char('b') => {
                    DARK_BACKGROUND.fetch_xor(true, Ordering::Relaxed);
                },
                _ => {},
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
        .style(Style::default().bg(background()))
        .border_style(Style::default().fg(accent()))
        .title_style(Style::default().fg(accent()))
        .title_top(Line::from(header_title(state.device.as_ref())))
        .title_top(Line::from(format!(" {} ", env!("CARGO_PKG_VERSION"))).right_aligned())
        .title_bottom(Line::from(format!(" {} ", theme().0)))
        .title_bottom(Line::from(" Color: c · BG: b · Exit: q ").centered())
        .title_bottom(Line::from(" 1000ms ").right_aligned());
    let area = outer.inner(frame.area());
    frame.render_widget(outer, frame.area());

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(50), Constraint::Percentage(25)])
        .split(area);

    // Row 1: CPU | GPU usage history charts.
    let top = split_horizontal(rows[0], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    let per_core = state.snapshot.as_ref().and_then(|s| s.cpu.as_ref()).map(|c| c.per_core.as_slice()).unwrap_or(&[]);
    render_cpu_cores(frame, top[0], cpu_title(state), per_core);
    render_chart(frame, top[1], gpu_title(state), &state.gpu_history, 100.0);

    // Row 2: left (ANE + power) | right (memory + model/network).
    let middle = split_horizontal(rows[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);

    let left = split_vertical(middle[0], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_chart(frame, left[0], ane_title(state), &state.ane_history, 100.0);
    let left_bottom = split_horizontal(left[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_power(frame, left_bottom[0], state);
    render_chart(frame, left_bottom[1], power_title(state), &state.power_history, state.max_power);

    let right = split_vertical(middle[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_chart(frame, right[0], memory_title(state), &state.memory_history, 100.0);
    let right_bottom = split_horizontal(right[1], [Constraint::Ratio(1, 3), Constraint::Ratio(2, 3)]);
    render_model(frame, right_bottom[0], state);
    render_network(frame, right_bottom[1], state);

    // Row 3: process list.
    render_processes(frame, rows[2], state);
}

/// A bordered green panel containing a braille filled-area history chart — the
/// same `⡇⢸⣿⣤` braille rendering as the power graph, for every metric. The
/// title carries the current values; the chart shows the recent history.
fn render_chart(
    frame: &mut Frame,
    area: Rect,
    title: String,
    history: &VecDeque<f64>,
    ceiling: f64,
) {
    let block = panel_owned(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Each cell is 2×4 braille dots; one vertical line per braille column gives
    // a smooth, high-resolution filled area (btop-style), newest at the right.
    let columns = (inner.width as usize * 2).max(1);
    let shown: Vec<f64> = history.iter().rev().take(columns).rev().copied().collect();
    let offset = columns - shown.len();
    let ceiling = ceiling.max(1e-9);
    let chart = Canvas::default()
        .marker(Marker::Braille)
        .background_color(background())
        .x_bounds([0.0, columns as f64])
        .y_bounds([0.0, ceiling])
        .paint(move |context| {
            for (index, &value) in shown.iter().enumerate() {
                let x = (offset + index) as f64;
                context.draw(&CanvasLine {
                    x1: x,
                    y1: 0.0,
                    x2: x,
                    y2: value,
                    color: accent(),
                });
            }
        });
    frame.render_widget(chart, inner);
}

/// Per-core CPU as one braille bar per logical core (current %, 0-100).
fn render_cpu_cores(
    frame: &mut Frame,
    area: Rect,
    title: String,
    per_core: &[Percent],
) {
    let block = panel_owned(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if per_core.is_empty() {
        return;
    }
    let values: Vec<f64> = per_core.iter().map(|core| core.value() as f64).collect();
    let cores = values.len() as f64;
    let chart = Canvas::default()
        .marker(Marker::Braille)
        .background_color(background())
        .x_bounds([0.0, cores])
        .y_bounds([0.0, 100.0])
        .paint(move |context| {
            for (index, &value) in values.iter().enumerate() {
                let x = index as f64 + 0.5;
                context.draw(&CanvasLine {
                    x1: x,
                    y1: 0.0,
                    x2: x,
                    y2: value,
                    color: accent(),
                });
            }
        });
    frame.render_widget(chart, inner);
}

fn cpu_title(state: &Telemetry) -> String {
    let cpu = state.snapshot.as_ref().and_then(|s| s.cpu.as_ref());
    let temperature = state.snapshot.as_ref().and_then(|s| s.temperatures.as_ref()).map(|t| t.cpu_average.value());
    match (cpu, state.device.as_ref()) {
        (Some(cpu), Some(device)) => format!(
            "{} Cores ({}E/{}P) {:.2}% @ E{:.1}/P{:.1} GHz ({:.0}°C)",
            device.efficiency_cores + device.performance_cores,
            device.efficiency_cores,
            device.performance_cores,
            cpu.usage.value(),
            cpu.ecpu_frequency.value() as f64 / 1000.0,
            cpu.pcpu_frequency.value() as f64 / 1000.0,
            temperature.unwrap_or(0.0),
        ),
        _ => "CPU Usage".to_string(),
    }
}

fn gpu_title(state: &Telemetry) -> String {
    let temperature = state.snapshot.as_ref().and_then(|s| s.temperatures.as_ref()).map(|t| t.gpu_average.value());
    match state.snapshot.as_ref().and_then(|s| s.gpu.as_ref()) {
        Some(gpu) => format!(
            "GPU Usage: {:.0}% @ {} MHz ({:.0}°C)",
            gpu.usage.value(),
            gpu.frequency.value(),
            temperature.unwrap_or(0.0),
        ),
        None => "GPU Usage".to_string(),
    }
}

fn ane_title(state: &Telemetry) -> String {
    match state.snapshot.as_ref().and_then(|s| s.neural_engine.as_ref()) {
        Some(ane) => format!("ANE Usage: {:.2}% @ {:.2} W", ane.active.value(), ane.power.value()),
        None => "ANE Usage".to_string(),
    }
}

fn memory_title(state: &Telemetry) -> String {
    let bandwidth = state
        .snapshot
        .as_ref()
        .and_then(|s| s.bandwidth.as_ref())
        .map(|b| b.dram_read.value() + b.dram_write.value())
        .unwrap_or(0.0);
    match state.snapshot.as_ref().and_then(|s| s.memory.as_ref()) {
        Some(memory) => format!(
            "Mem: {:.2} GB / {:.2} GB (Swap: {:.2}/{:.2} GB) BW: {:.1} GB/s",
            memory.ram_usage.value() as f64 / 1e9,
            memory.ram_total.value() as f64 / 1e9,
            memory.swap_usage.value() as f64 / 1e9,
            memory.swap_total.value() as f64 / 1e9,
            bandwidth,
        ),
        None => "Memory Usage".to_string(),
    }
}

fn power_title(state: &Telemetry) -> String {
    let current = state.power_history.back().copied().unwrap_or(0.0);
    format!("{current:.2} W Total (Max: {:.2} W)", state.max_power)
}

fn render_power(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let mut lines = Vec::new();
    if let Some(power) = state.snapshot.as_ref().and_then(|s| s.power.as_ref()) {
        // Fixed-width labels (5) and values (6) so the `|` and right column stay
        // aligned as the numbers grow.
        lines.push(Line::from(format!(
            "{:<6}{:>6.2} W | {:<5}{:>6.2} W",
            "CPU:",
            power.cpu.value(),
            "GPU:",
            power.gpu.value() + power.gpu_sram.value(),
        )));
        lines.push(Line::from(format!(
            "{:<6}{:>6.2} W | {:<5}{:>6.2} W",
            "ANE:",
            power.ane.value(),
            "DRAM:",
            power.ram.value(),
        )));
        lines.push(Line::from(format!("{:<6}{:>6.2} W", "Total:", power.total.value())));
        lines.push(Line::from(format!("{:<6}{:>6.2} W", "Pkg:", power.package.value())));
    }
    let thermals =
        state.snapshot.as_ref().and_then(|s| s.thermal_pressure).map(|t| format!("{t:?}")).unwrap_or_default();
    lines.push(Line::from(format!("Thermals: {thermals}")));
    lines.push(Line::from(format!("Uptime: {}", format_uptime(state.uptime_seconds))));
    if let Some(battery) = state.snapshot.as_ref().and_then(|s| s.battery.as_ref()).filter(|b| b.present) {
        let status = if battery.charging {
            "charging"
        } else if battery.on_ac_power {
            "AC"
        } else {
            "battery"
        };
        lines.push(Line::from(format!("Battery: {:.0}% ({status})", battery.percent.value())));
    }
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(accent())).block(panel("Power Usage")), area);
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
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(accent())).block(panel("Apple Silicon")), area);
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
    for disk in state.disks.iter().take(2) {
        lines.push(Line::from(format!("{}: {} / {}", disk.name, human_bytes(disk.used), human_bytes(disk.total),)));
    }
    if let Some(fans) = state.snapshot.as_ref().and_then(|s| s.fans.as_ref()) {
        for (index, fan) in fans.fans.iter().enumerate() {
            lines.push(Line::from(format!(
                "Fan {}: {:.0}/{:.0} rpm",
                index + 1,
                fan.actual.value(),
                fan.maximum.value()
            )));
        }
    }
    frame.render_widget(
        Paragraph::new(lines).style(Style::default().fg(accent())).block(panel("Network · Disk · Fans")),
        area,
    );
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
    frame.render_widget(List::new(items).style(Style::default().fg(accent())).block(panel("Process List")), area);
}

/// A square-bordered, green panel with a title (matches mactop's inner widgets).
fn panel(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(accent()))
        .title_style(Style::default().fg(accent()))
        .title(title)
}

fn panel_owned(title: String) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(accent()))
        .title_style(Style::default().fg(accent()))
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

fn push_history(
    history: &mut VecDeque<f64>,
    value: f64,
) {
    history.push_back(value);
    while history.len() > HISTORY {
        history.pop_front();
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
