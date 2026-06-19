use std::{
    collections::{HashSet, VecDeque},
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use keisoku::{Collector, Device, Snapshot};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph},
};
use sysinfo::{Disks, Networks, ProcessRefreshKind, ProcessesToUpdate, System};

const HISTORY: usize = 256;
const PROCESS_ROWS: usize = 8;
const NETWORK_ROWS: usize = 4;

const MIN_INTERVAL_MS: u64 = 100;
const MAX_INTERVAL_MS: u64 = 5000;
const INTERVAL_STEP_MS: u64 = 100;
static INTERVAL_MS: AtomicU64 = AtomicU64::new(1000);

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
static DARK_BACKGROUND: AtomicBool = AtomicBool::new(true);

static SHOW_INFO: AtomicBool = AtomicBool::new(false);

static DATA_VERSION: AtomicU64 = AtomicU64::new(0);

const DISK_REFRESH: Duration = Duration::from_secs(3);

const APPLE_ART: [&str; 17] = [
    "                    'c.        ",
    "                 ,xNMM.        ",
    "               .OMMMMo         ",
    "               OMMM0,          ",
    "     .;loddo:' loolloddol;.    ",
    "   cKMMMMMMMMMMNWMMMMMMMMMM0:  ",
    " .KMMMMMMMMMMMMMMMMMMMMMMMWd.  ",
    " XMMMMMMMMMMMMMMMMMMMMMMMX.    ",
    ";MMMMMMMMMMMMMMMMMMMMMMMM:     ",
    ":MMMMMMMMMMMMMMMMMMMMMMMM:     ",
    ".MMMMMMMMMMMMMMMMMMMMMMMMX.    ",
    " kMMMMMMMMMMMMMMMMMMMMMMMMWd.  ",
    " .XMMMMMMMMMMMMMMMMMMMMMMMMMMk ",
    "  .XMMMMMMMMMMMMMMMMMMMMMMMMK. ",
    "    kMMMMMMMMMMMMMMMMMMMMMMd   ",
    "     ;KMMMMMMMWXXWMMMMMMMk.    ",
    "       .cooc,.    .,coo:.      ",
];

fn theme() -> (&'static str, Color) {
    THEMES[THEME_INDEX.load(Ordering::Relaxed) % THEMES.len()]
}

fn accent() -> Color {
    theme().1
}

fn background() -> Color {
    if DARK_BACKGROUND.load(Ordering::Relaxed) {
        Color::Black
    } else {
        Color::Reset
    }
}

fn interval() -> Duration {
    Duration::from_millis(INTERVAL_MS.load(Ordering::Relaxed))
}

fn adjust_interval(delta: i64) {
    let current = INTERVAL_MS.load(Ordering::Relaxed) as i64;
    let next = (current + delta).clamp(MIN_INTERVAL_MS as i64, MAX_INTERVAL_MS as i64) as u64;
    INTERVAL_MS.store(next, Ordering::Relaxed);
}

struct ProcessRow {
    cpu: f32,
    memory: u64,
    name: String,
}

struct DiskRow {
    name: String,
    used: u64,
    total: u64,
}

struct NetInterface {
    name: String,
    down: f64,
    up: f64,
}

struct HostInfo {
    user: String,
    hostname: String,
    os: String,
    kernel: String,
    shell: String,
}

#[derive(Default)]
struct Telemetry {
    device: Option<Device>,
    host: Option<HostInfo>,
    snapshot: Option<Snapshot>,
    cpu_history: VecDeque<f64>,
    gpu_history: VecDeque<f64>,
    ane_history: VecDeque<f64>,
    memory_history: VecDeque<f64>,
    power_history: VecDeque<f64>,
    max_power: f64,
    uptime_seconds: u64,
    network_down: f64,
    network_up: f64,
    network_packets_down: f64,
    network_packets_up: f64,
    network_total_down: u64,
    network_total_up: u64,
    network_interfaces: Vec<NetInterface>,
    disk_read: f64,
    disk_written: f64,
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

    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &telemetry);
    ratatui::restore();

    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    result
}

fn sample_loop(
    telemetry: &Arc<Mutex<Telemetry>>,
    stop: &Arc<AtomicBool>,
) {
    let mut collector = Collector::new();
    let device = collector.device();
    let host = HostInfo {
        user: std::env::var("USER").unwrap_or_else(|_| "user".into()),
        hostname: System::host_name().unwrap_or_default(),
        os: System::long_os_version().unwrap_or_default(),
        kernel: System::kernel_version().unwrap_or_default(),
        shell: std::env::var("SHELL")
            .ok()
            .and_then(|path| path.rsplit('/').next().map(str::to_owned))
            .unwrap_or_default(),
    };
    if let Ok(mut state) = telemetry.lock() {
        state.device = Some(device);
        state.host = Some(host);
    }

    let mut system = System::new();
    let mut networks = Networks::new_with_refreshed_list();
    let mut all_disks = Disks::new_with_refreshed_list();
    let mut last_refresh = Instant::now();
    let mut last_disk_refresh: Option<Instant> = None;

    let process_kind = ProcessRefreshKind::nothing().with_cpu().with_memory().with_disk_usage();

    while !stop.load(Ordering::Relaxed) {
        let snapshot = collector.sample(interval());

        system.refresh_processes_specifics(ProcessesToUpdate::All, true, process_kind);
        networks.refresh(true);
        let elapsed = last_refresh.elapsed().as_secs_f64().max(0.001);
        last_refresh = Instant::now();

        let (mut received, mut transmitted) = (0u64, 0u64);
        let (mut packets_in, mut packets_out) = (0u64, 0u64);
        let (mut total_in, mut total_out) = (0u64, 0u64);
        let mut interfaces: Vec<NetInterface> = Vec::new();
        for (name, data) in networks.iter() {
            received += data.received();
            transmitted += data.transmitted();
            packets_in += data.packets_received();
            packets_out += data.packets_transmitted();
            total_in += data.total_received();
            total_out += data.total_transmitted();

            if data.total_received() > 0 || data.total_transmitted() > 0 {
                interfaces.push(NetInterface {
                    name: name.clone(),
                    down: data.received() as f64 / elapsed,
                    up: data.transmitted() as f64 / elapsed,
                });
            }
        }
        interfaces.sort_by(|a, b| (b.down + b.up).partial_cmp(&(a.down + a.up)).unwrap_or(std::cmp::Ordering::Equal));
        interfaces.truncate(NETWORK_ROWS);

        let disks = last_disk_refresh.is_none_or(|at| at.elapsed() >= DISK_REFRESH).then(|| {
            last_disk_refresh = Some(Instant::now());
            all_disks.refresh(true);
            let mut seen_disks = HashSet::new();
            all_disks
                .list()
                .iter()
                .filter(|disk| seen_disks.insert(disk.name().to_os_string()))
                .map(|disk| DiskRow {
                    name: disk.name().to_string_lossy().into_owned(),
                    used: disk.total_space().saturating_sub(disk.available_space()),
                    total: disk.total_space(),
                })
                .collect::<Vec<_>>()
        });

        let (mut disk_read, mut disk_written) = (0u64, 0u64);
        for process in system.processes().values() {
            let io = process.disk_usage();
            disk_read += io.read_bytes;
            disk_written += io.written_bytes;
        }

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

        let cpu = snapshot.cpu.as_ref().map(|c| c.usage.value() as f64).unwrap_or(0.0);
        let gpu = snapshot.gpu.as_ref().map(|g| g.usage.value() as f64).unwrap_or(0.0);
        let ane = snapshot.neural_engine.as_ref().map(|a| a.active.value() as f64).unwrap_or(0.0);
        let memory = snapshot
            .memory
            .as_ref()
            .filter(|m| m.ram_total.value() > 0)
            .map(|m| m.ram_usage.value() as f64 / m.ram_total.value() as f64 * 100.0)
            .unwrap_or(0.0);

        let power = snapshot.power.as_ref().map(|p| p.package.value() as f64).unwrap_or(0.0);

        if let Ok(mut state) = telemetry.lock() {
            push_history(&mut state.cpu_history, cpu);
            push_history(&mut state.gpu_history, gpu);
            push_history(&mut state.ane_history, ane);
            push_history(&mut state.memory_history, memory);
            push_history(&mut state.power_history, power);
            state.max_power = state.max_power.max(power);
            state.snapshot = Some(snapshot);
            state.uptime_seconds = System::uptime();
            state.network_down = received as f64 / elapsed;
            state.network_up = transmitted as f64 / elapsed;
            state.network_packets_down = packets_in as f64 / elapsed;
            state.network_packets_up = packets_out as f64 / elapsed;
            state.network_total_down = total_in;
            state.network_total_up = total_out;
            state.network_interfaces = interfaces;
            state.disk_read = disk_read as f64 / elapsed;
            state.disk_written = disk_written as f64 / elapsed;
            if let Some(disks) = disks {
                state.disks = disks;
            }
            state.processes = processes;
        }
        DATA_VERSION.fetch_add(1, Ordering::Relaxed);
    }
}

fn run(
    terminal: &mut DefaultTerminal,
    telemetry: &Arc<Mutex<Telemetry>>,
) -> io::Result<()> {
    let mut shown_version = u64::MAX;
    let mut redraw = true;
    loop {
        let version = DATA_VERSION.load(Ordering::Relaxed);
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
                        KeyCode::Char('c') => {
                            THEME_INDEX.fetch_add(1, Ordering::Relaxed);
                        },
                        KeyCode::Char('b') => {
                            DARK_BACKGROUND.fetch_xor(true, Ordering::Relaxed);
                        },
                        KeyCode::Char('i') => {
                            SHOW_INFO.fetch_xor(true, Ordering::Relaxed);
                        },
                        KeyCode::Char('+') | KeyCode::Char('=') => adjust_interval(-(INTERVAL_STEP_MS as i64)),
                        KeyCode::Char('-') | KeyCode::Char('_') => adjust_interval(INTERVAL_STEP_MS as i64),
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
        .title_bottom(Line::from(" Info: i · Color: c · BG: b · Exit: q ").centered())
        .title_bottom(Line::from(format!(" -/+ {}ms ", INTERVAL_MS.load(Ordering::Relaxed))).right_aligned());
    let area = outer.inner(frame.area());
    frame.render_widget(outer, frame.area());

    if SHOW_INFO.load(Ordering::Relaxed) {
        render_info(frame, area, state);
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(50), Constraint::Percentage(25)])
        .split(area);

    let top = split_horizontal(rows[0], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_chart(frame, top[0], cpu_title(state), &state.cpu_history, 100.0);
    render_chart(frame, top[1], gpu_title(state), &state.gpu_history, 100.0);

    let middle = split_horizontal(rows[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);

    let left = split_vertical(middle[0], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_chart(frame, left[0], ane_title(state), &state.ane_history, 100.0);
    let left_bottom = split_horizontal(left[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_power(frame, left_bottom[0], state);
    render_chart(frame, left_bottom[1], power_title(state), &state.power_history, state.max_power);

    let right = split_vertical(middle[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_chart(frame, right[0], memory_title(state), &state.memory_history, 100.0);
    let right_bottom = split_horizontal(right[1], [Constraint::Percentage(50), Constraint::Percentage(50)]);
    render_model(frame, right_bottom[0], state);
    render_fans(frame, right_bottom[1], state);

    let bottom = split_horizontal(
        rows[2],
        [
            Constraint::Percentage(40),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ],
    );
    render_processes(frame, bottom[0], state);
    render_io(frame, bottom[1], state);
    render_disk(frame, bottom[2], state);
    render_network(frame, bottom[3], state);
}

fn render_info(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let art_width = APPLE_ART.iter().map(|line| line.chars().count()).max().unwrap_or(0) as u16 + 2;
    let columns = split_horizontal(area, [Constraint::Length(art_width), Constraint::Min(0)]);

    let art: Vec<Line> = APPLE_ART.iter().map(|line| Line::from(*line)).collect();
    frame.render_widget(Paragraph::new(art).style(Style::default().fg(accent())), columns[0]);
    frame.render_widget(Paragraph::new(build_info_lines(state)).style(Style::default().fg(accent())), columns[1]);
}

fn info_line(
    label: &str,
    value: String,
) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<11}"), Style::default().fg(accent()).add_modifier(Modifier::BOLD)),
        Span::raw(value),
    ])
}

fn build_info_lines(state: &Telemetry) -> Vec<Line<'static>> {
    let snapshot = state.snapshot.as_ref();
    let mut lines = Vec::new();

    if let Some(host) = &state.host {
        lines.push(Line::from(Span::styled(
            format!("{}@{}", host.user, host.hostname),
            Style::default().fg(accent()).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from("-".repeat(25)));
        lines.push(info_line("OS", host.os.clone()));
        if let Some(device) = &state.device {
            lines.push(info_line("Host", device.chip.clone()));
        }
        lines.push(info_line("Kernel", host.kernel.clone()));
        lines.push(info_line("Uptime", format_uptime(state.uptime_seconds)));
        lines.push(info_line("Shell", host.shell.clone()));
    }
    if let Some(device) = &state.device {
        lines.push(info_line(
            "CPU",
            format!("{} ({}E+{}P)", device.chip, device.efficiency_cores, device.performance_cores),
        ));
        lines.push(info_line("GPU", format!("{} Cores", device.gpu_cores)));
    }
    if let Some(memory) = snapshot.and_then(|s| s.memory.as_ref()) {
        lines.push(info_line(
            "Memory",
            format!(
                "{:.2} GB / {:.2} GB",
                memory.ram_usage.value() as f64 / 1e9,
                memory.ram_total.value() as f64 / 1e9
            ),
        ));
        lines.push(info_line(
            "Swap",
            format!(
                "{:.2} GB / {:.2} GB",
                memory.swap_usage.value() as f64 / 1e9,
                memory.swap_total.value() as f64 / 1e9
            ),
        ));
    }

    lines.push(Line::from(""));
    if let Some(cpu) = snapshot.and_then(|s| s.cpu.as_ref()) {
        lines.push(info_line("CPU Usage", format!("{:.2}%", cpu.usage.value())));
    }
    if let Some(gpu) = snapshot.and_then(|s| s.gpu.as_ref()) {
        lines.push(info_line("GPU Usage", format!("{:.0}%", gpu.usage.value())));
    }
    if let Some(ane) = snapshot.and_then(|s| s.neural_engine.as_ref()) {
        lines.push(info_line("ANE Usage", format!("{:.0}%", ane.active.value())));
    }
    if let Some(power) = snapshot.and_then(|s| s.power.as_ref()) {
        lines.push(info_line("Power", format!("{:.2} W (max {:.2} W)", power.package.value(), state.max_power)));
    }
    if let Some(temperatures) = snapshot.and_then(|s| s.temperatures.as_ref()) {
        lines.push(info_line(
            "Thermals",
            format!("CPU {:.0}°C  GPU {:.0}°C", temperatures.cpu_average.value(), temperatures.gpu_average.value()),
        ));
    }
    lines.push(info_line(
        "Network",
        format!("↑ {}/s  ↓ {}/s", human_bytes(state.network_up as u64), human_bytes(state.network_down as u64)),
    ));
    lines.push(info_line(
        "Disk",
        format!("R {}/s  W {}/s", human_bytes(state.disk_read as u64), human_bytes(state.disk_written as u64)),
    ));
    if let Some(bandwidth) = snapshot.and_then(|s| s.bandwidth.as_ref()) {
        lines.push(info_line(
            "DRAM BW",
            format!(
                "R {:.1}  W {:.1}  ({:.1}) GB/s",
                bandwidth.dram_read.value(),
                bandwidth.dram_write.value(),
                bandwidth.dram_read.value() + bandwidth.dram_write.value(),
            ),
        ));
    }
    if let Some(battery) = snapshot.and_then(|s| s.battery.as_ref()).filter(|battery| battery.present) {
        let status = if battery.charging {
            "charging"
        } else if battery.on_ac_power {
            "AC"
        } else {
            "battery"
        };
        lines.push(info_line("Battery", format!("{:.0}% ({status})", battery.percent.value())));
    }

    if let Some(fans) = snapshot.and_then(|s| s.fans.as_ref()).filter(|fans| !fans.fans.is_empty()) {
        lines.push(Line::from(""));
        for (index, fan) in fans.fans.iter().enumerate() {
            lines.push(info_line(
                &format!("Fan {}", index + 1),
                format!("{:.0} rpm ({:.0}–{:.0})", fan.actual.value(), fan.minimum.value(), fan.maximum.value()),
            ));
        }
    }

    if !state.disks.is_empty() {
        lines.push(Line::from("-".repeat(25)));
        for disk in &state.disks {
            lines.push(info_line(&disk.name, format!("{} / {}", human_bytes(disk.used), human_bytes(disk.total))));
        }
    }

    lines
}

const BRAILLE_DOTS: [[u16; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];

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
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let columns = inner.width as usize;
    let shown: Vec<f64> = history.iter().rev().take(columns).rev().copied().collect();
    if shown.is_empty() {
        return;
    }
    let offset = columns - shown.len();
    let ceiling = ceiling.max(1e-9);
    let cell_rows = inner.height as usize;
    let sub_rows = cell_rows * 4;
    let last = shown.len() - 1;
    let color = accent();
    let background = background();

    let fill_at = |sub_x: usize| -> Option<usize> {
        let position = sub_x as f64 / 2.0 - offset as f64;
        if position < 0.0 {
            return None;
        }
        let low = (position.floor() as usize).min(last);
        let high = (low + 1).min(last);
        let blend = position - low as f64;
        let value = shown[low] * (1.0 - blend) + shown[high] * blend;
        Some(((value / ceiling).clamp(0.0, 1.0) * sub_rows as f64).round() as usize)
    };

    let buffer = frame.buffer_mut();
    for cell_x in 0..columns {
        let fills = [fill_at(cell_x * 2), fill_at(cell_x * 2 + 1)];
        if fills.iter().all(Option::is_none) {
            continue;
        }
        let x = inner.left() + cell_x as u16;
        for row in 0..cell_rows {
            let mut dots = 0u16;
            for (sub, fill) in fills.iter().enumerate() {
                let Some(fill) = fill else {
                    continue;
                };
                for internal in 0..4 {
                    if row * 4 + (3 - internal) < *fill {
                        dots |= BRAILLE_DOTS[sub][internal];
                    }
                }
            }
            if dots == 0 {
                continue;
            }
            let symbol = char::from_u32(0x2800 + dots as u32).unwrap_or(' ');
            let y = inner.bottom() - 1 - row as u16;
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.set_char(symbol).set_fg(color).set_bg(background);
            }
        }
    }
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

fn render_fans(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let mut lines = Vec::new();
    if let Some(fans) = state.snapshot.as_ref().and_then(|s| s.fans.as_ref()) {
        for (index, fan) in fans.fans.iter().enumerate() {
            lines.push(Line::from(format!("Fan {}:  {:.0} rpm", index + 1, fan.actual.value())));
            lines.push(Line::from(format!(
                "  range {:.0}–{:.0} · target {:.0}",
                fan.minimum.value(),
                fan.maximum.value(),
                fan.target.value(),
            )));
        }
    }
    if lines.is_empty() {
        lines.push(Line::from("no fans"));
    }
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(accent())).block(panel("Fans")), area);
}

fn render_io(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let lines = vec![
        Line::from(format!("R {}/s", human_bytes(state.disk_read as u64))),
        Line::from(format!("W {}/s", human_bytes(state.disk_written as u64))),
    ];
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(accent())).block(panel("IO")), area);
}

fn render_disk(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let lines: Vec<Line> = state
        .disks
        .iter()
        .map(|disk| Line::from(format!("{}: {} / {}", disk.name, human_bytes(disk.used), human_bytes(disk.total))))
        .collect();
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(accent())).block(panel("Disk")), area);
}

fn render_network(
    frame: &mut Frame,
    area: Rect,
    state: &Telemetry,
) {
    let mut lines = vec![
        Line::from(format!(
            "{:<6}↑ {}/s  ↓ {}/s",
            "Rate",
            human_bytes(state.network_up as u64),
            human_bytes(state.network_down as u64),
        )),
        Line::from(format!("{:<6}↑ {:.0}/s  ↓ {:.0}/s", "Pkts", state.network_packets_up, state.network_packets_down,)),
        Line::from(format!(
            "{:<6}↑ {}  ↓ {}",
            "Total",
            human_bytes(state.network_total_up),
            human_bytes(state.network_total_down),
        )),
    ];
    for iface in &state.network_interfaces {
        lines.push(Line::from(format!(
            "{:<6}↑ {}/s ↓ {}/s",
            iface.name,
            human_bytes(iface.up as u64),
            human_bytes(iface.down as u64),
        )));
    }
    frame.render_widget(Paragraph::new(lines).style(Style::default().fg(accent())).block(panel("Network")), area);
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
