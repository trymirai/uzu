use keisoku::Device;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::Line,
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph},
};

use crate::{
    chart::render_chart,
    format::{battery_status, format_uptime, human_bytes},
    info::render_info,
    state::{accent, background, interval_ms, show_info, theme},
    telemetry::Telemetry,
    widgets::{panel, split_horizontal, split_vertical},
};

pub(crate) fn draw(
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
        .title_bottom(Line::from(format!(" -/+ {}ms ", interval_ms())).right_aligned());
    let area = outer.inner(frame.area());
    frame.render_widget(outer, frame.area());

    if show_info() {
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

fn cpu_title(state: &Telemetry) -> String {
    let cpu = state.snapshot.as_ref().and_then(|s| s.cpu.as_ref());
    let temperature =
        state.snapshot.as_ref().and_then(|s| s.temperatures.as_ref()).and_then(|t| t.cpu_average).map(|c| c.value());
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
    let temperature =
        state.snapshot.as_ref().and_then(|s| s.temperatures.as_ref()).and_then(|t| t.gpu_average).map(|c| c.value());
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
    let Some(snapshot) = state.snapshot.as_ref() else {
        return "ANE Usage".to_string();
    };
    let Some(ane) = snapshot.neural_engine.as_ref() else {
        return "ANE Usage".to_string();
    };
    let watts = snapshot.power.as_ref().map_or(0.0, |power| power.ane.value());
    format!("ANE Usage: {:.2}% @ {:.2} W", ane.active.value(), watts)
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
        lines.push(Line::from(format!("{:<6}{:>6.2} W", "Total:", power.total().value())));
        lines.push(Line::from(format!("{:<6}{:>6.2} W", "Pkg:", power.package.value())));
    }
    let thermals =
        state.snapshot.as_ref().and_then(|s| s.thermal_pressure).map(|t| format!("{t:?}")).unwrap_or_default();
    lines.push(Line::from(format!("Thermals: {thermals}")));
    lines.push(Line::from(format!("Uptime: {}", format_uptime(state.uptime_seconds))));
    if let Some(battery) = state.snapshot.as_ref().and_then(|s| s.battery.as_ref()).filter(|b| b.present) {
        let status = battery_status(battery);
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

fn header_title(device: Option<&Device>) -> String {
    match device {
        Some(device) => format!(
            " kanshi  •  {}  •  {}C ({}E+{}P)  •  {} GPU  •  {:.0} GB ",
            device.chip,
            device.efficiency_cores + device.performance_cores,
            device.efficiency_cores,
            device.performance_cores,
            device.gpu_cores,
            device.ram_total.value() as f64 / 1e9,
        ),
        None => " kanshi ".to_string(),
    }
}
