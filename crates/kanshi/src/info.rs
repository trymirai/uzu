use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::{
    format::{battery_status, format_uptime, human_bytes},
    state::accent,
    telemetry::Telemetry,
    widgets::split_horizontal,
};

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

pub(crate) fn render_info(
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
        let mut parts = Vec::new();
        if let Some(cpu) = temperatures.cpu_average {
            parts.push(format!("CPU {:.0}°C", cpu.value()));
        }
        if let Some(gpu) = temperatures.gpu_average {
            parts.push(format!("GPU {:.0}°C", gpu.value()));
        }
        if !parts.is_empty() {
            lines.push(info_line("Thermals", parts.join("  ")));
        }
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
        let status = battery_status(battery);
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
