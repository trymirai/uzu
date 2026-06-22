use std::time::Duration;

use keisoku::Collector;
use ratatui::{
    Terminal,
    backend::TestBackend,
    layout::Constraint,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Row, Table},
};

#[test]
fn render_table() {
    let mut collector = Collector::new();
    let device = collector.device();
    let snapshot = collector.sample(Duration::from_millis(300));

    let na = "—";
    let mut lines: Vec<[String; 3]> = vec![
        ["CPU".into(), snapshot.cpu.as_ref().map_or(na.into(), |c| format!("{:.1} %", c.usage.value())), String::new()],
        ["GPU".into(), snapshot.gpu.as_ref().map_or(na.into(), |g| format!("{:.0} %", g.usage.value())), String::new()],
        [
            "ANE".into(),
            snapshot.neural_engine.as_ref().map_or(na.into(), |a| format!("{:.0} %", a.active.value())),
            String::new(),
        ],
        [
            "Power".into(),
            snapshot.power.as_ref().map_or(na.into(), |p| format!("{:.2} W", p.package.value())),
            String::new(),
        ],
        [
            "Memory".into(),
            snapshot.memory.as_ref().map_or(na.into(), |m| {
                format!("{:.1} / {:.1} GB", m.ram_usage.value() as f64 / 1e9, m.ram_total.value() as f64 / 1e9)
            }),
            String::new(),
        ],
        [
            "CPU temp".into(),
            snapshot
                .temperatures
                .as_ref()
                .and_then(|t| t.cpu_average)
                .map_or(na.into(), |c| format!("{:.1} °C", c.value())),
            String::new(),
        ],
        [
            "GPU temp".into(),
            snapshot
                .temperatures
                .as_ref()
                .and_then(|t| t.gpu_average)
                .map_or(na.into(), |g| format!("{:.1} °C", g.value())),
            String::new(),
        ],
        ["────────".into(), "──────".into(), "──────────".into()],
    ];
    for sensor in &snapshot.sensors {
        lines.push([sensor.name.clone(), format!("{:.1}", sensor.value), sensor.component.to_string()]);
    }

    let accent = Style::default().fg(Color::Green);
    let height = (lines.len() + 4) as u16;
    let header = Row::new(["Metric", "Value", "Detail"]).style(accent.add_modifier(Modifier::BOLD));
    let rows: Vec<Row> = lines.into_iter().map(Row::new).collect();
    let table = Table::new(rows, [Constraint::Length(20), Constraint::Length(16), Constraint::Min(22)])
        .header(header)
        .block(Block::default().borders(Borders::ALL).border_style(accent).title(format!(
            " kanshi · {} · {:.0} GB ",
            device.chip,
            device.ram_total.value() as f64 / 1e9
        )));

    let mut terminal = Terminal::new(TestBackend::new(66, height)).unwrap();
    terminal.draw(|frame| frame.render_widget(table, frame.area())).unwrap();
    println!("\n{}", terminal.backend());
}
