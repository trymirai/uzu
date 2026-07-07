#![cfg(target_os = "macos")]

use std::time::Duration;

use keisoku::{
    Chip, CpuUsage, GpuUsage, Instant, Interval, Memory, NeuralEngine, Power, RamTotal, Static, TemperatureSensors,
    Temps,
};
use ratatui::{
    Terminal,
    backend::TestBackend,
    layout::Constraint,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Row, Table},
};

#[test]
fn render_table() {
    let (chip, ram_total) = Static::<(Chip, RamTotal)>::new().into_inner();

    let mut soc = Interval::<(CpuUsage, GpuUsage, NeuralEngine, Power)>::new();
    let session = soc.begin();
    std::thread::sleep(Duration::from_millis(300));
    let (cpu, gpu, neural_engine, power) = soc.end(session);

    let mut gauges = Instant::<(Memory, Temps, TemperatureSensors)>::new();
    let (memory, temperatures, sensors) = gauges.read();

    let na = "—";
    let mut lines: Vec<[String; 3]> = vec![
        ["CPU".into(), format!("{:.1} %", cpu.usage.value()), String::new()],
        ["GPU".into(), format!("{:.0} %", gpu.usage.value()), String::new()],
        ["ANE".into(), format!("{:.0} %", neural_engine.active.value()), String::new()],
        ["Power".into(), format!("{:.2} W", power.package.value()), String::new()],
        [
            "Memory".into(),
            memory.as_ref().map_or(na.into(), |memory| {
                format!(
                    "{:.1} / {:.1} GB",
                    memory.ram_usage.value() as f64 / 1e9,
                    memory.ram_total.value() as f64 / 1e9
                )
            }),
            String::new(),
        ],
        [
            "CPU temp".into(),
            temperatures.as_ref().and_then(|t| t.cpu_average).map_or(na.into(), |c| format!("{:.1} °C", c.value())),
            String::new(),
        ],
        [
            "GPU temp".into(),
            temperatures.as_ref().and_then(|t| t.gpu_average).map_or(na.into(), |g| format!("{:.1} °C", g.value())),
            String::new(),
        ],
        ["────────".into(), "──────".into(), "──────────".into()],
    ];
    for sensor in sensors.iter() {
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
            chip,
            ram_total.value() as f64 / 1e9
        )));

    let mut terminal = Terminal::new(TestBackend::new(66, height)).unwrap();
    terminal.draw(|frame| frame.render_widget(table, frame.area())).unwrap();
    println!("\n{}", terminal.backend());
}
