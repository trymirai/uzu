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
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::DefaultTerminal;

use crate::{
    panels::draw,
    sampler::sample_loop,
    state::{cycle_theme, data_version, slow_down, speed_up, toggle_background, toggle_info},
    telemetry::Telemetry,
};

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
