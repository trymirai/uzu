#![cfg(target_os = "macos")]

mod app;
mod events;
mod models;
mod sections;
mod ui;

use std::{io, sync::Arc};

use clap::{Parser, ValueEnum};
use crossterm::{
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use download_manager::FileDownloadManagerType;
use ratatui::{Terminal, backend::CrosstermBackend};
use uzu::engine::{Engine, EngineConfig};

use crate::{app::App, events::EventHandler};

#[derive(Debug, Clone, Parser)]
struct Cli {
    #[arg(long, value_enum, default_value_t = DownloadManagerCliType::default())]
    download_manager: DownloadManagerCliType,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum DownloadManagerCliType {
    #[default]
    Apple,
    Universal,
}

impl From<DownloadManagerCliType> for FileDownloadManagerType {
    fn from(download_manager_type: DownloadManagerCliType) -> Self {
        match download_manager_type {
            DownloadManagerCliType::Apple => Self::Apple,
            DownloadManagerCliType::Universal => Self::Universal,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    dotenvy::dotenv().ok();
    let runtime = tokio::runtime::Handle::current();
    let engine =
        Arc::new(Engine::new_with_download_manager_type(EngineConfig::default(), cli.download_manager.into()).await?);

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and event handler
    let app = App::new(engine.clone(), runtime).await;
    let event_handler = EventHandler::new();

    // Run app
    let res = run_app(&mut terminal, app, event_handler).await;

    // Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        tracing::error!("Application error: {:?}", e);
    }

    Ok(())
}

async fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
    mut event_handler: EventHandler,
) -> io::Result<()> {
    // Spawn model state listener
    app.spawn_state_listener().await;

    loop {
        terminal
            .draw(|f| ui::draw(f, &mut app))
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error.to_string()))?;

        if app.should_quit {
            break;
        }

        // Handle events
        if event_handler.poll_event().await? {
            if let Some(event) = event_handler.next_event() {
                app.handle_event(event).await;
            }
        }

        // Small delay to reduce CPU usage
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    Ok(())
}
