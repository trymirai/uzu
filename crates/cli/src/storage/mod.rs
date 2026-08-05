mod app;
mod events;
mod models;
mod sections;
mod ui;

use std::{io, sync::Arc};

use anyhow::Result;
use clap::ValueEnum;
use crossterm::{
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use uzu::engine::{DownloadManagerType, Engine, EngineConfig};

use self::{app::App, events::EventHandler};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum DownloadManagerCliType {
    #[cfg(target_vendor = "apple")]
    Native,
    Universal,
}

impl Default for DownloadManagerCliType {
    fn default() -> Self {
        DownloadManagerType::default().into()
    }
}

impl From<DownloadManagerType> for DownloadManagerCliType {
    fn from(download_manager_type: DownloadManagerType) -> Self {
        match download_manager_type {
            #[cfg(target_vendor = "apple")]
            DownloadManagerType::Native => Self::Native,
            DownloadManagerType::Universal => Self::Universal,
        }
    }
}

impl From<DownloadManagerCliType> for DownloadManagerType {
    fn from(download_manager_type: DownloadManagerCliType) -> Self {
        match download_manager_type {
            #[cfg(target_vendor = "apple")]
            DownloadManagerCliType::Native => Self::Native,
            DownloadManagerCliType::Universal => Self::Universal,
        }
    }
}

pub(crate) async fn run(download_manager: DownloadManagerCliType) -> Result<()> {
    dotenvy::dotenv().ok();
    let runtime = tokio::runtime::Handle::current();
    let config = EngineConfig::default().with_download_manager_type(download_manager.into());
    let engine = Arc::new(Engine::new(config).await?);

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

    if let Err(error) = res {
        tracing::error!("Application error: {error:?}");
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
        terminal.draw(|f| ui::draw(f, &mut app)).map_err(|error| io::Error::other(error.to_string()))?;

        if app.should_quit {
            break;
        }

        // Handle events
        if event_handler.poll_event().await?
            && let Some(event) = event_handler.next_event()
        {
            app.handle_event(event).await;
        }

        // Small delay to reduce CPU usage
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    Ok(())
}
