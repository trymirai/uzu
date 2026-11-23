use std::{io, time::Duration};

use crossterm::event::{self, Event, KeyEvent};

pub struct EventHandler {
    last_event: Option<AppEvent>,
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Key(KeyEvent),
    #[allow(dead_code)]
    Tick,
}

impl EventHandler {
    pub fn new() -> Self {
        Self {
            last_event: None,
        }
    }

    pub async fn poll_event(&mut self) -> io::Result<bool> {
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
            // Windows also reports each key's release, which must not act a second time.
            && !key.is_release()
        {
            self.last_event = Some(AppEvent::Key(key));
            return Ok(true);
        }
        Ok(false)
    }

    pub fn next_event(&mut self) -> Option<AppEvent> {
        self.last_event.take()
    }
}
