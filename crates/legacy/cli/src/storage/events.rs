use std::{io, time::Duration};

use crossterm::event::{self, Event, KeyEvent, KeyEventKind};

pub struct EventHandler {
    last_event: Option<AppEvent>,
}

#[derive(Debug, Clone, PartialEq)]
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
        while event::poll(Duration::from_millis(100))? {
            if let Some(event) = app_event(event::read()?) {
                self.last_event = Some(event);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn next_event(&mut self) -> Option<AppEvent> {
        self.last_event.take()
    }
}

fn app_event(event: Event) -> Option<AppEvent> {
    match event {
        Event::Key(key) if key.kind != KeyEventKind::Release => Some(AppEvent::Key(key)),
        _ => None,
    }
}

#[cfg(test)]
#[path = "../../unit/storage/events_test.rs"]
mod tests;
