use crossterm::event::{KeyCode, KeyEventState, KeyModifiers};

use super::*;

fn key(kind: KeyEventKind) -> KeyEvent {
    KeyEvent {
        code: KeyCode::Char('d'),
        modifiers: KeyModifiers::NONE,
        kind,
        state: KeyEventState::NONE,
    }
}

/// A key acts once: on its press and on each repeat, never on the release Windows reports after it.
#[test]
fn only_presses_and_repeats_become_events() {
    for kind in [KeyEventKind::Press, KeyEventKind::Repeat] {
        assert_eq!(app_event(Event::Key(key(kind))), Some(AppEvent::Key(key(kind))));
    }
    assert_eq!(app_event(Event::Key(key(KeyEventKind::Release))), None);
    assert_eq!(app_event(Event::Resize(80, 24)), None);
}
