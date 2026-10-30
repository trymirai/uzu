use std::io::{IsTerminal, stdout};

use crossterm::{
    event::{KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags},
    execute,
};
use iocraft::{ComponentUpdater, Hook};

#[derive(Default)]
pub struct KeyboardEnhancement {
    enabled: bool,
}

impl Hook for KeyboardEnhancement {
    fn post_component_update(
        &mut self,
        updater: &mut ComponentUpdater,
    ) {
        if self.enabled || !stdout().is_terminal() || updater.terminal_events().is_none() {
            return;
        }

        self.enabled = execute!(
            stdout(),
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                    | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
            )
        )
        .is_ok();
    }
}

impl Drop for KeyboardEnhancement {
    fn drop(&mut self) {
        if self.enabled {
            let _ = execute!(stdout(), PopKeyboardEnhancementFlags);
        }
    }
}
