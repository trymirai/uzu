use std::io::{self, IsTerminal};

use anyhow::{Result, bail};
use crossterm::{
    cursor::Show,
    execute,
    terminal::{LeaveAlternateScreen, disable_raw_mode},
};

pub fn require_terminal() -> Result<()> {
    if !io::stdout().is_terminal() {
        bail!("stdout is not a terminal");
    }
    Ok(())
}

pub fn restore_terminal_on_panic() {
    let print_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        print_panic(info);
        std::process::exit(101);
    }));
}

pub fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, Show)
}
