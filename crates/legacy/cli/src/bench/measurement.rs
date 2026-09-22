use std::io::{self, BufRead, Write};

use anyhow::{Result, ensure};

pub fn synchronize(
    event: &str,
    command: &str,
) -> Result<()> {
    exchange(&mut io::stdin().lock(), &mut io::stdout().lock(), event, command)
}

fn exchange(
    input: &mut impl BufRead,
    output: &mut impl Write,
    event: &str,
    command: &str,
) -> Result<()> {
    writeln!(output, "\n{}", serde_json::json!({"benchmark_event": event}))?;
    output.flush()?;
    let mut line = String::new();
    ensure!(input.read_line(&mut line)? > 0, "Measurement controller disconnected");
    ensure!(line == format!("{command}\n"), "Unexpected measurement controller command");
    Ok(())
}

#[cfg(test)]
#[path = "../../unit/bench/measurement_test.rs"]
mod tests;
