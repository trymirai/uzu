use std::fmt::Write;

use chrono::{DateTime, Local, TimeZone};

/// Formats the current local time according to the provided `strftime` pattern
pub fn strftime_now(format_string: String) -> String {
    format_time(&Local::now(), &format_string)
}

fn format_time<Tz: TimeZone>(
    time: &DateTime<Tz>,
    format_string: &str,
) -> String
where
    Tz::Offset: std::fmt::Display,
{
    let mut output = String::new();
    match write!(output, "{}", time.format(format_string)) {
        Ok(()) => output,
        Err(_) => String::new(),
    }
}

#[cfg(test)]
#[path = "../../../../../unit/chat/hanashi/renderer/functions/strftime_now.rs"]
mod tests;
