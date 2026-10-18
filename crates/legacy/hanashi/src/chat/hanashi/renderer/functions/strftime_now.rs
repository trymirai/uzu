use minijinja::{Error, ErrorKind};

/// Formats the current local time according to the provided `strftime` pattern
pub fn strftime_now(format_string: String) -> Result<String, Error> {
    let mut output = String::new();
    chrono::Local::now().format(&format_string).write_to(&mut output).map_err(|error| {
        Error::new(ErrorKind::InvalidOperation, format!("invalid strftime format: {format_string:?}"))
            .with_source(error)
    })?;
    Ok(output)
}
