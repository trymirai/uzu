mod file_logger;
mod log_file;

use std::io;

pub use file_logger::FileLogger;
use tracing_subscriber::{EnvFilter, fmt::writer::MakeWriterExt};

pub fn init(
    stdout_max_level: tracing::Level,
    file_logger: Option<FileLogger>,
    file_max_level: tracing::Level,
) -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("off,cli=trace"))
        .without_time()
        .with_level(false)
        .with_target(false)
        .with_ansi(false);

    let console = io::stdout.with_max_level(stdout_max_level);
    let result = if let Some(file_logger) = file_logger {
        subscriber.with_writer(console.and(file_logger.with_max_level(file_max_level))).try_init()
    } else {
        subscriber.with_writer(console).try_init()
    };
    result.map_err(|error| anyhow::anyhow!("Failed to initialize server tracing: {error}"))
}
