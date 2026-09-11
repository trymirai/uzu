mod file_logger;
mod log_file;
mod request_trace;

use std::io;

pub use file_logger::FileLogger;
use tracing_subscriber::{EnvFilter, fmt::writer::MakeWriterExt, layer::SubscriberExt, util::SubscriberInitExt};

use self::request_trace::{RequestContextLayer, RequestFormatter};

pub fn init(
    stdout_max_level: tracing::Level,
    file_logger: Option<FileLogger>,
    file_max_level: tracing::Level,
) -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::new("off,cli=debug,hanashi=debug,nagare::chat::token=debug"))
        .with(RequestContextLayer);
    let format = tracing_subscriber::fmt::layer().event_format(RequestFormatter).with_ansi(false);

    let console = io::stdout.with_max_level(stdout_max_level);
    let result = if let Some(file_logger) = file_logger {
        subscriber.with(format.with_writer(console.and(file_logger.with_max_level(file_max_level)))).try_init()
    } else {
        subscriber.with(format.with_writer(console)).try_init()
    };
    result.map_err(|error| anyhow::anyhow!("Failed to initialize server tracing: {error}"))
}
