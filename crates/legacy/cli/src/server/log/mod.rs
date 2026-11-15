mod file_logger;
mod log_file;
mod request_logging_fairing;
mod request_trace;
mod response_body_logger;

use std::io;

pub use file_logger::FileLogger;
pub use request_logging_fairing::RequestLoggingFairing;
use tracing_subscriber::{EnvFilter, fmt::writer::MakeWriterExt, layer::SubscriberExt, util::SubscriberInitExt};

use self::request_trace::{RequestContextLayer, RequestFormatter};

pub fn init(
    stdout_max_level: tracing::Level,
    file_logger: Option<FileLogger>,
    file_max_level: tracing::Level,
) -> anyhow::Result<()> {
    // Colored console output.
    let console_layer = tracing_subscriber::fmt::layer()
        .event_format(RequestFormatter::colored())
        .with_writer(io::stdout.with_max_level(stdout_max_level));

    let subscriber = tracing_subscriber::registry()
        .with(EnvFilter::new("off,cli=debug,hanashi=debug,nagare::chat::token=debug"))
        .with(RequestContextLayer)
        .with(console_layer);

    let result = if let Some(file_logger) = file_logger {
        // Plain (ANSI-free) file output.
        subscriber
            .with(
                tracing_subscriber::fmt::layer()
                    .event_format(RequestFormatter::plain())
                    .with_writer(file_logger.with_max_level(file_max_level)),
            )
            .try_init()
    } else {
        subscriber.try_init()
    };
    result.map_err(|error| anyhow::anyhow!("Failed to initialize server tracing: {error}"))
}
