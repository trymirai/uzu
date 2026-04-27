use std::{env::current_dir, fs::create_dir_all, io::stdout, mem::forget, path::PathBuf, sync::Once};

use chrono::Utc;
use tracing_appender::{non_blocking, rolling};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[allow(dead_code)]
static INIT: Once = Once::new();

#[allow(dead_code)]
pub fn init_test_tracing() {
    INIT.call_once(|| {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");

        // Use workspace root for test_logs
        let log_dir = current_dir().unwrap_or_else(|_| PathBuf::from(".")).join("test_logs");
        create_dir_all(&log_dir).ok();

        let file_appender = rolling::never(&log_dir, format!("file_download_manager_{}.log", timestamp));

        let (non_blocking, guard) = non_blocking(file_appender);

        // Keep guard alive for entire test run
        forget(guard);

        tracing_subscriber::registry()
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")))
            .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
            .with(fmt::layer().with_writer(stdout))
            .init();

        tracing::info!("Test logging initialized, writing to test_logs/");
    });
}
