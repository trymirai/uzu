#![allow(dead_code)]

use std::sync::Once;

use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

static INIT: Once = Once::new();

pub fn init_test_tracing() {
    INIT.call_once(|| {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let log_dir = std::env::temp_dir().join("uzu-storage-test-logs");
        std::fs::create_dir_all(&log_dir).ok();

        let file_appender = tracing_appender::rolling::never(&log_dir, format!("storage_{}.log", timestamp));

        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        // Keep guard alive for entire test run
        std::mem::forget(guard);

        tracing_subscriber::registry()
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")))
            .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
            .with(fmt::layer().with_test_writer())
            .init();

        tracing::info!("Test logging initialized, writing to test_logs/");
    });
}
