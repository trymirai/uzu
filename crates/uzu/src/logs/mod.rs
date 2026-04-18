use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    sync::Once,
};

use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

static INIT: Once = Once::new();

const MAX_LOG_SIZE_BYTES: u64 = 10 * 1024 * 1024;
const KEEP_LINES: usize = 5000;

/// Initialize tracing for production use
/// Logs are written to a single file that gets trimmed when it exceeds MAX_LOG_SIZE_BYTES
/// Keeps only the last KEEP_LINES lines when trimming
pub fn start(
    path: PathBuf,
    name: &str,
    stdout: bool,
) {
    INIT.call_once(|| {
        init(path, name, MAX_LOG_SIZE_BYTES, KEEP_LINES, stdout);
    });
}

fn init(
    path: PathBuf,
    name: &str,
    max_size_bytes: u64,
    keep_lines: usize,
    also_stdout: bool,
) {
    if let Err(e) = fs::create_dir_all(&path) {
        eprintln!("Failed to create log directory {:?}: {}", path, e);
        return;
    }

    let log_file_path = path.join(name);
    // Spawn trimming worker
    start_trim_thread(log_file_path.clone(), max_size_bytes, keep_lines);

    // File appender (no built-in rotation)
    let file_appender = tracing_appender::rolling::never(&path, name);
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    // Keep guard alive for the process lifetime
    std::mem::forget(guard);

    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let registry = tracing_subscriber::registry().with(env_filter).with(
        fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true),
    );

    if also_stdout {
        registry.with(fmt::layer().with_writer(std::io::stdout)).init();
    } else {
        registry.init();
    }
}

fn trim_log_file(
    log_path: &PathBuf,
    keep_lines: usize,
) -> std::io::Result<()> {
    let file = File::open(log_path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader
        .lines()
        .filter_map(Result::ok)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take(keep_lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let temp_path = log_path.with_extension("log.tmp");
    let mut temp_file = OpenOptions::new().write(true).create(true).truncate(true).open(&temp_path)?;

    for line in lines {
        writeln!(temp_file, "{}", line)?;
    }
    temp_file.sync_all()?;
    drop(temp_file);

    fs::rename(&temp_path, log_path)?;
    Ok(())
}

fn start_trim_thread(
    log_path: PathBuf,
    max_size_bytes: u64,
    keep_lines: usize,
) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
            if let Ok(metadata) = fs::metadata(&log_path) {
                if metadata.len() > max_size_bytes {
                    let _ = trim_log_file(&log_path, keep_lines);
                }
            }
        }
    });
}
