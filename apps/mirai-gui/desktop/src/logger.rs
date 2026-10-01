use std::{
    io::Write,
    path::{Path, PathBuf},
};

use crate::error::AppResult;

const MAX_LOG_SIZE_BYTES: u64 = 5 * 1024 * 1024;

pub fn log_file_path() -> AppResult<PathBuf> {
    Ok(crate::storage::mirai_data_dir()?.join("mirai.log"))
}

pub fn rotated_log_file_path() -> AppResult<PathBuf> {
    Ok(crate::storage::mirai_data_dir()?.join("mirai.log.1"))
}

fn rotate_if_needed(path: &Path) {
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    if meta.len() <= MAX_LOG_SIZE_BYTES {
        return;
    }
    let rotated = path.with_extension("log.1");
    let _ = std::fs::remove_file(&rotated);
    let _ = std::fs::rename(path, &rotated);
}

static LOG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn write_line(
    level: &str,
    message: &str,
    data: Option<serde_json::Value>,
) {
    let Ok(path) = log_file_path() else {
        return;
    };
    let Ok(_guard) = LOG_LOCK.lock() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    rotate_if_needed(&path);
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ");
    let suffix = data.map(|d| format!(" {d}")).unwrap_or_default();
    let line = format!("{ts} [{level}] {message}{suffix}\n");
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = file.write_all(line.as_bytes());
    }
}

pub fn info(
    message: &str,
    data: Option<serde_json::Value>,
) {
    write_line("INFO", message, data);
}

pub fn warn(
    message: &str,
    data: Option<serde_json::Value>,
) {
    write_line("WARN", message, data);
}

pub fn error(
    message: &str,
    data: Option<serde_json::Value>,
) {
    write_line("ERROR", message, data);
}

#[tauri::command]
pub fn get_log_file_path() -> AppResult<String> {
    Ok(log_file_path()?.to_string_lossy().to_string())
}
