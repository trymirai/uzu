use std::{
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
};

use uzu::{engine::Engine, storage::types::DownloadPhase};

use crate::{persistence, tts_history};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CleanupCategory {
    Dialogs,
    Files,
    Models,
    Logs,
}

impl CleanupCategory {
    pub const ALL: [Self; 4] = [Self::Dialogs, Self::Files, Self::Models, Self::Logs];

    pub fn label(self) -> &'static str {
        match self {
            Self::Dialogs => "Dialogs",
            Self::Files => "Generated audio",
            Self::Models => "Downloaded models",
            Self::Logs => "Logs",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CategoryStats {
    pub count: u32,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct CleanupPreview {
    pub dialogs: CategoryStats,
    pub files: CategoryStats,
    pub models: CategoryStats,
    pub logs_size_bytes: u64,
}

pub fn log_file_path() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join(".cache").join("mirai").join("mirai.log")
}

pub fn tts_audio_dir() -> PathBuf {
    persistence::mirai_data_dir().join("tts-audio")
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

pub fn category_description(
    cat: CleanupCategory,
    preview: &CleanupPreview,
) -> String {
    match cat {
        CleanupCategory::Dialogs => {
            let n = preview.dialogs.count;
            let word = if n == 1 {
                "chat"
            } else {
                "chats"
            };
            format!("{n} {word} · {}", format_bytes(preview.dialogs.size_bytes))
        },
        CleanupCategory::Files => {
            let n = preview.files.count;
            let word = if n == 1 {
                "file"
            } else {
                "files"
            };
            format!("{n} {word} · {}", format_bytes(preview.files.size_bytes))
        },
        CleanupCategory::Models => {
            let n = preview.models.count;
            let word = if n == 1 {
                "model"
            } else {
                "models"
            };
            format!("{n} {word} · {}", format_bytes(preview.models.size_bytes))
        },
        CleanupCategory::Logs => format_bytes(preview.logs_size_bytes),
    }
}

fn dir_file_stats(
    dir: &Path,
    ext: Option<&str>,
) -> CategoryStats {
    let Ok(entries) = fs::read_dir(dir) else {
        return CategoryStats::default();
    };
    let mut stats = CategoryStats::default();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if ext.is_some_and(|e| path.extension().and_then(|x| x.to_str()) != Some(e)) {
            continue;
        }
        stats.count += 1;
        stats.size_bytes += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    }
    stats
}

fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

pub fn cleanup_preview_disk() -> CleanupPreview {
    let chats = persistence::chats_dir();
    CleanupPreview {
        dialogs: dir_file_stats(&chats, Some("md")),
        files: dir_file_stats(&tts_audio_dir(), None),
        logs_size_bytes: file_size(&log_file_path()) + file_size(&log_file_path_with_suffix(".1")),
        ..Default::default()
    }
}

pub async fn model_cleanup_stats(engine: &Engine) -> CategoryStats {
    let Ok(models) = engine.models().await else {
        return CategoryStats::default();
    };
    let states = engine.download_states().await;
    let mut stats = CategoryStats::default();
    for model in models {
        let Some(state) = states.get(&model.identifier) else {
            continue;
        };
        if !matches!(state.phase, DownloadPhase::Downloaded {} | DownloadPhase::Error { .. }) {
            continue;
        }
        stats.count += 1;
        stats.size_bytes += model.properties.as_ref().map(|p| p.size.max(0) as u64).unwrap_or(0);
    }
    stats
}

pub fn export_chats_zip() -> Option<Vec<u8>> {
    let dir = persistence::chats_dir();
    let mut names: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md")
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
                && seen.insert(name.to_string())
            {
                names.push(name.to_string());
            }
        }
    }

    for chat in persistence::list_chats() {
        let name = format!("{}.md", chat.id);
        if seen.insert(name.clone()) {
            names.push(name);
        }
    }
    if names.is_empty() {
        return None;
    }
    names.sort();

    let mut buf = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for name in names {
            let path = dir.join(&name);
            let content = if path.exists() {
                fs::read_to_string(&path).unwrap_or_default()
            } else if let Some(stem) = name.strip_suffix(".md") {
                persistence::load_chat(stem).map(|c| persistence::serialize_markdown(&c)).unwrap_or_default()
            } else {
                String::new()
            };
            if content.is_empty() {
                continue;
            }
            zip.start_file(name, opts).ok()?;
            zip.write_all(content.as_bytes()).ok()?;
        }
        zip.finish().ok()?;
    }
    Some(buf.into_inner())
}

pub fn export_zip_default_name() -> String {
    let stamp = persistence::fmt_utc(persistence::now_ms()).replace(':', ".").replace(" UTC", "");
    format!("mirai-chats {stamp}.zip")
}

pub fn read_log_bytes() -> Option<Vec<u8>> {
    let rotated = log_file_path_with_suffix(".1");
    let primary = log_file_path();
    let mut bytes = Vec::new();
    if let Ok(mut rotated_bytes) = fs::read(&rotated) {
        bytes.append(&mut rotated_bytes);
    }
    if let Ok(mut primary_bytes) = fs::read(&primary) {
        bytes.append(&mut primary_bytes);
    }
    if bytes.is_empty() {
        None
    } else {
        Some(bytes)
    }
}

fn log_file_path_with_suffix(suffix: &str) -> PathBuf {
    let mut p = log_file_path().into_os_string();
    p.push(suffix);
    PathBuf::from(p)
}

pub fn clear_dialogs() -> bool {
    let dir = persistence::chats_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        return true;
    };
    let mut ok = true;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str());
        if ext == Some("md") || ext == Some("json") {
            ok &= fs::remove_file(path).is_ok();
        }
    }
    let _ = fs::remove_file(persistence::global_instructions_path());
    ok
}

pub fn clear_generated_audio() -> bool {
    tts_history::clear_all();
    let dir = tts_audio_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        return true;
    };
    let mut ok = true;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            ok &= fs::remove_file(path).is_ok();
        }
    }
    ok
}

pub fn clear_logs() -> bool {
    let primary = log_file_path();
    let mut ok = true;
    if primary.exists() {
        ok &= fs::write(&primary, []).is_ok();
    }
    let rotated = log_file_path_with_suffix(".1");
    if rotated.exists() {
        ok &= fs::remove_file(&rotated).is_ok();
    }
    ok
}

pub async fn clear_downloaded_models(engine: &Engine) -> bool {
    let Ok(models) = engine.models().await else {
        return false;
    };
    let states = engine.download_states().await;
    let mut ok = true;
    for model in models {
        let Some(state) = states.get(&model.identifier) else {
            continue;
        };
        if !matches!(state.phase, DownloadPhase::Downloaded {} | DownloadPhase::Error { .. }) {
            continue;
        }
        if engine.downloader(&model).delete().await.is_err() {
            ok = false;
        }
    }
    ok
}

pub async fn execute_cleanup(
    engine: Option<&Engine>,
    categories: &[CleanupCategory],
) -> Vec<(CleanupCategory, bool)> {
    let mut results = Vec::new();
    for &cat in categories {
        let ok = match cat {
            CleanupCategory::Dialogs => clear_dialogs(),
            CleanupCategory::Files => clear_generated_audio(),
            CleanupCategory::Logs => clear_logs(),
            CleanupCategory::Models => {
                if let Some(engine) = engine {
                    clear_downloaded_models(engine).await
                } else {
                    false
                }
            },
        };
        results.push((cat, ok));
    }
    results
}
