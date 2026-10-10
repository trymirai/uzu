pub mod cleanup;

use std::path::{Path, PathBuf};

use crate::error::{AppError, AppResult};

// Must stay "~/Library/Application Support/mirai": user chats and settings
// live here, and renaming the directory would strand them.
pub fn mirai_data_dir() -> AppResult<PathBuf> {
    let home = std::env::var_os("HOME").ok_or_else(|| AppError::msg("HOME not set"))?;
    Ok(PathBuf::from(home).join("Library/Application Support/mirai"))
}

pub fn chats_dir() -> AppResult<PathBuf> {
    Ok(mirai_data_dir()?.join("chats"))
}

pub fn settings_path() -> AppResult<PathBuf> {
    Ok(mirai_data_dir()?.join("settings.json"))
}

// tmp-then-rename so a crash mid-write can't leave a truncated file behind.
pub(crate) fn write_atomic(
    path: &Path,
    content: &[u8],
) -> AppResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, content)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// Confines a client-supplied path to the chats dir: strips any directory
// components so "../escape" can't reach outside the store.
fn resolve(rel: &str) -> AppResult<PathBuf> {
    let name = Path::new(rel).file_name().ok_or_else(|| AppError::msg(format!("Invalid chat path: {rel}")))?;
    Ok(chats_dir()?.join(name))
}

#[tauri::command]
pub async fn chat_list_files() -> AppResult<Vec<String>> {
    let dir = chats_dir()?;
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    let mut files = Vec::new();
    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str()
            && name.ends_with(".md")
        {
            files.push(name.to_string());
        }
    }
    Ok(files)
}

#[tauri::command]
pub async fn chat_load_file(path: String) -> AppResult<Option<String>> {
    read_optional(&resolve(&path)?)
}

#[tauri::command]
pub async fn chat_save_file(
    path: String,
    content: String,
) -> AppResult<()> {
    let full = resolve(&path)?;
    write_atomic(&full, content.as_bytes())
}

#[tauri::command]
pub async fn chat_delete_file(path: String) -> AppResult<()> {
    let full = resolve(&path)?;
    match std::fs::remove_file(full) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn global_instructions_path() -> AppResult<PathBuf> {
    Ok(chats_dir()?.join("global-instructions.txt"))
}

#[tauri::command]
pub async fn global_instructions_save(content: String) -> AppResult<()> {
    write_atomic(&global_instructions_path()?, content.as_bytes())
}

#[tauri::command]
pub async fn global_instructions_load() -> AppResult<Option<String>> {
    read_optional(&global_instructions_path()?)
}

fn read_optional(path: &Path) -> AppResult<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(content) => Ok(Some(content)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[tauri::command]
pub async fn save_binary_file(
    absolute_path: String,
    data: Vec<u8>,
) -> AppResult<bool> {
    match std::fs::write(&absolute_path, &data) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

#[tauri::command]
pub async fn read_text_file(absolute_path: String) -> AppResult<Option<String>> {
    Ok(std::fs::read_to_string(&absolute_path).ok())
}

// settings.json keys are an on-disk contract: renaming one silently resets
// that preference for everyone who already has the file.
static SETTINGS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// A missing file is a fresh install. An unreadable one must not be treated the
// same way: the next patch would overwrite every stored preference with just
// its own keys. A file that does not parse is moved aside so it can be recovered.
fn read_settings() -> AppResult<serde_json::Map<String, serde_json::Value>> {
    let path = settings_path()?;
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(serde_json::Map::new()),
        Err(e) => return Err(AppError::msg(format!("Failed to read settings: {e}"))),
    };
    match serde_json::from_str(&raw) {
        Ok(serde_json::Value::Object(map)) => Ok(map),
        _ => {
            let backup = path.with_extension("corrupt.json");
            crate::logger::error(
                "settings:corrupt",
                Some(serde_json::json!({ "path": path.display().to_string(), "backup": backup.display().to_string() })),
            );
            std::fs::rename(&path, &backup)
                .map_err(|e| AppError::msg(format!("Failed to set aside corrupt settings: {e}")))?;
            Ok(serde_json::Map::new())
        },
    }
}

#[tauri::command]
pub fn settings_load() -> AppResult<serde_json::Value> {
    let _guard = SETTINGS_LOCK.lock().expect("settings lock poisoned");
    read_settings().map(serde_json::Value::Object)
}

fn write_settings(settings: &serde_json::Map<String, serde_json::Value>) -> AppResult<()> {
    let json = serde_json::to_string_pretty(settings)?;
    write_atomic(&settings_path()?, json.as_bytes())
}

pub fn settings_merge(
    patch: serde_json::Map<String, serde_json::Value>
) -> AppResult<serde_json::Map<String, serde_json::Value>> {
    let _guard = SETTINGS_LOCK.lock().expect("settings lock poisoned");
    let mut current = read_settings()?;
    current.extend(patch);
    write_settings(&current)?;
    Ok(current)
}

#[tauri::command]
pub fn settings_patch(
    analytics: tauri::State<'_, crate::analytics::AnalyticsState>,
    patch: serde_json::Value,
) -> AppResult<()> {
    let serde_json::Value::Object(patch) = patch else {
        return Err(AppError::msg("settings patch must be an object"));
    };
    analytics.patch_settings(patch)
}

// One model's entry changes under the settings lock, so the other models'
// parameters are never rewritten from a client-side copy.
#[tauri::command]
pub fn model_params_set(
    repo_id: String,
    params: Option<serde_json::Value>,
) -> AppResult<()> {
    let _guard = SETTINGS_LOCK.lock().expect("settings lock poisoned");
    let mut current = read_settings()?;
    let mut model_params = match current.remove("modelParams") {
        Some(serde_json::Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    match params {
        Some(params) => model_params.insert(repo_id, params),
        None => model_params.remove(&repo_id),
    };
    current.insert("modelParams".to_string(), serde_json::Value::Object(model_params));
    write_settings(&current)
}
