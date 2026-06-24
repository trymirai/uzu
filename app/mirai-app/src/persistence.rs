//! Chat persistence: one JSON file per chat under the OS data dir. Synchronous
//! I/O (files are small, saved per exchange). mirai-chat's markdown format for
//! cross-app interop is a follow-up.

use std::{fs, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct StoredMessage {
    pub role: String, // "user" | "assistant"
    pub text: String,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tps: Option<f32>,
    #[serde(default)]
    pub tokens: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StoredChat {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub model_name: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub messages: Vec<StoredMessage>,
}

fn mirai_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Mirai")
}

fn chats_dir() -> PathBuf {
    mirai_dir().join("chats")
}

fn welcome_marker_path() -> PathBuf {
    mirai_dir().join("welcomed")
}

fn settings_path() -> PathBuf {
    mirai_dir().join("settings.json")
}

/// True once the user has dismissed the welcome/onboarding screen.
pub fn has_seen_welcome() -> bool {
    welcome_marker_path().exists()
}

pub fn set_seen_welcome() {
    if fs::create_dir_all(mirai_dir()).is_ok() {
        let _ = fs::write(welcome_marker_path(), b"1");
    }
}

/// Persisted app settings (Settings screen).
#[derive(Serialize, Deserialize, Clone)]
pub struct AppSettings {
    #[serde(default = "default_true")]
    pub dark_mode: bool,
    #[serde(default = "default_true")]
    pub reasoning: bool,
}

fn default_true() -> bool {
    true
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            dark_mode: true,
            reasoning: true,
        }
    }
}

pub fn load_settings() -> AppSettings {
    fs::read(settings_path())
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub fn save_settings(settings: &AppSettings) {
    if fs::create_dir_all(mirai_dir()).is_ok() {
        if let Ok(json) = serde_json::to_string_pretty(settings) {
            let _ = fs::write(settings_path(), json);
        }
    }
}

fn global_instructions_path() -> PathBuf {
    chats_dir().join("global-instructions.txt")
}

/// Milliseconds since the Unix epoch (used for ids + timestamps).
pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn save_chat(chat: &StoredChat) {
    let dir = chats_dir();
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(json) = serde_json::to_string_pretty(chat) {
        let _ = fs::write(dir.join(format!("{}.json", chat.id)), json);
    }
}

pub fn load_chat(id: &str) -> Option<StoredChat> {
    let bytes = fs::read(chats_dir().join(format!("{id}.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub fn delete_chat(id: &str) {
    let _ = fs::remove_file(chats_dir().join(format!("{id}.json")));
}

/// All saved chats, newest first.
pub fn list_chats() -> Vec<StoredChat> {
    let mut chats: Vec<StoredChat> = Vec::new();
    if let Ok(entries) = fs::read_dir(chats_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(bytes) = fs::read(&path) {
                    if let Ok(chat) = serde_json::from_slice::<StoredChat>(&bytes) {
                        chats.push(chat);
                    }
                }
            }
        }
    }
    chats.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    chats
}

pub fn global_instructions() -> String {
    fs::read_to_string(global_instructions_path()).unwrap_or_default()
}

#[allow(dead_code)] // wired to the Settings/Chats editor in a later step
pub fn set_global_instructions(text: &str) {
    let dir = chats_dir();
    if fs::create_dir_all(&dir).is_ok() {
        let _ = fs::write(global_instructions_path(), text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chat() -> StoredChat {
        StoredChat {
            id: "abc123".into(),
            title: "What is 2+2?".into(),
            model_name: Some("Qwen/Qwen3-0.6B".into()),
            created_at: 1_700_000_000_000,
            updated_at: 1_700_000_005_000,
            messages: vec![
                StoredMessage {
                    role: "user".into(),
                    text: "What is 2+2?".into(),
                    reasoning: None,
                    tps: None,
                    tokens: None,
                },
                StoredMessage {
                    role: "assistant".into(),
                    text: "**2 + 2** equals **4**.".into(),
                    reasoning: Some("The user asks a simple sum.".into()),
                    tps: Some(42.5),
                    tokens: Some(12),
                },
            ],
        }
    }

    // Golden file format: a change here means existing on-disk chats may no
    // longer round-trip. Review the snapshot diff before accepting.
    #[test]
    fn chat_json_format_is_stable() {
        insta::assert_json_snapshot!(sample_chat());
    }

    #[test]
    fn chat_round_trips() {
        let chat = sample_chat();
        let json = serde_json::to_string(&chat).unwrap();
        let back: StoredChat = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, chat.id);
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.messages[1].tokens, Some(12));
    }

    // Optional fields must default when absent (older files / mirai-chat).
    #[test]
    fn legacy_message_without_optionals_parses() {
        let json = r#"{"role":"assistant","text":"hi"}"#;
        let msg: StoredMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.text, "hi");
        assert!(msg.reasoning.is_none());
        assert!(msg.tps.is_none());
    }

    #[test]
    fn settings_default_to_dark_with_reasoning() {
        let s = AppSettings::default();
        assert!(s.dark_mode);
        assert!(s.reasoning);
    }
}
