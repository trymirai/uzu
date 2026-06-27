//! Chat persistence: one JSON file per chat under the OS data dir. Synchronous
//! I/O (files are small, saved per exchange). mirai-chat's markdown format for
//! cross-app interop is a follow-up.

use std::{
    fs,
    path::{Path, PathBuf},
};

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
    #[cfg(test)]
    if let Ok(guard) = TEST_DATA_DIR.lock() {
        if let Some(path) = guard.as_ref() {
            return path.clone();
        }
    }
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("Mirai")
}

#[cfg(test)]
static TEST_DATA_DIR: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// Point persistence at a temp directory for deterministic snapshot tests.
#[cfg(test)]
pub fn set_test_data_dir(path: Option<PathBuf>) {
    *TEST_DATA_DIR.lock().expect("test data dir lock") = path;
}

pub fn mirai_data_dir() -> PathBuf {
    mirai_dir()
}

pub fn chats_dir() -> PathBuf {
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
    #[serde(default)]
    pub run_on_startup: bool,
    #[serde(default)]
    pub show_in_menu_bar: bool,
    #[serde(default)]
    pub auto_eject: bool,
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout_minutes: u32,
    #[serde(default)]
    pub share_usage_data: bool,
}

fn default_true() -> bool {
    true
}

fn default_idle_timeout() -> u32 {
    2
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            dark_mode: true,
            reasoning: true,
            run_on_startup: false,
            show_in_menu_bar: false,
            auto_eject: false,
            idle_timeout_minutes: 2,
            share_usage_data: false,
        }
    }
}

pub fn load_settings() -> AppSettings {
    fs::read(settings_path()).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default()
}

pub fn save_settings(settings: &AppSettings) {
    if fs::create_dir_all(mirai_dir()).is_ok() {
        if let Ok(json) = serde_json::to_string_pretty(settings) {
            let _ = fs::write(settings_path(), json);
        }
    }
}

pub(crate) fn global_instructions_path() -> PathBuf {
    chats_dir().join("global-instructions.txt")
}

/// Milliseconds since the Unix epoch (used for ids + timestamps).
pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// A file's modification time in epoch-ms, or 0. Used as the timestamp fallback
/// when reading a markdown chat whose dates aren't in our format.
fn file_mtime_ms(path: &Path) -> u64 {
    use std::time::UNIX_EPOCH;
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
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
    // Human-readable mirror for cross-app interop with mirai-chat. Loading still
    // prefers the JSON, so a bad mirror can't corrupt the stored chat.
    let _ = fs::write(dir.join(format!("{}.md", chat.id)), serialize_markdown(chat));
}

pub fn load_chat(id: &str) -> Option<StoredChat> {
    let dir = chats_dir();
    if let Ok(bytes) = fs::read(dir.join(format!("{id}.json"))) {
        if let Ok(chat) = serde_json::from_slice(&bytes) {
            return Some(chat);
        }
    }
    // Fall back to a markdown file (e.g. one authored by mirai-chat).
    let md_path = dir.join(format!("{id}.md"));
    let text = fs::read_to_string(&md_path).ok()?;
    parse_markdown(&text, id, file_mtime_ms(&md_path))
}

pub fn delete_chat(id: &str) {
    let dir = chats_dir();
    let _ = fs::remove_file(dir.join(format!("{id}.json")));
    let _ = fs::remove_file(dir.join(format!("{id}.md")));
}

pub fn rename_chat(
    id: &str,
    title: &str,
) -> bool {
    let Some(mut chat) = load_chat(id) else {
        return false;
    };
    chat.title = title.to_string();
    chat.updated_at = now_ms();
    save_chat(&chat);
    true
}

/// All saved chats, newest first. JSON files are authoritative; markdown files
/// with no JSON sibling (e.g. imported from mirai-chat) are parsed in too.
pub fn list_chats() -> Vec<StoredChat> {
    let dir = chats_dir();
    let mut chats: Vec<StoredChat> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(bytes) = fs::read(&path) {
                    if let Ok(chat) = serde_json::from_slice::<StoredChat>(&bytes) {
                        seen.insert(chat.id.clone());
                        chats.push(chat);
                    }
                }
            }
        }
    }
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if seen.contains(stem) {
                continue;
            }
            if let Ok(text) = fs::read_to_string(&path) {
                if let Some(chat) = parse_markdown(&text, stem, file_mtime_ms(&path)) {
                    chats.push(chat);
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

// ---- mirai-chat-compatible markdown chat-file format -----------------------
//
// Mirrors external/mirai-chat's `storage/markdown` so chat files interoperate: a
// metadata header, then per-message blocks delimited by `## 👤/🤖` headers and
// HTML-comment sentinels (START_CONTENT, START_COT, START_PERF). We persist a
// single (current) version per message, matching `StoredMessage`; when reading a
// multi-version mirai-chat block we keep the active version.

/// Serialize a chat to the mirai-chat markdown format.
pub fn serialize_markdown(chat: &StoredChat) -> String {
    let date = fmt_utc(chat.created_at);
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", chat.title));
    out.push_str(&format!("**Model:** {}\n", chat.model_name.as_deref().unwrap_or("Unknown")));
    out.push_str(&format!("**Created:** {}\n", fmt_utc(chat.created_at)));
    out.push_str(&format!("**Updated:** {}\n", fmt_utc(chat.updated_at)));
    out.push_str(&format!("**Messages:** {}\n\n---\n\n", chat.messages.len()));

    for (i, m) in chat.messages.iter().enumerate() {
        let (icon, who) = if m.role == "assistant" {
            ("🤖", "Assistant")
        } else {
            ("👤", "User")
        };
        let model = if m.role == "assistant" {
            chat.model_name.as_deref().map(|n| format!(" ({n})")).unwrap_or_default()
        } else {
            String::new()
        };
        out.push_str(&format!("## {icon} {who}{model} - {date}\n\n"));
        out.push_str(&format!("<!-- ID: msg-{i} -->\n"));

        let mut perf: Vec<String> = Vec::new();
        if let Some(tps) = m.tps.filter(|v| *v > 0.0) {
            perf.push(format!("**TPS:** {}", tps.round() as i64));
        }
        if let Some(tok) = m.tokens.filter(|v| *v > 0) {
            perf.push(format!("**TokensOut:** {tok}"));
        }
        if !perf.is_empty() {
            out.push_str("<!-- START_PERF -->\n");
            for line in &perf {
                out.push_str(line);
                out.push('\n');
            }
            out.push_str("<!-- END_PERF -->\n");
        }
        if let Some(cot) = m.reasoning.as_deref().filter(|s| !s.is_empty()) {
            out.push_str("<!-- START_COT -->\n");
            out.push_str(cot);
            out.push_str("\n<!-- END_COT -->\n");
        }
        out.push_str("<!-- START_CONTENT -->\n");
        out.push_str(&m.text);
        out.push_str("\n<!-- END_CONTENT -->\n\n---\n\n");
    }
    out
}

/// Parse a mirai-chat markdown chat file. `fallback_ms` is used for any
/// timestamp that isn't in our `YYYY-MM-DD HH:MM UTC` form (e.g. mirai-chat's
/// locale dates). Returns `None` only if no title/messages can be found.
pub fn parse_markdown(
    md: &str,
    id: &str,
    fallback_ms: u64,
) -> Option<StoredChat> {
    let title = first_line_prefixed(md, "# ")?.trim().to_string();
    let model_name = first_line_prefixed(md, "**Model:** ").map(|s| s.trim().to_string()).filter(|s| s != "Unknown");
    let created_at = first_line_prefixed(md, "**Created:** ").and_then(parse_utc).unwrap_or(fallback_ms);
    let updated_at = first_line_prefixed(md, "**Updated:** ").and_then(parse_utc).unwrap_or(fallback_ms);

    let mut messages = Vec::new();
    for block in message_blocks(md) {
        let role = if block.starts_with("## 🤖") {
            "assistant"
        } else {
            "user"
        };
        let scope = active_version_slice(block);
        let Some(text) = between(scope, "<!-- START_CONTENT -->\n", "\n<!-- END_CONTENT -->") else {
            continue;
        };
        let reasoning = between(scope, "<!-- START_COT -->\n", "\n<!-- END_COT -->")
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty());
        messages.push(StoredMessage {
            role: role.to_string(),
            text: text.to_string(),
            reasoning,
            tps: perf_value(scope, "TPS").map(|v| v as f32),
            tokens: perf_value(scope, "TokensOut").map(|v| v as u32),
        });
    }
    if messages.is_empty() {
        return None;
    }
    Some(StoredChat {
        id: id.to_string(),
        title,
        model_name,
        created_at,
        updated_at,
        messages,
    })
}

/// The first line with `prefix` stripped (whole remainder of that line).
fn first_line_prefixed<'a>(
    md: &'a str,
    prefix: &str,
) -> Option<&'a str> {
    md.lines().find_map(|l| l.strip_prefix(prefix))
}

/// Slice between (after) `start` and (before) `end`, first occurrence.
fn between<'a>(
    text: &'a str,
    start: &str,
    end: &str,
) -> Option<&'a str> {
    let s = text.find(start)? + start.len();
    let e = text[s..].find(end)? + s;
    Some(&text[s..e])
}

/// Byte ranges of each `## 👤/🤖` message block in document order.
fn message_blocks(md: &str) -> Vec<&str> {
    let mut starts = Vec::new();
    let mut idx = 0;
    for line in md.split_inclusive('\n') {
        let t = line.trim_end_matches('\n');
        if t.starts_with("## 👤") || t.starts_with("## 🤖") {
            starts.push(idx);
        }
        idx += line.len();
    }
    starts.iter().enumerate().map(|(i, &s)| &md[s..starts.get(i + 1).copied().unwrap_or(md.len())]).collect()
}

/// For a multi-version block (`#### Version N`), the slice of the active version
/// (the one marked `⭐ ACTIVE`, else the first). Single-version blocks pass
/// through unchanged.
fn active_version_slice(block: &str) -> &str {
    if !block.contains("#### Version") {
        return block;
    }
    let mut starts = Vec::new();
    let mut idx = 0;
    for line in block.split_inclusive('\n') {
        if line.trim_end().starts_with("#### Version") {
            starts.push(idx);
        }
        idx += line.len();
    }
    let seg = |i: usize| &block[starts[i]..starts.get(i + 1).copied().unwrap_or(block.len())];
    let active = (0..starts.len()).find(|&i| seg(i).contains("⭐ ACTIVE")).unwrap_or(0);
    if starts.is_empty() {
        block
    } else {
        seg(active)
    }
}

/// `**LABEL:** <number>` (optional trailing `s`) anywhere in `text`.
fn perf_value(
    text: &str,
    label: &str,
) -> Option<f64> {
    let needle = format!("**{label}:** ");
    text.lines()
        .find_map(|l| l.trim().strip_prefix(needle.as_str()))
        .and_then(|v| v.trim().trim_end_matches('s').parse::<f64>().ok())
}

/// Epoch-ms → `YYYY-MM-DD HH:MM UTC` (minute granularity).
pub fn fmt_utc_public(ms: u64) -> String {
    fmt_utc(ms)
}

/// Epoch-ms → `YYYY-MM-DD HH:MM UTC` (minute granularity).
fn fmt_utc(ms: u64) -> String {
    let secs = (ms / 1000) as i64;
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", rem / 3600, (rem % 3600) / 60)
}

/// Parse `YYYY-MM-DD HH:MM[ UTC]` back to epoch-ms; `None` for other formats.
fn parse_utc(s: &str) -> Option<u64> {
    let s = s.trim().trim_end_matches("UTC").trim();
    let (date, time) = s.split_once(' ')?;
    let mut dp = date.split('-');
    let y: i64 = dp.next()?.parse().ok()?;
    let mo: i64 = dp.next()?.parse().ok()?;
    let d: i64 = dp.next()?.parse().ok()?;
    let mut tp = time.split(':');
    let h: i64 = tp.next()?.parse().ok()?;
    let mi: i64 = tp.next()?.parse().ok()?;
    let secs = days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60;
    (secs >= 0).then(|| secs as u64 * 1000)
}

// Howard Hinnant's civil<->days algorithms (proleptic Gregorian, days from the
// 1970 epoch). Used only for the markdown date fields.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 {
        z
    } else {
        z - 146_096
    } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 {
        mp + 3
    } else {
        mp - 9
    } as u32;
    (
        if m <= 2 {
            y + 1
        } else {
            y
        },
        m,
        d,
    )
}

fn days_from_civil(
    y: i64,
    m: i64,
    d: i64,
) -> i64 {
    let y = if m <= 2 {
        y - 1
    } else {
        y
    };
    let era = if y >= 0 {
        y
    } else {
        y - 399
    } / 400;
    let yoe = y - era * 400;
    let doy =
        (153 * (if m > 2 {
            m - 3
        } else {
            m + 9
        }) + 2)
            / 5
            + d
            - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
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

    // Minute-aligned timestamps so the YYYY-MM-DD HH:MM round-trip is exact.
    fn md_sample() -> StoredChat {
        let mut chat = sample_chat();
        chat.created_at = 1_700_000_040_000;
        chat.updated_at = 1_700_000_100_000;
        chat
    }

    #[test]
    fn utc_date_round_trips() {
        let ms = 1_700_000_040_000;
        assert_eq!(fmt_utc(ms), "2023-11-14 22:14 UTC");
        assert_eq!(parse_utc(&fmt_utc(ms)), Some(ms));
    }

    #[test]
    fn markdown_round_trips() {
        let chat = md_sample();
        let back = parse_markdown(&serialize_markdown(&chat), "abc123", 0).expect("parse");
        assert_eq!(back.title, chat.title);
        assert_eq!(back.model_name, chat.model_name);
        assert_eq!(back.created_at, chat.created_at);
        assert_eq!(back.updated_at, chat.updated_at);
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.messages[0].role, "user");
        assert_eq!(back.messages[0].text, "What is 2+2?");
        assert_eq!(back.messages[1].text, "**2 + 2** equals **4**.");
        assert_eq!(back.messages[1].reasoning.as_deref(), Some("The user asks a simple sum."));
        assert_eq!(back.messages[1].tokens, Some(12));
        assert_eq!(back.messages[1].tps, Some(43.0)); // 42.5 rounds to 43 on write
    }

    // Golden on-disk markdown format. Mirrors mirai-chat's sentinels.
    #[test]
    fn markdown_format_is_stable() {
        insta::assert_snapshot!(serialize_markdown(&md_sample()));
    }

    // A hand-written mirai-chat file: locale dates (unparseable → fallback) and a
    // multi-version block whose ACTIVE version must win.
    #[test]
    fn parses_mirai_chat_multi_version_block() {
        let md = "\
# Sky color

**Model:** Qwen/Qwen3-0.6B
**Created:** Apr 29, 2025, 10:36:00 AM
**Updated:** Apr 29, 2025, 10:37:00 AM
**Messages:** 2

---

## 👤 User - Apr 29, 2025, 10:36:00 AM

<!-- ID: msg-a -->
<!-- START_CONTENT -->
Why is the sky blue?
<!-- END_CONTENT -->

---

## 🤖 Assistant (Qwen/Qwen3-0.6B) - Apr 29, 2025, 10:37:00 AM

<!-- ID: msg-b -->
### 📝 Versions (2)

#### Version 1
<!-- VID: v1 -->
**Time:** Apr 29, 2025, 10:37:00 AM
<!-- START_CONTENT -->
First take.
<!-- END_CONTENT -->

#### Version 2 ⭐ ACTIVE
<!-- VID: v2 -->
**Time:** Apr 29, 2025, 10:38:00 AM
<!-- START_PERF -->
**TPS:** 50
**TokensOut:** 7
<!-- END_PERF -->
<!-- START_COT -->
Rayleigh scattering.
<!-- END_COT -->
<!-- START_CONTENT -->
Because of Rayleigh scattering.
<!-- END_CONTENT -->

---
";
        let chat = parse_markdown(md, "sky", 999).expect("parse");
        assert_eq!(chat.title, "Sky color");
        assert_eq!(chat.model_name.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(chat.created_at, 999); // locale date isn't our format → fallback
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].text, "Why is the sky blue?");
        assert_eq!(chat.messages[1].text, "Because of Rayleigh scattering.");
        assert_eq!(chat.messages[1].reasoning.as_deref(), Some("Rayleigh scattering."));
        assert_eq!(chat.messages[1].tokens, Some(7));
        assert_eq!(chat.messages[1].tps, Some(50.0));
    }
}
