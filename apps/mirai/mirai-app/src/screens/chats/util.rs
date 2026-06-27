//! Pure helpers for the Chats screen: relative timestamps and rename validation.

use crate::persistence;

/// Human-readable "time ago" for a millisecond timestamp (à la date-fns
/// `formatDistanceToNow`), used as the chat row subtitle.
pub(super) fn relative_time(updated_at: u64) -> String {
    let now = persistence::now_ms();
    let secs = now.saturating_sub(updated_at) / 1000;
    let mins = secs / 60;
    let hours = mins / 60;
    let days = hours / 24;
    if secs < 45 {
        "just now".to_string()
    } else if mins < 60 {
        let n = mins.max(1);
        format!("{n} minute{} ago", if n == 1 { "" } else { "s" })
    } else if hours < 24 {
        format!("{hours} hour{} ago", if hours == 1 { "" } else { "s" })
    } else if days < 30 {
        format!("{days} day{} ago", if days == 1 { "" } else { "s" })
    } else if days < 365 {
        let n = days / 30;
        format!("{n} month{} ago", if n == 1 { "" } else { "s" })
    } else {
        let n = days / 365;
        format!("{n} year{} ago", if n == 1 { "" } else { "s" })
    }
}

pub(super) fn validate_rename_name(name: &str) -> Result<String, &'static str> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("Name cannot be empty");
    }
    if trimmed.len() < 3 {
        return Err("Name must be at least 3 characters long");
    }
    if trimmed.len() > 50 {
        return Err("Name must be at most 50 characters long");
    }
    Ok(trimmed.to_string())
}
