use std::fmt::Write;

use chrono::{
    Local,
    format::{Item, StrftimeItems},
};

/// Formats the current local time according to the provided `strftime` pattern
pub fn strftime_now(format_string: String) -> String {
    let items = StrftimeItems::new(&format_string);
    if items.clone().any(|item| matches!(item, Item::Error)) {
        return String::new();
    }

    let mut result = String::new();
    if write!(result, "{}", Local::now().format_with_items(items)).is_err() {
        return String::new();
    }
    result
}
