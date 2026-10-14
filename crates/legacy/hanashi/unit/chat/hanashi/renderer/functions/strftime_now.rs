use chrono::{TimeZone, Utc};

use super::*;

fn time() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2024, 7, 26, 9, 5, 3).unwrap()
}

#[test]
fn formats_the_patterns_chat_templates_use() {
    for (pattern, expected) in [
        ("%Y-%m-%d", "2024-07-26"),
        ("%d %b %Y", "26 Jul 2024"),
        ("%B %d, %Y", "July 26, 2024"),
        ("%A", "Friday"),
        ("%H:%M:%S", "09:05:03"),
        ("%Y-%m-%d %H:%M:%S", "2024-07-26 09:05:03"),
    ] {
        assert_eq!(format_time(&time(), pattern), expected, "pattern {pattern}");
    }
}

#[test]
fn invalid_or_empty_pattern_gives_empty_string() {
    assert_eq!(format_time(&time(), ""), "");
    assert_eq!(format_time(&time(), "%Q"), "");
}

#[test]
fn now_is_not_empty_for_a_valid_pattern() {
    assert_eq!(strftime_now("%Y".to_string()).len(), 4);
}
