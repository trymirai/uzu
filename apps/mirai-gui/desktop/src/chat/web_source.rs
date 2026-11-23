use serde::Serialize;

use super::messages::{clip_chars, collapse_whitespace};

const TITLE_MAX_CHARS: usize = 200;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WebSource {
    pub title: String,
    pub url: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct PageContent {
    pub(super) title: String,
    pub(super) text: String,
    pub(super) published_at: Option<String>,
    /// The reader returned only the start of a longer page.
    pub(super) cut_short: bool,
}

pub(super) fn title_line(title: &str) -> String {
    clip_chars(&collapse_whitespace(title), TITLE_MAX_CHARS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_title_is_one_line_of_bounded_length() {
        assert_eq!(title_line(" Breaking\u{2028}news \n today "), "Breaking news today");
        assert_eq!(title_line(&"word ".repeat(100)).chars().count(), TITLE_MAX_CHARS);
    }
}
