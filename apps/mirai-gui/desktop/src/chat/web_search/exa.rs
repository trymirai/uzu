use serde_json::json;

use super::{Provider, RESULT_COUNT, SNIPPET_MAX_CHARS, SearchError, SearchResult, mcp};
use crate::chat::messages::clip_chars;

const ENDPOINT: mcp::Endpoint = mcp::Endpoint {
    url: "https://mcp.exa.ai/mcp",
    provider: Provider::Exa,
    rate_limit_flag: Some("ai.exa/rateLimited"),
    unknown_key_wording: Some("Invalid API key"),
};
const SEARCH_TOOL: &str = "web_search_exa";
pub(super) const PAGE_TOOL: mcp::PageTool = mcp::PageTool {
    endpoint: &ENDPOINT,
    name: "web_fetch_exa",
    arguments: |url, max_chars| json!({ "urls": [url], "maxCharacters": max_chars }),
    title_label: "# ",
};
const RULE: &str = "---";
const ELLIPSIS_LINE: &str = "...";
const NOT_AVAILABLE: &str = "N/A";

pub(super) async fn search(
    query: &str,
    key: Option<&str>,
) -> Result<Vec<SearchResult>, SearchError> {
    let arguments = json!({ "query": query, "numResults": RESULT_COUNT });
    let text = mcp::call_tool(&ENDPOINT, SEARCH_TOOL, arguments, key).await?;
    let results = parse_results(&text);
    if results.is_empty() && !text.trim().is_empty() {
        return Err(SearchError::failed(Provider::Exa, "unrecognized response"));
    }
    Ok(results)
}

#[derive(Default)]
struct Draft<'text> {
    title: &'text str,
    url: &'text str,
    published: &'text str,
    in_highlights: bool,
    highlights: Vec<&'text str>,
}

impl Draft<'_> {
    fn finish(self) -> Option<SearchResult> {
        (!self.url.is_empty()).then(|| SearchResult {
            title: self.title.to_string(),
            url: self.url.to_string(),
            snippet: clip_chars(&self.highlights.join(" "), SNIPPET_MAX_CHARS),
            published_at: (!self.published.is_empty() && self.published != NOT_AVAILABLE)
                .then(|| self.published.to_string()),
        })
    }
}

fn parse_results(text: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let mut draft: Option<Draft> = None;
    let mut at_block_start = true;
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if line == RULE {
            at_block_start = true;
            continue;
        }
        let starts_result = at_block_start.then(|| line.strip_prefix("Title:")).flatten();
        at_block_start = false;
        if let Some(title) = starts_result {
            results.extend(draft.take().and_then(Draft::finish));
            draft = Some(Draft {
                title: title.trim(),
                ..Draft::default()
            });
            continue;
        }
        let Some(draft) = draft.as_mut() else {
            continue;
        };
        if draft.in_highlights {
            // The excerpts open with the page title, which the result already has.
            let repeats_title = draft.highlights.is_empty() && line == draft.title;
            if line != ELLIPSIS_LINE && !repeats_title {
                draft.highlights.push(line);
            }
        } else if let Some(url) = line.strip_prefix("URL:") {
            draft.url = url.trim();
        } else if let Some(published) = line.strip_prefix("Published:") {
            draft.published = published.trim();
        } else if line.starts_with("Highlights:") {
            draft.in_highlights = true;
        }
    }
    results.extend(draft.and_then(Draft::finish));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::web_search::live;

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_exa_refuses_a_made_up_key() {
        let outcome = search(live::QUERY, Some("not-a-real-key")).await;
        assert_eq!(outcome, Err(SearchError::KeyRejected(Provider::Exa)));
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_exa_reads_a_page_that_draws_itself_with_scripts() {
        let page = mcp::read_page(&PAGE_TOOL, live::SCRIPT_DRAWN_PAGE, 2_000, None).await.unwrap();
        assert!(page.text.contains(live::SCRIPT_DRAWN_TEXT));
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_exa_answers_a_keyless_search() {
        live::assert_results(&search(live::QUERY, None).await.unwrap());
    }

    const RESPONSE: &str = "Title: Announcing Rust 1.99.1 | Rust Blog\n\
        URL: https://blog.rust-lang.org/2026/10/15/Rust-1.99.1/\n\
        Published: 2026-10-15T00:00:00.000Z\n\
        Author: The Rust Release Team\n\
        Highlights:\n\
        Announcing Rust 1.99.1 | Rust Blog\n\
        \n\
        ## Announcing Rust 1.99.1\n\
        ...\n\
        The Rust team has published a new point release.\n\
        \n\
        ---\n\
        \n\
        Title: Rust Release Notes\n\
        URL: https://doc.rust-lang.org/stable/releases.html\n\
        Published: N/A\n\
        Author: N/A\n\
        Highlights:\n\
        # Version 1.99.1\n\
        \n\
        ---\n\
        \n\
        - library: fix missing symbol\n\
        Title: not a new result\n";

    #[test]
    fn results_are_read_from_the_labelled_blocks_whatever_their_excerpts_contain() {
        let results = parse_results(RESPONSE);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Announcing Rust 1.99.1 | Rust Blog");
        assert_eq!(results[0].url, "https://blog.rust-lang.org/2026/10/15/Rust-1.99.1/");
        assert_eq!(results[0].published_at.as_deref(), Some("2026-10-15T00:00:00.000Z"));
        assert_eq!(results[0].snippet, "## Announcing Rust 1.99.1 The Rust team has published a new point release.");
        assert_eq!(results[1].title, "Rust Release Notes");
        assert_eq!(results[1].published_at, None);
        assert_eq!(results[1].snippet, "# Version 1.99.1 - library: fix missing symbol Title: not a new result");
    }

    #[test]
    fn long_excerpts_are_clipped_to_the_snippet_limit() {
        let long = format!("Title: Long\nURL: https://example.com/\nHighlights:\n{}", "word ".repeat(400));
        let snippet = &parse_results(&long)[0].snippet;
        assert_eq!(snippet.chars().count(), SNIPPET_MAX_CHARS);
        assert!(snippet.ends_with('…'));
    }

    #[test]
    fn blocks_without_an_address_and_unlabelled_text_yield_no_results() {
        assert!(parse_results("Title: No address\nHighlights:\ntext").is_empty());
        assert!(parse_results("Something else entirely.").is_empty());
        assert!(parse_results("").is_empty());
    }
}
