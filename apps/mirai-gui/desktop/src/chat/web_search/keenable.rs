use serde::Deserialize;
use serde_json::json;

use super::{Provider, RESULT_COUNT, SNIPPET_MAX_CHARS, SearchError, SearchResult, mcp};
use crate::chat::messages::{clip_chars, collapse_whitespace};

const ENDPOINT: mcp::Endpoint = mcp::Endpoint {
    url: "https://api.keenable.ai/mcp",
    provider: Provider::Keenable,
    rate_limit_flag: None,
    unknown_key_wording: None,
};
const SEARCH_TOOL: &str = "search_web_pages";
pub(super) const PAGE_TOOL: mcp::PageTool = mcp::PageTool {
    endpoint: &ENDPOINT,
    name: "fetch_page_content",
    arguments: |url, max_chars| json!({ "url": url, "max_chars": max_chars }),
    title_label: "Title:",
};

#[derive(Deserialize)]
struct Payload {
    results: Vec<Page>,
}

#[derive(Deserialize)]
struct Page {
    url: Option<String>,
    title: Option<String>,
    snippet: Option<String>,
    description: Option<String>,
    published_at: Option<String>,
}

impl Page {
    fn into_result(self) -> Option<SearchResult> {
        let filled = |value: Option<String>| value.filter(|value| !value.is_empty());
        // Snippets are page excerpts in Markdown, line breaks included.
        let snippet = filled(self.snippet).or_else(|| filled(self.description)).unwrap_or_default();
        let snippet = collapse_whitespace(&snippet);
        Some(SearchResult {
            url: filled(self.url)?,
            title: self.title.unwrap_or_default(),
            snippet: clip_chars(&snippet, SNIPPET_MAX_CHARS),
            published_at: filled(self.published_at),
        })
    }
}

pub(super) async fn search(
    query: &str,
    key: Option<&str>,
) -> Result<Vec<SearchResult>, SearchError> {
    let arguments = json!({
        "query": query,
        "max_results": RESULT_COUNT,
        "snippet_max_length": SNIPPET_MAX_CHARS,
        "output_format": "json",
    });
    let text = mcp::call_tool(&ENDPOINT, SEARCH_TOOL, arguments, key).await?;
    parse_results(&text)
}

fn parse_results(text: &str) -> Result<Vec<SearchResult>, SearchError> {
    let payload = serde_json::from_str::<Payload>(text)
        .map_err(|_| SearchError::failed(Provider::Keenable, "unreadable results"))?;
    Ok(payload.results.into_iter().filter_map(Page::into_result).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::web_search::live;

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_keenable_refuses_a_made_up_key() {
        let outcome = search(live::QUERY, Some("keen_notarealkey")).await;
        assert_eq!(outcome, Err(SearchError::KeyRejected(Provider::Keenable)));
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_keenable_reads_a_page_that_draws_itself_with_scripts() {
        let page = mcp::read_page(&PAGE_TOOL, live::SCRIPT_DRAWN_PAGE, 2_000, None).await.unwrap();
        assert!(page.text.contains(live::SCRIPT_DRAWN_TEXT));
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_keenable_answers_a_keyless_search() {
        live::assert_results(&search(live::QUERY, None).await.unwrap());
    }

    #[test]
    fn results_are_normalized_and_pages_without_an_address_are_dropped() {
        let payload = json!({
            "query": "current stable Rust version",
            "mode": "pro",
            "results": [
                {
                    "title": "Rust",
                    "url": "https://endoflife.date/rust",
                    "description": "",
                    "snippet": "| Release | Latest |\n|---|---|\n| 1.100 | 1.100.0 |\n\nA new stable release every six weeks.",
                    "published_at": "2026-11-16T15:27:49Z",
                    "acquired_at": "2026-11-17T12:23:06Z",
                },
                {
                    "title": "Description only",
                    "url": "https://example.com/",
                    "description": "page description",
                    "snippet": "",
                    "acquired_at": "2026-11-07T08:43:54Z",
                },
                { "title": "No address", "snippet": "dropped" },
            ],
        });

        let results = parse_results(&payload.to_string()).unwrap();

        assert_eq!(
            results,
            [
                SearchResult {
                    title: "Rust".into(),
                    url: "https://endoflife.date/rust".into(),
                    snippet: "| Release | Latest | |---|---| | 1.100 | 1.100.0 | A new stable release every six weeks."
                        .into(),
                    published_at: Some("2026-11-16T15:27:49Z".into()),
                },
                SearchResult {
                    title: "Description only".into(),
                    url: "https://example.com/".into(),
                    snippet: "page description".into(),
                    published_at: None,
                },
            ]
        );
    }

    #[test]
    fn long_snippets_are_clipped_to_the_snippet_limit() {
        let payload = json!({ "results": [{ "url": "https://example.com/", "snippet": "word ".repeat(400) }] });
        let snippet = &parse_results(&payload.to_string()).unwrap()[0].snippet;
        assert_eq!(snippet.chars().count(), SNIPPET_MAX_CHARS);
    }

    #[test]
    fn an_empty_list_is_an_answer_and_any_other_shape_is_a_failure() {
        assert!(parse_results(r#"{ "results": [] }"#).unwrap().is_empty());
        for unreadable in ["{}", "Search failed", r#"{ "results": {} }"#] {
            assert_eq!(
                parse_results(unreadable).unwrap_err().to_string(),
                "Keenable search failed: unreadable results"
            );
        }
    }
}
