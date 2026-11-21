use dom_query::Document;
use serde::Deserialize;

use super::{HTTP_CLIENT, Provider, RESULT_COUNT, SNIPPET_MAX_CHARS, SearchError, SearchResult};
use crate::chat::messages::{clip_chars, collapse_whitespace};

const ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";
/// What a key Brave does not know is answered with, under HTTP 422.
const INVALID_KEY_CODE: &str = "SUBSCRIPTION_TOKEN_INVALID";

#[derive(Deserialize)]
struct Payload {
    r#type: String,
    web: Option<Web>,
}

#[derive(Deserialize)]
struct Web {
    #[serde(default)]
    results: Vec<Page>,
}

#[derive(Deserialize)]
struct Page {
    url: Option<String>,
    title: Option<String>,
    description: Option<String>,
    page_age: Option<String>,
}

pub(super) async fn search(
    query: &str,
    key: Option<&str>,
) -> Result<Vec<SearchResult>, SearchError> {
    let key = key.ok_or(SearchError::NoKey(Provider::Brave))?;
    let response = HTTP_CLIENT
        .get(ENDPOINT)
        .query(&[("q", query), ("count", &RESULT_COUNT.to_string())])
        .header("accept", "application/json")
        .header("x-subscription-token", key)
        .send()
        .await
        .map_err(|error| SearchError::request(Provider::Brave, error))?;
    let status = response.status();
    let body = super::response_text(response, Provider::Brave).await?;
    match SearchError::from_status(Provider::Brave, status, true) {
        Some(refusal) => Err(failure(refusal, &body)),
        None => parse_results(&body),
    }
}

fn failure(
    refusal: SearchError,
    body: &str,
) -> SearchError {
    let body = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
    if body.pointer("/error/code").and_then(serde_json::Value::as_str) == Some(INVALID_KEY_CODE) {
        SearchError::KeyRejected(Provider::Brave)
    } else {
        refusal
    }
}

fn parse_results(body: &str) -> Result<Vec<SearchResult>, SearchError> {
    let payload = serde_json::from_str::<Payload>(body)
        .ok()
        .filter(|payload| payload.r#type == "search")
        .ok_or_else(|| SearchError::failed(Provider::Brave, "unreadable results"))?;
    let pages = payload.web.map(|web| web.results).unwrap_or_default();
    Ok(pages
        .into_iter()
        .filter_map(|page| {
            Some(SearchResult {
                url: page.url.filter(|url| !url.is_empty())?,
                title: plain_text(&page.title.unwrap_or_default()),
                snippet: clip_chars(&plain_text(&page.description.unwrap_or_default()), SNIPPET_MAX_CHARS),
                published_at: page.page_age.filter(|age| !age.is_empty()),
            })
        })
        .collect())
}

/// Brave marks the matched words in its strings with HTML tags.
fn plain_text(marked: &str) -> String {
    collapse_whitespace(&Document::from(marked).text())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_search_without_a_key_is_not_sent() {
        let outcome = search("query", None).await;
        assert_eq!(outcome.unwrap_err().to_string(), "Brave needs an API key, which is added in Settings");
    }

    #[tokio::test]
    #[ignore = "needs the network"]
    async fn live_brave_refuses_a_made_up_key() {
        let outcome = search(crate::chat::web_search::live::QUERY, Some("not-a-real-key")).await;
        assert_eq!(outcome, Err(SearchError::KeyRejected(Provider::Brave)));
    }

    #[test]
    fn results_are_read_as_plain_text_and_pages_without_an_address_are_dropped() {
        let body = serde_json::json!({
            "type": "search",
            "query": { "original": "rust stable", "more_results_available": true },
            "web": {
                "results": [
                    {
                        "title": "Rust <strong>Versions</strong>",
                        "url": "https://releases.rs/",
                        "description": "The current <strong>stable</strong>\n release &amp; the next ones.",
                        "page_age": "2026-11-12T00:00:00",
                    },
                    { "title": "No address", "description": "dropped" },
                    { "title": "No description", "url": "https://example.com/" },
                ],
            },
        });
        assert_eq!(
            parse_results(&body.to_string()).unwrap(),
            [
                SearchResult {
                    title: "Rust Versions".into(),
                    url: "https://releases.rs/".into(),
                    snippet: "The current stable release & the next ones.".into(),
                    published_at: Some("2026-11-12T00:00:00".into()),
                },
                SearchResult {
                    title: "No description".into(),
                    url: "https://example.com/".into(),
                    snippet: String::new(),
                    published_at: None,
                },
            ]
        );
        assert!(parse_results(r#"{ "type": "search", "query": {} }"#).unwrap().is_empty());
    }

    #[test]
    fn an_answer_that_is_not_a_search_result_is_a_failure() {
        for unreadable in ["<html>", "{}", r#"{ "type": "ErrorResponse", "error": { "code": "VALIDATION" } }"#] {
            assert_eq!(parse_results(unreadable).unwrap_err().to_string(), "Brave search failed: unreadable results");
        }
    }

    #[test]
    fn a_refused_key_a_spent_quota_and_other_failures_are_told_apart() {
        use reqwest::StatusCode;
        let invalid = r#"{"error":{"code":"SUBSCRIPTION_TOKEN_INVALID","detail":"The provided subscription token is invalid.","meta":{"component":"authentication"},"status":422},"type":"ErrorResponse"}"#;
        let failure = |status, body| failure(SearchError::from_status(Provider::Brave, status, true).unwrap(), body);
        assert_eq!(failure(StatusCode::UNPROCESSABLE_ENTITY, invalid), SearchError::KeyRejected(Provider::Brave));
        assert_eq!(failure(StatusCode::UNAUTHORIZED, ""), SearchError::KeyRejected(Provider::Brave));
        assert_eq!(failure(StatusCode::TOO_MANY_REQUESTS, "{}"), SearchError::RateLimited(Provider::Brave));
        let other = r#"{"error":{"code":"VALIDATION","status":422},"type":"ErrorResponse"}"#;
        assert_eq!(failure(StatusCode::UNPROCESSABLE_ENTITY, other).to_string(), "Brave search failed: HTTP 422");
    }
}
