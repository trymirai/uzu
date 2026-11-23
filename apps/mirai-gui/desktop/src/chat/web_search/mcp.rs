use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{HTTP_CLIENT, Provider, SearchError};
use crate::chat::web_source::PageContent;

pub(super) struct Endpoint {
    pub(super) url: &'static str,
    pub(super) provider: Provider,
    /// The `_meta` key a provider sets on a throttled call instead of an HTTP 429.
    pub(super) rate_limit_flag: Option<&'static str>,
    /// What a failed tool call says about an unknown key, when no HTTP status does.
    pub(super) unknown_key_wording: Option<&'static str>,
}

pub(super) struct PageTool {
    pub(super) endpoint: &'static Endpoint,
    pub(super) name: &'static str,
    pub(super) arguments: fn(&str, usize) -> Value,
    pub(super) title_label: &'static str,
}

#[derive(Deserialize)]
struct Response {
    result: Option<ToolResult>,
    error: Option<ProtocolError>,
}

#[derive(Deserialize)]
struct ProtocolError {
    #[serde(default)]
    message: String,
}

#[derive(Deserialize)]
struct ToolResult {
    #[serde(default)]
    content: Vec<ContentBlock>,
    #[serde(default, rename = "isError")]
    is_error: bool,
    #[serde(default, rename = "_meta")]
    meta: Map<String, Value>,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: Option<String>,
}

pub(super) async fn call_tool(
    endpoint: &Endpoint,
    tool: &str,
    arguments: Value,
    key: Option<&str>,
) -> Result<String, SearchError> {
    let provider = endpoint.provider;
    let call = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": tool, "arguments": arguments },
    });
    let request = HTTP_CLIENT.post(endpoint.url).header("accept", "application/json, text/event-stream").json(&call);
    let request = match key {
        Some(key) => request.header("x-api-key", key),
        None => request,
    };
    let response = request.send().await.map_err(|error| SearchError::request(provider, error))?;
    if let Some(refusal) = SearchError::from_status(provider, response.status(), key.is_some()) {
        return Err(refusal);
    }
    let body = super::response_text(response, provider).await?;
    match tool_result_text(&body, provider, endpoint.rate_limit_flag) {
        Err(SearchError::Rejected {
            message,
            ..
        }) if key.is_some() && endpoint.unknown_key_wording.is_some_and(|wording| message.contains(wording)) => {
            Err(SearchError::KeyRejected(provider))
        },
        outcome => outcome,
    }
}

pub(super) async fn read_page(
    tool: &PageTool,
    url: &str,
    max_chars: usize,
    key: Option<&str>,
) -> Result<PageContent, SearchError> {
    let answer = call_tool(tool.endpoint, tool.name, (tool.arguments)(url, max_chars), key).await?;
    labelled_page(&answer, tool.title_label)
        .ok_or_else(|| SearchError::failed(tool.endpoint.provider, "unrecognized page"))
}

fn labelled_page(
    answer: &str,
    title_label: &str,
) -> Option<PageContent> {
    let (labels, text) = answer.split_once("\n\n")?;
    let labelled = |label: &str| {
        let value = labels.lines().find_map(|line| line.strip_prefix(label))?.trim();
        (!value.is_empty()).then(|| value.to_string())
    };
    // Without a URL line this is not the format we know.
    labelled("URL:")?;
    let text = text.trim();
    (!text.is_empty()).then(|| PageContent {
        title: labelled(title_label).unwrap_or_default(),
        text: text.to_string(),
        published_at: labelled("Published:"),
        cut_short: labelled("Truncated:").is_some(),
    })
}

fn tool_result_text(
    body: &str,
    provider: Provider,
    rate_limit_flag: Option<&str>,
) -> Result<String, SearchError> {
    let response = find_response(body).ok_or_else(|| SearchError::failed(provider, "unreadable response"))?;
    if let Some(error) = response.error {
        return Err(SearchError::rejected(provider, &error.message));
    }
    let result = response.result.ok_or_else(|| SearchError::failed(provider, "response has no result"))?;
    if rate_limit_flag.and_then(|flag| result.meta.get(flag)).and_then(Value::as_bool) == Some(true) {
        return Err(SearchError::RateLimited(provider));
    }
    let text = result.content.into_iter().find_map(|block| block.text);
    if result.is_error {
        return Err(SearchError::rejected(provider, text.as_deref().unwrap_or("tool call failed")));
    }
    text.ok_or_else(|| SearchError::failed(provider, "response has no text"))
}

fn find_response(body: &str) -> Option<Response> {
    let answered = |response: &Response| response.result.is_some() || response.error.is_some();
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|event| serde_json::from_str::<Response>(event).ok())
        .find(answered)
        .or_else(|| serde_json::from_str::<Response>(body).ok().filter(answered))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROVIDER: Provider = Provider::Exa;
    const FLAG: &str = "ai.exa/rateLimited";

    fn text(body: &str) -> Result<String, SearchError> {
        tool_result_text(body, PROVIDER, Some(FLAG))
    }

    #[test]
    fn the_answer_is_found_among_other_stream_events() {
        let body = concat!(
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\"}}\n\n",
            "event: message\n",
            "data: {\"result\":{\"_meta\":{\"provider/usage\":{\"sku\":\"search\"}},",
            "\"content\":[{\"type\":\"text\",\"text\":\"payload\"}]}}\n\n",
        );
        assert_eq!(text(body).unwrap(), "payload");
    }

    #[test]
    fn a_plain_json_body_is_read_without_stream_framing() {
        assert_eq!(text(r#"{"result":{"content":[{"type":"text","text":"payload"}]}}"#).unwrap(), "payload");
    }

    #[test]
    fn a_throttled_success_response_is_a_rate_limit_only_under_the_providers_flag() {
        let throttled =
            r#"data: {"result":{"_meta":{"ai.exa/rateLimited":true},"content":[{"type":"text","text":"Slow down"}]}}"#;
        assert_eq!(text(throttled).unwrap_err(), SearchError::RateLimited(PROVIDER));
        assert_eq!(tool_result_text(throttled, PROVIDER, None).unwrap(), "Slow down");

        let other_flag = r#"{"result":{"_meta":{"other/rateLimited":true},"content":[{"type":"text","text":"ok"}]}}"#;
        assert_eq!(text(other_flag).unwrap(), "ok");
        let cleared = r#"{"result":{"_meta":{"ai.exa/rateLimited":false},"content":[{"type":"text","text":"ok"}]}}"#;
        assert_eq!(text(cleared).unwrap(), "ok");
    }

    #[test]
    fn protocol_and_tool_errors_carry_the_providers_message() {
        let protocol = r#"{"error":{"code":-32602,"message":"Unknown tool"}}"#;
        assert_eq!(text(protocol).unwrap_err().to_string(), "Exa rejected the search: Unknown tool");

        let tool = r#"{"result":{"isError":true,"content":[{"type":"text","text":"Query is empty"}]}}"#;
        assert_eq!(text(tool).unwrap_err().to_string(), "Exa rejected the search: Query is empty");
    }

    #[test]
    fn a_provider_page_is_its_labels_and_the_text_after_them() {
        let exa = "# Rust Versions | Rust Changelogs\nURL: http://releases.rs\n\nRust Versions\n\n- Stable: 1.100.0\n";
        assert_eq!(
            labelled_page(exa, "# "),
            Some(PageContent {
                title: "Rust Versions | Rust Changelogs".into(),
                text: "Rust Versions\n\n- Stable: 1.100.0".into(),
                published_at: None,
                cut_short: false,
            })
        );
        let keenable = "Title: Rust\nURL: https://example.com/rust\nPublished: 2026-10-23\n\
                        Truncated: showing the first 1997 of 42880 characters. Raise max_chars to get more.\n\n\
                        Rust is a language...";
        assert_eq!(
            labelled_page(keenable, "Title:"),
            Some(PageContent {
                title: "Rust".into(),
                text: "Rust is a language...".into(),
                published_at: Some("2026-10-23".into()),
                cut_short: true,
            })
        );
        for unrecognized in
            ["", "Just some text", "A page\n\nwith no address line", "Title: Empty\nURL: https://a.example/\n\n \n"]
        {
            assert_eq!(labelled_page(unrecognized, "Title:"), None, "{unrecognized:?}");
        }
    }

    #[test]
    fn a_body_without_an_answer_is_a_failure() {
        for unreadable in ["<html>gateway error</html>", "", "{}"] {
            assert_eq!(text(unreadable).unwrap_err().to_string(), "Exa search failed: unreadable response");
        }
    }
}
