use std::sync::Arc;

use cli::server::{
    handle_chat_completions,
    state::{RunSession, SessionState, SessionWrapper},
};
use rocket::{http::Status, local::asynchronous::Client, routes};
use serde_json::{Value, json};
use uzu::session::{
    config::RunConfig,
    types::{Error, FinishReason, Input, Output, ParsedText, RunStats, Stats, StepStats, Text, TotalStats},
};

struct MockSession {
    result: Result<String, Error>,
}

impl RunSession for MockSession {
    fn run(
        &mut self,
        _input: Input,
        _config: RunConfig,
        _progress: Option<Box<dyn Fn(Output) -> bool>>,
    ) -> Result<Output, Error> {
        self.result.as_ref().map_err(|_| Error::GenerateFailed).map(|text| Output {
            text: Text {
                original: text.clone(),
                parsed: ParsedText {
                    chain_of_thought: None,
                    response: Some(text.clone()),
                },
            },
            finish_reason: Some(FinishReason::Stop),
            stats: Stats {
                prefill_stats: StepStats {
                    duration: 0.0,
                    suffix_length: 0,
                    tokens_count: 0,
                    tokens_per_second: 0.0,
                    processed_tokens_per_second: 0.0,
                    model_run: RunStats {
                        count: 0,
                        average_duration: 0.0,
                    },
                    run: None,
                },
                generate_stats: None,
                total_stats: TotalStats {
                    duration: 0.0,
                    tokens_count_input: 0,
                    tokens_count_output: 0,
                },
            },
        })
    }
}

async fn make_client(mock: MockSession) -> Client {
    let state = SessionState {
        model_name: "test-model".to_string(),
        session_wrapper: Arc::new(SessionWrapper::new(mock)),
    };
    let rocket = rocket::build().manage(state).mount("/v1", routes![handle_chat_completions]);
    Client::tracked(rocket).await.expect("valid rocket instance")
}

#[rocket::async_test]
async fn test_non_streaming_success() {
    let client = make_client(MockSession {
        result: Ok("Hello, world!".to_string()),
    })
    .await;

    let response = client
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .dispatch()
        .await;

    assert_eq!(response.status(), Status::Ok);
    let body: Value = response.into_json().await.expect("valid json");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello, world!");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["model"], "test-model");
}

#[rocket::async_test]
async fn test_non_streaming_session_error() {
    let client = make_client(MockSession {
        result: Err(Error::GenerateFailed),
    })
    .await;

    let response = client
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .dispatch()
        .await;

    assert_eq!(response.status(), Status::Ok);
    let body: Value = response.into_json().await.expect("valid json");
    assert_eq!(body["choices"][0]["finish_reason"], "error");
    assert!(
        body["choices"][0]["message"]["content"].as_str().unwrap().contains("Generate failed"),
        "expected error text in content"
    );
}

#[rocket::async_test]
async fn test_streaming_content_type() {
    let client = make_client(MockSession {
        result: Ok("Stream me".to_string()),
    })
    .await;

    let response = client
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .dispatch()
        .await;

    assert_eq!(response.status(), Status::Ok);
    assert_eq!(response.content_type().map(|ct| ct.to_string()), Some("text/event-stream".to_string()));
}

#[rocket::async_test]
async fn test_streaming_done_sentinel() {
    let client = make_client(MockSession {
        result: Ok("Stream me".to_string()),
    })
    .await;

    let body = client
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .dispatch()
        .await
        .into_string()
        .await
        .expect("body");

    assert!(body.contains("data: [DONE]"), "missing [DONE] sentinel:\n{body}");

    for line in body.lines().filter(|l| !l.is_empty()) {
        assert!(line.starts_with("data: "), "unexpected SSE line: {line}");
    }
}

#[rocket::async_test]
async fn test_streaming_error_in_stream() {
    let client = make_client(MockSession {
        result: Err(Error::GenerateFailed),
    })
    .await;

    let body = client
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .dispatch()
        .await
        .into_string()
        .await
        .expect("body");

    assert!(body.contains("Generate failed"), "expected error text in stream:\n{body}");
    assert!(body.contains(r#""finish_reason":"error""#), "expected finish_reason error:\n{body}");
    assert!(body.contains("data: [DONE]"), "stream must still terminate:\n{body}");
}

/// Builds a string of exactly `prefix_bytes` ASCII bytes followed by a
/// 2-byte UTF-8 character (Cyrillic 'а', U+0430), so that a byte slice at
/// `prefix_bytes` would cut the character in half.
fn utf8_trap(prefix_bytes: usize) -> String {
    let mut s = "a".repeat(prefix_bytes);
    s.push('а'); // U+0430: 0xD0 0xB0 — 2 bytes
    s
}

#[rocket::async_test]
async fn test_request_content_utf8_boundary_does_not_panic() {
    // Content is 10001 bytes; byte 10000 is the second byte of 'а'.
    // &content[..10000] will panic with a byte-boundary assertion.
    let content = utf8_trap(9999);
    assert_eq!(content.len(), 10001);

    let client = make_client(MockSession {
        result: Ok("ok".to_string()),
    })
    .await;
    let response = client
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": content}],
            "stream": false
        }))
        .dispatch()
        .await;

    assert_eq!(response.status(), Status::Ok, "handler must not panic on multibyte content");
}

#[rocket::async_test]
async fn test_response_text_utf8_boundary_does_not_panic() {
    // Session returns 3001-byte text where byte 3000 is mid-char.
    let text = utf8_trap(2999);
    assert_eq!(text.len(), 3001);

    let client = make_client(MockSession {
        result: Ok(text),
    })
    .await;
    let response = client
        .post("/v1/chat/completions")
        .json(&json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .dispatch()
        .await;

    assert_eq!(response.status(), Status::Ok, "handler must not panic on multibyte response");
}
