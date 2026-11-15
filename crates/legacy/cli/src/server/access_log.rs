use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::Instant,
};

use chrono::{SecondsFormat, Utc};
use rocket::{
    Request, Response,
    fairing::{Fairing, Info, Kind},
};
use serde::Serialize;

use crate::server::{ServerState, request_info::RequestInfo};

/// Characters of the last user message kept in a journal record.
pub(crate) const JOURNAL_PREVIEW_CHARS: usize = 256;

pub fn now_ts() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Request parameters worth journaling (the OpenAI wire request, sampled down).
#[derive(Debug, Clone, Serialize)]
pub struct AccessChatRequest {
    pub stream: bool,
    pub messages: usize,
    pub preview: String,
    pub images: usize,
    pub tools: usize,
    pub effort: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Response metadata for a generation: finish reason and the tok/s family of stats.
#[derive(Debug, Clone, Serialize)]
pub struct AccessChatResponse {
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    /// Seconds until the first token (streaming latency).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_first_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculator: Option<AccessSpeculatorStats>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccessSpeculatorStats {
    pub tokens_per_forward_pass: f64,
    pub num_decode_forward_passes: u32,
}

/// One journal line: HTTP envelope plus the chat request/response payloads when
/// the route is a generation.
#[derive(Debug, Clone, Serialize)]
pub struct AccessRecord {
    /// RFC 3339 UTC timestamp with millisecond precision.
    pub ts: String,
    pub request_id: String,
    pub method: String,
    pub path: String,
    /// HTTP status when known; streaming generations commit 200 before the
    /// outcome is known, so generation records may leave this unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    pub duration_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<AccessChatRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<AccessChatResponse>,
}

/// Append-only JSONL access log, one record per HTTP request. Always alive so
/// the `/state` counters keep working; file writing is the optional part.
pub struct AccessLog {
    file: Option<Mutex<File>>,
    path: Option<PathBuf>,
    requests_started: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
}

impl AccessLog {
    pub fn open(path: PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            file: Some(Mutex::new(file)),
            path: Some(path),
            requests_started: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
        })
    }

    /// Journaling disabled (`--no-request-log`): counters still work.
    pub fn disabled() -> Self {
        Self {
            file: None,
            path: None,
            requests_started: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
        }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn mark_started(&self) {
        self.requests_started.fetch_add(1, Ordering::Relaxed);
    }

    /// (started, completed, failed) since server boot.
    pub fn counters(&self) -> (u64, u64, u64) {
        (
            self.requests_started.load(Ordering::Relaxed),
            self.requests_completed.load(Ordering::Relaxed),
            self.requests_failed.load(Ordering::Relaxed),
        )
    }

    /// Appends one JSON line. Write failures never affect the request; they
    /// surface as a warning on the console log.
    pub fn record(&self, record: &AccessRecord) {
        if record.error.is_some() || record.status.is_some_and(|status| status >= 400) {
            self.requests_failed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_completed.fetch_add(1, Ordering::Relaxed);
        }
        let Some(file) = &self.file else {
            return;
        };
        let mut line = match serde_json::to_string(record) {
            Ok(line) => line,
            Err(error) => {
                tracing::warn!("request journal serialization failed: {error}");
                return;
            },
        };
        line.push('\n');
        // Recover from a poisoned lock (a previous write panicked) rather than
        // panicking this worker thread too — the journal must not take the
        // server down with it.
        let mut file = match file.lock() {
            Ok(file) => file,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(error) = file.write_all(line.as_bytes()) {
            tracing::warn!("request journal write failed: {error}");
        }
    }
}

fn is_chat_completions(request: &Request<'_>) -> bool {
    request.uri().path().as_str().ends_with("/chat/completions")
}

fn is_internal(request: &Request<'_>) -> bool {
    request.uri().path().as_str() == "/logs"
}

struct RequestStarted(Instant);

/// HTTP-level journal for every non-generation route (`/models`, `/state`,
/// decisions, ...). `/chat/completions` keeps its richer per-generation record
/// (request parameters + tok/s stats), written by `RequestLog`.
pub struct AccessLogFairing;

#[rocket::async_trait]
impl Fairing for AccessLogFairing {
    fn info(&self) -> Info {
        Info {
            name: "Access log",
            kind: Kind::Request | Kind::Response,
        }
    }

    async fn on_request(&self, request: &mut Request<'_>, _data: &mut rocket::Data<'_>) {
        request.local_cache(|| RequestStarted(Instant::now()));
        if is_chat_completions(request) || is_internal(request) {
            return;
        }
        if let Some(state) = request.rocket().state::<ServerState>() {
            state.access_log.mark_started();
        }
    }

    async fn on_response<'r>(&self, request: &'r Request<'_>, response: &mut Response<'r>) {
        if is_chat_completions(request) || is_internal(request) {
            return;
        }
        let Some(state) = request.rocket().state::<ServerState>() else {
            return;
        };
        let started_at = request.local_cache(|| RequestStarted(Instant::now())).0;
        let request_info = request.local_cache(|| RequestInfo::new(request.method(), request.uri().to_string()));
        state.access_log.record(&AccessRecord {
            ts: now_ts(),
            request_id: request_info.id.clone(),
            method: request.method().as_str().to_string(),
            path: request_info.uri.clone(),
            status: Some(response.status().code),
            duration_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            session_id: request_info.session_id.clone(),
            error: None,
            request: None,
            response: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> AccessRecord {
        AccessRecord {
            ts: now_ts(),
            request_id: "abc123".to_string(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            status: None,
            duration_ms: 1234.5,
            session_id: Some("sess".to_string()),
            error: None,
            request: Some(AccessChatRequest {
                stream: true,
                messages: 2,
                preview: "hello".to_string(),
                images: 1,
                tools: 0,
                effort: "high".to_string(),
                model: "m".to_string(),
                temperature: Some(0.7),
                top_p: None,
                top_k: None,
                repetition_penalty: None,
                seed: None,
                max_tokens: Some(512),
                response_format: None,
                session_id: None,
            }),
            response: Some(AccessChatResponse {
                finish_reason: "stop".to_string(),
                prompt_tokens: Some(100),
                cached_tokens: Some(64),
                completion_tokens: Some(50),
                time_to_first_token: Some(0.12),
                prefill_tokens_per_second: Some(900.0),
                generate_tokens_per_second: Some(24.5),
                speculator: None,
                notes: Vec::new(),
            }),
        }
    }

    #[test]
    fn writes_jsonl_records() {
        let dir = std::env::temp_dir().join(format!("uzu-access-log-test-{}", std::process::id()));
        let path = dir.join("requests.jsonl");
        let log = AccessLog::open(path.clone()).unwrap();
        log.mark_started();
        log.record(&sample_record());
        let failed = AccessRecord {
            error: Some("boom".to_string()),
            ..sample_record()
        };
        log.record(&failed);

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<serde_json::Value> = content
            .lines()
            .map(|line| serde_json::from_str(line).expect("each line is a JSON object"))
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["request_id"], "abc123");
        assert_eq!(lines[0]["request"]["preview"], "hello");
        assert_eq!(lines[0]["response"]["cached_tokens"], 64);
        // None fields stay out of the line.
        assert!(lines[0]["status"].is_null());
        assert_eq!(lines[1]["error"], "boom");
        let counters = log.counters();
        assert_eq!(counters, (1, 1, 1));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn disabled_journal_keeps_counters_but_no_file() {
        let log = AccessLog::disabled();
        log.record(&sample_record());
        assert!(log.path().is_none());
        assert_eq!(log.counters(), (0, 1, 0));
    }
}
