use std::{sync::Arc, time::Instant};

use uzu::types::session::chat::ChatReplyStats;

use crate::server::{
    access_log::{AccessChatRequest, AccessChatResponse, AccessLog, AccessRecord, AccessSpeculatorStats, now_ts},
    chat_completions::{ChatCompletionRequest, OaiMessage},
    request_info::RequestInfo,
};

const PREVIEW_MAX_CHARS: usize = 48;

pub(crate) fn preview(
    messages: &[OaiMessage],
    max_chars: usize,
) -> String {
    let text = messages
        .iter()
        .rev()
        .find(|message| {
            message.role == "user" && message.content.as_deref().is_some_and(|content| !content.trim().is_empty())
        })
        .or_else(|| {
            messages
                .iter()
                .rev()
                .find(|message| message.content.as_deref().is_some_and(|content| !content.trim().is_empty()))
        })
        .and_then(|message| message.content.as_deref())
        .unwrap_or("");
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > max_chars {
        format!("{}…", collapsed.chars().take(max_chars).collect::<String>())
    } else {
        collapsed
    }
}

/// Per-request tracing: one event when the request arrives, one when it ends,
/// correlated by a short request-id tag added by the log formatter. The same
/// lifecycle appends one JSONL record to the access journal (see `AccessLog`).
pub struct RequestLog {
    started: Instant,
    access: Arc<AccessLog>,
    record: AccessRecord,
}

impl RequestLog {
    pub fn start(
        request: &ChatCompletionRequest,
        model: &str,
        stream: bool,
        effort: &str,
        request_info: &RequestInfo,
        access: &Arc<AccessLog>,
    ) -> Self {
        let mode = if stream { "stream" } else { "blocking" };
        let images = request.messages.iter().map(|m| m.images.len() + m.image_urls.len()).sum::<usize>();
        let tools = request.tools.as_ref().map_or(0, Vec::len);
        tracing::info!(
            "▶ {mode} | {} msgs | {images} img | {tools} tools | effort={effort} | \"{}\"",
            request.messages.len(),
            preview(&request.messages, PREVIEW_MAX_CHARS),
        );
        access.mark_started();
        let chat_request = AccessChatRequest {
            stream,
            messages: request.messages.len(),
            preview: preview(&request.messages, crate::server::access_log::JOURNAL_PREVIEW_CHARS),
            images,
            tools,
            effort: effort.to_string(),
            model: model.to_string(),
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            repetition_penalty: request.repetition_penalty,
            seed: request.seed,
            max_tokens: request.max_completion_tokens.or(request.max_tokens),
            response_format: request
                .response_format
                .as_ref()
                .and_then(|value| value.get("type"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            session_id: request.session_id.clone(),
        };
        Self {
            started: Instant::now(),
            access: Arc::clone(access),
            record: AccessRecord {
                ts: now_ts(),
                request_id: request_info.id.clone(),
                method: request_info.method.as_str().to_string(),
                path: request_info.uri.clone(),
                status: None,
                duration_ms: 0.0,
                session_id: request.session_id.clone().or_else(|| request_info.session_id.clone()),
                error: None,
                request: Some(chat_request),
                response: None,
            },
        }
    }

    pub fn finish(
        &self,
        finish_reason: &str,
        stats: Option<&ChatReplyStats>,
        notes: Vec<String>,
    ) {
        let mut parts = vec![format!("✓ {:.2}s", self.started.elapsed().as_secs_f64())];
        if let Some(stats) = stats {
            if let Some(ttft) = stats.time_to_first_token {
                parts.push(format!("ttft {ttft:.2}s"));
            }
            let cached = stats.tokens_count_input_cached.unwrap_or(0);
            let prefilled = stats.tokens_count_input.unwrap_or(0);
            parts.push(format!(
                "prompt {} tok ({} cached, {} prefill @ {} tok/s)",
                cached + prefilled,
                cached,
                prefilled,
                stats.prefill_tokens_per_second.map_or("-".to_string(), |tps| format!("{tps:.1}")),
            ));
            if let Some(generated) = stats.tokens_count_output {
                parts.push(format!(
                    "out {generated} tok @ {} tok/s",
                    stats.generate_tokens_per_second.map_or("-".to_string(), |tps| format!("{tps:.1}")),
                ));
            }
            if let Some(speculator) = &stats.speculator_stats {
                parts.push(format!(
                    "spec {:.2} tok/pass ({} passes)",
                    speculator.tokens_per_forward_pass, speculator.num_decode_forward_passes
                ));
            }
        }
        parts.push(format!("finish={finish_reason}"));
        parts.extend(notes.iter().cloned());
        tracing::info!("{}", parts.join(" | "));

        let mut record = self.record.clone();
        record.duration_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        record.response = Some(AccessChatResponse {
            finish_reason: finish_reason.to_string(),
            prompt_tokens: stats.map(|stats| {
                stats.tokens_count_input_cached.unwrap_or(0) + stats.tokens_count_input.unwrap_or(0)
            }),
            cached_tokens: stats.and_then(|stats| stats.tokens_count_input_cached),
            completion_tokens: stats.and_then(|stats| stats.tokens_count_output),
            time_to_first_token: stats.and_then(|stats| stats.time_to_first_token),
            prefill_tokens_per_second: stats.and_then(|stats| stats.prefill_tokens_per_second),
            generate_tokens_per_second: stats.and_then(|stats| stats.generate_tokens_per_second),
            speculator: stats.and_then(|stats| stats.speculator_stats.as_ref()).map(|speculator| {
                AccessSpeculatorStats {
                    tokens_per_forward_pass: speculator.tokens_per_forward_pass,
                    num_decode_forward_passes: speculator.num_decode_forward_passes,
                }
            }),
            notes,
        });
        self.access.record(&record);
    }

    pub fn fail(
        &self,
        error: &str,
    ) {
        tracing::warn!("✗ {:.2}s | failed: {error}", self.started.elapsed().as_secs_f64());
        let mut record = self.record.clone();
        record.duration_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        record.error = Some(error.to_string());
        self.access.record(&record);
    }
}
