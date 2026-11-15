//! Raw completion API: direct autoregressive inference over the resident
//! model, bypassing every template/rendering/parsing layer.
//!
//! `POST /v1/completions` takes a `prompt` that is tokenized verbatim (or
//! literal `raw_tokens`) and prefilled as-is; the reply is the model's raw
//! continuation — no chat template, no reasoning/tool-call parsing. This is
//! the observability surface for model-behavior work (template contracts,
//! sampling edge cases, quantization audits): the same prompt bytes always
//! reach the model, and the response shows exactly what came back.
//!
//! Slot semantics mirror chat: the `model` field must match the resident
//! model (409 otherwise), the session mutex is held for the whole request so
//! raw generation never overlaps a chat generation on the GPU, and each
//! request runs on a fresh KV state (VLM states come from the pipeline pool;
//! text models build an empty state from the language model directly).

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use rocket::{
    Request, State,
    futures::Stream,
    http::Status,
    post,
    response::{
        Responder, status,
        stream::{Event, EventStream},
    },
    serde::json::Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uzu_engine::{
    backends::metal::Metal,
    engine::language_model::{LanguageModel, raw::{raw_decode_step, raw_prefill}, state::LanguageModelState},
    vision::{AnyVlmPipeline, StreamingTokenDecoder, VlmSamplingConfig, VlmSampler},
};

use crate::server::{SlotSnapshot, model_slot::LoadedModel, request_info::RequestInfo};
use crate::server::chat_completions::{ModelNotLoadedBody, model_not_loaded_body, oai_error_body};

/// Live SSE emission state for a streamed raw completion: chunk events carry
/// decoded deltas as the generation loop produces them (the loop runs on a
/// worker thread; the channel feeds the response body concurrently).
struct RawStreamOut {
    tx: mpsc::UnboundedSender<Event>,
    id: String,
    model: String,
    created: i64,
}

fn emit_chunk(
    stream: &RawStreamOut,
    text: &str,
) {
    let _ = stream.tx.send(Event::data(
        serde_json::to_string(&serde_json::json!({
            "id": stream.id,
            "object": "text_completion.chunk",
            "created": stream.created,
            "model": stream.model,
            "choices": [{"index": 0, "text": text, "finish_reason": null}],
        }))
        .unwrap_or_default(),
    ));
}

#[derive(Deserialize)]
pub struct RawCompletionRequest {
    /// Must match the resident model (same 409 contract as chat).
    pub model: String,
    /// Prompt text, tokenized verbatim — no template is applied.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Token ids to prefill directly, bypassing the tokenizer entirely
    /// (useful for probing tokenizer boundaries and special tokens).
    #[serde(default)]
    pub raw_tokens: Option<Vec<u64>>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    /// Not supported on the raw path (no repetition context ring); rejected.
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Literal sequences that end generation. Non-streaming responses are
    /// truncated exactly before the match; streaming may already have emitted
    /// part of a match as deltas before detection.
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> u32 {
    64
}

#[derive(Serialize)]
pub struct RawChoice {
    text: String,
    finish_reason: String,
}

#[derive(Serialize)]
pub struct RawUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Serialize)]
pub struct RawCompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<RawChoice>,
    usage: RawUsage,
}

pub enum RawCompletionResult {
    Json(Status, Json<RawCompletionResponse>),
    Error(status::Custom<Json<crate::server::error::OaiErrorResponse>>),
    ModelNotLoaded(status::Custom<Json<ModelNotLoadedBody>>),
    Stream(EventStream<Pin<Box<dyn Stream<Item = Event> + Send>>>),
}

impl<'r> Responder<'r, 'r> for RawCompletionResult {
    fn respond_to(self, request: &'r Request<'_>) -> rocket::response::Result<'r> {
        match self {
            RawCompletionResult::Json(status, body) => (status, body).respond_to(request),
            RawCompletionResult::Error(error) => error.respond_to(request),
            RawCompletionResult::ModelNotLoaded(error) => error.respond_to(request),
            RawCompletionResult::Stream(stream) => stream.respond_to(request),
        }
    }
}

fn invalid(param: &str, message: String) -> RawCompletionResult {
    RawCompletionResult::Error(status::Custom(
        Status::BadRequest,
        Json(crate::server::error::OaiErrorResponse {
            error: crate::server::error::OaiError {
                message,
                kind: "invalid_request_error".to_string(),
                param: Some(param.to_string()),
                code: Some("invalid_request".to_string()),
            },
        }),
    ))
}

#[post("/completions", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_raw_completions(
    body: rocket::serde::json::Json<RawCompletionRequest>,
    chat: &State<crate::server::ChatState>,
    request_info: &RequestInfo,
) -> RawCompletionResult {
    let request = body.into_inner();

    // Slot gate, same contract as chat.
    let loaded: Arc<LoadedModel> = match chat.slot.snapshot().await {
        SlotSnapshot::Ready(loaded) => loaded,
        other => {
            let detail = match &other {
                SlotSnapshot::Loading { model, .. } => format!("model '{model}' is loading"),
                SlotSnapshot::Failed { model, error, .. } => format!("model '{model}' failed to load: {error}"),
                _ => "no model is loaded".to_string(),
            };
            return RawCompletionResult::ModelNotLoaded(status::Custom(
                Status::Conflict,
                Json(model_not_loaded_body(Some(&request.model), other.model_name(), &detail)),
            ));
        },
    };
    if request.model != loaded.model_name {
        let detail = format!(
            "requested model '{}' is not resident (current: '{}')",
            request.model, loaded.model_name
        );
        return RawCompletionResult::ModelNotLoaded(status::Custom(
            Status::Conflict,
            Json(model_not_loaded_body(Some(&request.model), Some(&loaded.model_name), &detail)),
        ));
    }

    // Prompt: literal tokens or verbatim text.
    let prompt_tokens: Vec<u64> = if let Some(tokens) = &request.raw_tokens {
        tokens.clone()
    } else {
        let Some(prompt) = request.prompt.as_deref() else {
            return invalid("prompt", "either 'prompt' or 'raw_tokens' is required".to_string());
        };
        match loaded.lm.tokenizer().encode(prompt, false) {
            Ok(encoding) => encoding.get_ids().iter().map(|&t| t as u64).collect(),
            Err(error) => return invalid("prompt", format!("failed to tokenize prompt: {error}")),
        }
    };
    if prompt_tokens.is_empty() {
        return invalid("prompt", "prompt is empty".to_string());
    }

    if request.repetition_penalty.is_some() {
        return invalid(
            "repetition_penalty",
            "raw completions do not support repetition_penalty".to_string(),
        );
    }
    if let Some(temperature) = request.temperature
        && !(0.0..=2.0).contains(&temperature)
    {
        return invalid("temperature", format!("temperature must be between 0 and 2, got {temperature}"));
    }

    let id = request_info.id.as_str().to_owned();
    let created = request_info.created_at;
    let model = loaded.model_name.clone();

    let vlm_arc = loaded.vlm_pipeline.clone();
    let mut sampler = VlmSampler::new(VlmSamplingConfig {
        temperature: request.temperature.map(|t| t as f32),
        min_p: None,
        top_p: request.top_p.map(|p| p as f32),
        top_k: request.top_k.map(|k| k.max(0) as usize),
        repetition_penalty: None,
        seed: request.seed,
        grammar: None,
        vocab_limit: None,
    });
    sampler.set_vocab_limit(loaded.lm.tokenizer().get_vocab_size(true));

    let stop_sequences = request.stop.clone().unwrap_or_default();
    let max_tokens = request.max_tokens.max(1) as usize;

    if request.stream {
        // Live streaming: the SSE body returns before the first token, and
        // chunk deltas flow through the channel as the decode loop produces
        // them. The session lock moves into the spawned task, so raw
        // generation still never overlaps a chat generation on the GPU.
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        let session = Arc::clone(&loaded.session);
        let lm = Arc::clone(&loaded.lm);
        let stream_out = RawStreamOut {
            tx: sender.clone(),
            id: id.clone(),
            model: model.clone(),
            created,
        };
        let stop = stop_sequences.clone();
        let prompt = prompt_tokens;
        let prompt_len = prompt.len();
        let seed = request.seed;
        tokio::spawn(async move {
            // A client may vanish while queued behind the session lock; don't
            // generate for it (same contract as the chat streaming path).
            let mut lock = std::pin::pin!(session.lock());
            let _guard = tokio::select! {
                guard = &mut lock => guard,
                () = sender.closed() => return,
            };
            let result =
                run_raw_generation(vlm_arc.as_deref(), &lm, &prompt, &mut sampler, max_tokens, &stop, seed, Some(stream_out))
                    .await;
            let (tokens, finish, errored) = match result {
                Ok(result) => result,
                Err(error) => {
                    tracing::error!("raw completion failed: {error}");
                    let body = oai_error_body(Status::InternalServerError, "backend_error", &format!("raw completion failed: {error}"));
                    let _ = sender.send(Event::data(serde_json::to_string(&body).unwrap_or_default()));
                    let _ = sender.send(Event::data("[DONE]"));
                    return;
                },
            };
            if errored {
                let body = oai_error_body(Status::InternalServerError, "backend_error", "raw completion failed mid-generation");
                let _ = sender.send(Event::data(serde_json::to_string(&body).unwrap_or_default()));
                let _ = sender.send(Event::data("[DONE]"));
                return;
            }
            let usage = serde_json::json!({
                "prompt_tokens": prompt_len,
                "completion_tokens": tokens.len(),
            });
            // Standard SSE closing frame: empty text + finish reason. The
            // delta that completes a stop sequence is not emitted, but deltas
            // before it may already show part of the match (same as before).
            let _ = sender.send(Event::data(
                serde_json::to_string(&serde_json::json!({
                    "id": id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": "", "finish_reason": finish}],
                    "usage": usage,
                }))
                .unwrap_or_default(),
            ));
            let _ = sender.send(Event::data("[DONE]"));
        });
        let body: Pin<Box<dyn Stream<Item = Event> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
        return RawCompletionResult::Stream(EventStream::from(body));
    }

    // Mutually exclusive with chat generation for the whole request.
    let session_guard = loaded.session.lock().await;

    let generated = run_raw_generation(
        vlm_arc.as_deref(),
        &loaded.lm,
        &prompt_tokens,
        &mut sampler,
        max_tokens,
        &stop_sequences,
        request.seed,
        None,
    )
    .await;

    drop(session_guard);

    let (generated_tokens, finish_reason, errored) = match generated {
        Ok(result) => result,
        Err(error) => {
            tracing::error!("raw completion failed: {error}");
            return RawCompletionResult::Error(status::Custom(
                Status::InternalServerError,
                Json(crate::server::error::OaiErrorResponse {
                    error: crate::server::error::OaiError {
                        message: format!("raw completion failed: {error}"),
                        kind: "backend_error".to_string(),
                        param: None,
                        code: Some("backend_error".to_string()),
                    },
                }),
            ));
        },
    };
    if errored {
        return RawCompletionResult::Error(status::Custom(
            Status::InternalServerError,
            Json(crate::server::error::OaiErrorResponse {
                error: crate::server::error::OaiError {
                    message: "raw completion failed mid-generation".to_string(),
                    kind: "backend_error".to_string(),
                    param: None,
                    code: Some("backend_error".to_string()),
                },
            }),
        ));
    }

    // Decode the full generation in one pass, then apply stop-sequence
    // truncation on the complete text (exact for non-streaming; streaming
    // below emits per-token deltas and may show part of a match).
    let mut text = loaded
        .lm
        .tokenizer()
        .decode(&generated_tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(), false)
        .unwrap_or_default();
    let mut finish = finish_reason;
    for sequence in &stop_sequences {
        if let Some(index) = text.find(sequence.as_str()) {
            text.truncate(index);
            finish = "stop_sequence".to_string();
            break;
        }
    }

    RawCompletionResult::Json(
        Status::Ok,
        Json(RawCompletionResponse {
            id,
            object: "text_completion",
            created,
            model,
            choices: vec![RawChoice {
                text,
                finish_reason: finish,
            }],
            usage: RawUsage {
                prompt_tokens: prompt_tokens.len() as u64,
                completion_tokens: generated_tokens.len() as u64,
            },
        }),
    )
}

/// The raw generation loop: prefill, then decode until EOS, a stop sequence,
/// or the token limit. When `stream` is set, every decoded delta is emitted
/// through the channel as it is produced (live SSE). Returns the generated
/// ids (prompt excluded), the finish reason, and an errored flag (a decode
/// failure leaves the text that did complete in `generated_tokens`).
#[allow(clippy::too_many_arguments)]
async fn run_raw_generation(
    vlm: Option<&AnyVlmPipeline>,
    lm: &Arc<LanguageModel<Metal>>,
    prompt_tokens: &[u64],
    sampler: &mut VlmSampler,
    max_tokens: usize,
    stop_sequences: &[String],
    seed: Option<u64>,
    stream: Option<RawStreamOut>,
) -> anyhow::Result<(Vec<u64>, String, bool)> {
    let started = Instant::now();
    // GPU work blocks the executor thread; keep Rocket's other connection
    // tasks off this worker only while the synchronous section runs.
    let result = tokio::task::block_in_place(|| {
        // Fresh KV state per request: VLM states come from the pipeline pool;
        // text models build one from the language model directly. Either way
        // nothing leaks into the chat session's state.
        // Capacity is a hint only: `max_tokens` is client-controlled and the
        // loop stops at EOS long before it in practice, so never pre-allocate
        // the full request ceiling.
        let mut generated: Vec<u64> = Vec::with_capacity(max_tokens.min(32 * 1024));
        let mut curr_token;
        let mut finish = "stop".to_string();
        let mut errored = false;

        // `state_guard` keeps the borrowed state alive for the loop below.
        let mut state_guard;
        let mut text_so_far = String::new();
        let mut decoder = None;

        enum RawState {
            Vlm(uzu_engine::vision::VlmState),
            Text(LanguageModelState<Metal>),
        }

        let first_token;
        let pipeline = vlm;
        match pipeline {
            Some(pipeline) => {
                let mut state = pipeline.acquire_state(lm, None)?;
                curr_token = pipeline
                    .prefill_incremental_text(lm, &mut state, prompt_tokens, sampler)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                first_token = curr_token;
                state_guard = RawState::Vlm(state);
            },
            None => {
                let mut state = lm
                    .create_empty_state(None, seed.unwrap_or(0))
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                curr_token = raw_prefill(lm, &mut state, prompt_tokens, sampler).map_err(|e| anyhow::anyhow!("{e}"))?;
                first_token = curr_token;
                state_guard = RawState::Text(state);
            },
        }
        if stream.is_some() || !stop_sequences.is_empty() {
            decoder = Some(StreamingTokenDecoder::new(lm.tokenizer(), false));
        }
        let mut first_chunk = None;
        let mut stop_hit = false;
        if let Some(decoder) = decoder.as_mut()
            && let Ok(Some(chunk)) = decoder.step(first_token as u32)
        {
            text_so_far.push_str(&chunk);
            first_chunk = Some(chunk);
            stop_hit = stop_sequences.iter().any(|sequence| text_so_far.contains(sequence.as_str()));
        }
        if stop_hit {
            finish = "stop_sequence".to_string();
        } else {
            if let (Some(stream), Some(chunk)) = (stream.as_ref(), first_chunk.as_deref()) {
                emit_chunk(stream, chunk);
            }
            generated.push(first_token);
            let mut hit_stop = false;
            while generated.len() < max_tokens {
                let next = match (&pipeline, &mut state_guard) {
                    (Some(pipeline), RawState::Vlm(state)) => pipeline
                        .decode_step_with_state(lm, state, curr_token, sampler)
                        .map_err(|e| anyhow::anyhow!("{e}")),
                    (None, RawState::Text(state)) => raw_decode_step(lm, state, curr_token, sampler).map_err(|e| anyhow::anyhow!("{e}")),
                    _ => unreachable!("state variant matches pipeline presence"),
                };
                match next {
                    Ok(token) => {
                        curr_token = token;
                        if is_eos(lm, token) {
                            break;
                        }
                        let mut completed_stop = false;
                        let mut decoded_chunk = None;
                        if let Some(decoder) = decoder.as_mut()
                            && let Ok(Some(decoded)) = decoder.step(token as u32)
                        {
                            text_so_far.push_str(&decoded);
                            decoded_chunk = Some(decoded);
                            completed_stop =
                                stop_sequences.iter().any(|sequence| text_so_far.contains(sequence.as_str()));
                        }
                        if completed_stop {
                            hit_stop = true;
                        } else if let (Some(stream), Some(decoded)) = (stream.as_ref(), decoded_chunk.as_deref()) {
                            emit_chunk(stream, decoded);
                        }
                        generated.push(token);
                        if hit_stop {
                            finish = "stop_sequence".to_string();
                            break;
                        }
                    },
                    Err(error) => {
                        tracing::error!("raw decode step failed: {error}");
                        errored = true;
                        break;
                    },
                }
            }
            if generated.len() >= max_tokens && finish == "stop" && !errored {
                finish = "length".to_string();
            }
        }
        Ok((generated, finish.to_string(), errored))
    });
    tracing::info!(
        "⚙️ raw completion: {} prompt tok, {} generated in {:.2}s",
        prompt_tokens.len(),
        result.as_ref().map(|(tokens, _, _)| tokens.len()).unwrap_or(0),
        started.elapsed().as_secs_f64()
    );
    result
}

/// The model's end-of-turn tokens (from the config's generation settings).
fn is_eos(lm: &LanguageModel<Metal>, token: u64) -> bool {
    lm.generation_config().stop_token_ids.contains(&token)
}
