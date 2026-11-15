use std::pin::Pin;
use std::sync::Arc;

use base64::Engine as _;
use rocket::form::{Form, FromForm};
use rocket::fs::TempFile;
use rocket::http::{ContentType, Status};
use rocket::request::Request;
use rocket::response::stream::{Event, EventStream};
use rocket::response::{Responder, status};
use rocket::serde::json::Json;
use rocket::{State, get, post};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt as _;
use uzu_engine::audio::lfm3_audio::pipeline::GenerationMode;

use crate::common::audio::{AudioGenerateOptions, AudioRuntime, DEFAULT_ASR_SYSTEM_PROMPT, DEFAULT_TTS_SYSTEM_PROMPT};
use crate::server::chat_completions::{ModelNotLoadedBody, model_not_loaded_body};
use crate::server::error::{OaiError, OaiErrorResponse};
use crate::server::model_slot::LoadedModel;
use crate::server::request_info::RequestInfo;
use crate::server::{ChatState, SlotSnapshot};

pub enum AudioApiResult {
    Json(Status, serde_json::Value),
    Text(Status, String),
    Wav(Vec<u8>),
    Pcm(Vec<u8>),
    Stream(EventStream<Pin<Box<dyn futures_util::Stream<Item = Event> + Send>>>),
    Error(status::Custom<Json<OaiErrorResponse>>),
    ModelNotLoaded(status::Custom<Json<ModelNotLoadedBody>>),
}

impl<'r> Responder<'r, 'r> for AudioApiResult {
    fn respond_to(self, request: &'r Request<'_>) -> rocket::response::Result<'r> {
        match self {
            AudioApiResult::Json(status, body) => (status, Json(body)).respond_to(request),
            AudioApiResult::Text(status, text) => (status, (ContentType::Plain, text)).respond_to(request),
            AudioApiResult::Wav(bytes) => (Status::Ok, (ContentType::WAV, bytes)).respond_to(request),
            AudioApiResult::Pcm(bytes) => (Status::Ok, (ContentType::Binary, bytes)).respond_to(request),
            AudioApiResult::Stream(stream) => stream.respond_to(request),
            AudioApiResult::Error(err) => err.respond_to(request),
            AudioApiResult::ModelNotLoaded(err) => err.respond_to(request),
        }
    }
}

fn api_error(status: Status, param: Option<&str>, kind: &str, code: &str, message: impl Into<String>) -> AudioApiResult {
    AudioApiResult::Error(status::Custom(
        status,
        Json(OaiErrorResponse {
            error: OaiError {
                message: message.into(),
                kind: kind.to_string(),
                param: param.map(str::to_string),
                code: Some(code.to_string()),
            },
        }),
    ))
}

async fn resolve_audio_model(
    chat: &ChatState,
    requested_model: Option<&str>,
) -> Result<(Arc<LoadedModel>, Arc<tokio::sync::Mutex<AudioRuntime>>), AudioApiResult> {
    let loaded = match chat.slot.snapshot().await {
        SlotSnapshot::Ready(loaded) => loaded,
        other => {
            let detail = match &other {
                SlotSnapshot::Loading { model, .. } => format!("model '{model}' is loading"),
                SlotSnapshot::Failed { model, error, .. } => format!("model '{model}' failed to load: {error}"),
                _ => "no model is loaded".to_string(),
            };
            return Err(AudioApiResult::ModelNotLoaded(status::Custom(
                Status::Conflict,
                Json(model_not_loaded_body(requested_model, other.model_name(), &detail)),
            )));
        },
    };
    if let Some(req_model) = requested_model
        && !req_model.is_empty() && req_model != loaded.model_name {
            return Err(api_error(
                Status::NotFound,
                Some("model"),
                "invalid_request_error",
                "model_not_found",
                format!("model '{req_model}' is not loaded (resident: '{}')", loaded.model_name),
            ));
        }
    let Some(audio) = loaded.audio.as_ref().map(Arc::clone) else {
        return Err(api_error(
            Status::BadRequest,
            Some("model"),
            "invalid_request_error",
            "audio_not_supported",
            format!("resident model '{}' does not support audio inference", loaded.model_name),
        ));
    };
    Ok((loaded, audio))
}

#[derive(FromForm)]
pub struct AudioTranscriptionForm<'r> {
    file: TempFile<'r>,
    model: Option<String>,
    prompt: Option<String>,
    system_prompt: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    seed: Option<i64>,
}

#[derive(Deserialize)]
pub struct AudioTranscriptionJsonRequest {
    pub model: Option<String>,
    pub audio_base64: Option<String>,
    pub audio_path: Option<String>,
    pub prompt: Option<String>,
    pub system_prompt: Option<String>,
    pub language: Option<String>,
    pub response_format: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<i64>,
}

#[derive(Deserialize)]
pub struct AudioSpeechRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub language: Option<String>,
    pub system_prompt: Option<String>,
    pub response_format: Option<String>,
    pub stream: Option<bool>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub seed: Option<i64>,
    pub audio_temperature: Option<f32>,
    pub audio_top_k: Option<usize>,
    pub audio_repetition_penalty: Option<f32>,
    pub audio_repetition_window: Option<usize>,
}

#[derive(Deserialize)]
pub struct AudioTranslateRequest {
    pub model: Option<String>,
    pub audio_base64: Option<String>,
    pub audio_path: Option<String>,
    pub text: Option<String>,
    pub source_language: String,
    pub target_language: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<i64>,
}

#[derive(Deserialize)]
pub struct LiveAsrStartRequest {
    pub model: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Deserialize)]
pub struct LiveAsrChunkRequest {
    pub model: Option<String>,
    pub session_id: String,
    pub audio_base64: Option<String>,
    pub format: Option<String>,
    pub sample_rate: Option<u32>,
    pub is_final: Option<bool>,
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub seed: Option<i64>,
}

#[derive(Deserialize)]
pub struct LiveAsrStopRequest {
    pub model: Option<String>,
    pub session_id: String,
}

#[derive(Serialize)]
struct AudioTiming {
    prep_ms: f64,
    prefill_ms: f64,
    depth_ms: f64,
    backbone_ms: f64,
    codec_ms: f64,
}

#[allow(clippy::too_many_arguments)]
async fn run_transcription(
    chat: &ChatState,
    model: Option<String>,
    wav_bytes: Vec<u8>,
    prompt: Option<String>,
    system_prompt: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    seed: Option<i64>,
) -> AudioApiResult {
    let (loaded, audio) = match resolve_audio_model(chat, model.as_deref()).await {
        Ok(pair) => pair,
        Err(err) => return err,
    };
    let sys = system_prompt.or_else(|| {
        language.as_deref().and_then(|l| {
            let trimmed = l.trim();
            if trimmed.is_empty() {
                None
            } else if trimmed.eq_ignore_ascii_case("auto") {
                Some("Perform ASR.".to_string())
            } else {
                Some(format!("Perform ASR in {trimmed}."))
            }
        })
    });
    let user_prompt = prompt.unwrap_or_default();
    let max_tokens = max_tokens.unwrap_or(256);
    let seed = seed.or(chat.seed);
    let lm = Arc::clone(&loaded.lm);

    let result = tokio::task::spawn_blocking(move || {
        let sys_ref = sys.as_deref().unwrap_or(DEFAULT_ASR_SYSTEM_PROMPT);
        let mut guard = audio.blocking_lock();
        guard.generate(
            &lm,
            Some(&wav_bytes),
            Some(sys_ref),
            &user_prompt,
            Some(GenerationMode::Text),
            false,
            max_tokens,
            temperature,
            seed,
        )
    })
    .await;

    let out = match result {
        Ok(Ok(out)) => out,
        Ok(Err(err)) => {
            return api_error(Status::BadRequest, Some("file"), "invalid_request_error", "audio_error", err.to_string());
        },
        Err(err) => {
            return api_error(
                Status::InternalServerError,
                None,
                "server_error",
                "task_join_error",
                err.to_string(),
            );
        },
    };

    if matches!(response_format.as_deref(), Some("text")) {
        return AudioApiResult::Text(Status::Ok, out.text);
    }

    AudioApiResult::Json(
        Status::Ok,
        serde_json::json!({
            "text": out.text,
            "model": loaded.model_name,
            "usage": {
                "tokens": out.tokens.len(),
                "timing": AudioTiming {
                    prep_ms: out.prep_ms,
                    prefill_ms: out.prefill_ms,
                    depth_ms: out.depth_ms,
                    backbone_ms: out.backbone_ms,
                    codec_ms: out.codec_ms,
                }
            }
        }),
    )
}

#[get("/")]
pub fn handle_web_ui() -> (ContentType, &'static str) {
    (ContentType::HTML, include_str!("ui.html"))
}

#[post("/audio/transcriptions", format = "multipart/form-data", data = "<form>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_audio_transcriptions_multipart<'r>(
    form: Form<AudioTranscriptionForm<'r>>,
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> AudioApiResult {
    let f = form.into_inner();
    let mut stream = match f.file.open().await {
        Ok(s) => s,
        Err(err) => {
            return api_error(
                Status::BadRequest,
                Some("file"),
                "invalid_request_error",
                "file_open_error",
                err.to_string(),
            );
        },
    };
    let mut wav_bytes = Vec::new();
    if let Err(err) = stream.read_to_end(&mut wav_bytes).await {
        return api_error(
            Status::BadRequest,
            Some("file"),
            "invalid_request_error",
            "file_read_error",
            err.to_string(),
        );
    }
    run_transcription(
        chat.inner(),
        f.model,
        wav_bytes,
        f.prompt,
        f.system_prompt,
        f.language,
        f.response_format,
        f.temperature,
        f.max_tokens,
        f.seed,
    )
    .await
}

#[post("/audio/transcriptions", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_audio_transcriptions_json(
    body: Json<AudioTranscriptionJsonRequest>,
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> AudioApiResult {
    let req = body.into_inner();
    let wav_bytes = if let Some(b64) = req.audio_base64 {
        let raw_b64 = b64.split_once(',').map_or(b64.as_str(), |(_, rest)| rest);
        match base64::engine::general_purpose::STANDARD.decode(raw_b64.trim()) {
            Ok(bytes) => bytes,
            Err(err) => {
                return api_error(
                    Status::BadRequest,
                    Some("audio_base64"),
                    "invalid_request_error",
                    "invalid_base64",
                    err.to_string(),
                );
            },
        }
    } else if let Some(path) = req.audio_path {
        match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(err) => {
                return api_error(
                    Status::BadRequest,
                    Some("audio_path"),
                    "invalid_request_error",
                    "file_read_error",
                    format!("failed to read '{path}': {err}"),
                );
            },
        }
    } else {
        return api_error(
            Status::BadRequest,
            Some("audio_base64"),
            "invalid_request_error",
            "missing_audio",
            "provide 'audio_base64' or 'audio_path' (or use multipart/form-data with 'file')",
        );
    };

    run_transcription(
        chat.inner(),
        req.model,
        wav_bytes,
        req.prompt,
        req.system_prompt,
        req.language,
        req.response_format,
        req.temperature,
        req.max_tokens,
        req.seed,
    )
    .await
}

#[post("/audio/speech", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_audio_speech(
    body: Json<AudioSpeechRequest>,
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> AudioApiResult {
    let req = body.into_inner();
    if req.input.trim().is_empty() {
        return api_error(
            Status::BadRequest,
            Some("input"),
            "invalid_request_error",
            "empty_input",
            "'input' must not be empty",
        );
    }
    let (loaded, audio) = match resolve_audio_model(chat.inner(), req.model.as_deref()).await {
        Ok(pair) => pair,
        Err(err) => return err,
    };

    let format = req.response_format.as_deref().unwrap_or("wav");
    if !matches!(format, "wav" | "json" | "pcm") {
        return api_error(
            Status::BadRequest,
            Some("response_format"),
            "invalid_request_error",
            "unsupported_format",
            format!("unsupported response_format '{format}' (supported: wav, json, pcm)"),
        );
    }

    let max_tokens = req.max_tokens.unwrap_or(512);
    let seed = req.seed.or(chat.seed);
    let lm = Arc::clone(&loaded.lm);
    let model_name = loaded.model_name.clone();
    let input = req.input;
    let resolved_sys = req.system_prompt.or_else(|| {
        if req.language.is_some() || req.voice.is_some() {
            let lang = req
                .language
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("American English");
            let voice = req
                .voice
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("female");
            Some(format!("Perform TTS in {lang} using a {voice} voice."))
        } else {
            None
        }
    });
    let temp = req.temperature;
    let options = AudioGenerateOptions {
        audio_temperature: req.audio_temperature,
        audio_top_k: req.audio_top_k,
        audio_repetition_penalty: req.audio_repetition_penalty,
        audio_repetition_window: req.audio_repetition_window,
        assistant_prefill: None,
    };

    if req.stream.unwrap_or(false) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        tokio::task::spawn_blocking(move || {
            let sys_ref = resolved_sys.as_deref().unwrap_or(DEFAULT_TTS_SYSTEM_PROMPT);
            let mut guard = audio.blocking_lock();
            let mut chunk_idx = 0usize;
            let res = guard.generate_with_options(
                &lm,
                None,
                Some(sys_ref),
                &input,
                Some(GenerationMode::Audio),
                true,
                max_tokens,
                temp,
                seed,
                &options,
                |wav_chunk| {
                    chunk_idx += 1;
                    let b64 = base64::engine::general_purpose::STANDARD.encode(wav_chunk);
                    let payload = serde_json::json!({
                        "type": "audio_chunk",
                        "index": chunk_idx,
                        "audio_base64": b64,
                    });
                    let _ = tx.send(Event::data(payload.to_string()));
                },
            );
            match res {
                Ok(out) => {
                    let done_payload = serde_json::json!({
                        "type": "done",
                        "model": model_name,
                        "frames": out.frames.len(),
                        "sample_rate": out.sample_rate,
                        "timing": AudioTiming {
                            prep_ms: out.prep_ms,
                            prefill_ms: out.prefill_ms,
                            depth_ms: out.depth_ms,
                            backbone_ms: out.backbone_ms,
                            codec_ms: out.codec_ms,
                        }
                    });
                    let _ = tx.send(Event::data(done_payload.to_string()));
                },
                Err(err) => {
                    let err_payload = serde_json::json!({
                        "type": "error",
                        "error": err.to_string(),
                    });
                    let _ = tx.send(Event::data(err_payload.to_string()));
                },
            }
        });
        let body: Pin<Box<dyn futures_util::Stream<Item = Event> + Send>> =
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|ev| (ev, rx))
            }));
        return AudioApiResult::Stream(EventStream::from(body));
    }

    let result = tokio::task::spawn_blocking(move || {
        let sys_ref = resolved_sys.as_deref().unwrap_or(DEFAULT_TTS_SYSTEM_PROMPT);
        let mut guard = audio.blocking_lock();
        guard.generate_with_options(
            &lm,
            None,
            Some(sys_ref),
            &input,
            Some(GenerationMode::Audio),
            true,
            max_tokens,
            temp,
            seed,
            &options,
            |_| {},
        )
    })
    .await;

    let out = match result {
        Ok(Ok(out)) => out,
        Ok(Err(err)) => {
            return api_error(
                Status::InternalServerError,
                None,
                "server_error",
                "audio_generation_error",
                err.to_string(),
            );
        },
        Err(err) => {
            return api_error(
                Status::InternalServerError,
                None,
                "server_error",
                "task_join_error",
                err.to_string(),
            );
        },
    };

    let wav = out.wav_bytes.unwrap_or_default();
    if format == "pcm" {
        let pcm = if wav.len() > 44 { wav[44..].to_vec() } else { Vec::new() };
        return AudioApiResult::Pcm(pcm);
    }
    if format == "json" {
        let audio_base64 = base64::engine::general_purpose::STANDARD.encode(&wav);
        return AudioApiResult::Json(
            Status::Ok,
            serde_json::json!({
                "model": loaded.model_name,
                "text": out.text,
                "frames": out.frames.len(),
                "sample_rate": out.sample_rate,
                "audio_base64": audio_base64,
                "timing": AudioTiming {
                    prep_ms: out.prep_ms,
                    prefill_ms: out.prefill_ms,
                    depth_ms: out.depth_ms,
                    backbone_ms: out.backbone_ms,
                    codec_ms: out.codec_ms,
                }
            }),
        );
    }

    AudioApiResult::Wav(wav)
}

#[get("/audio/capabilities")]
pub async fn handle_audio_capabilities(chat: &State<ChatState>) -> AudioApiResult {
    let (loaded, audio) = match resolve_audio_model(chat.inner(), None).await {
        Ok(pair) => pair,
        Err(err) => return err,
    };
    let guard = audio.lock().await;
    let cfg = &guard.pipeline.config;
    let has_vision = loaded.vlm_pipeline.is_some();
    let max_ctx = loaded.lm.recommended_context_length().unwrap_or(32_768);
    let modalities: Vec<&str> = if has_vision {
        vec!["text", "image", "video", "audio_in", "audio_out", "interleaved"]
    } else {
        vec!["text", "audio_in", "audio_out", "interleaved"]
    };
    AudioApiResult::Json(
        Status::Ok,
        serde_json::json!({
            "model": loaded.model_name,
            "omni": has_vision,
            "modalities": modalities,
            "max_context_tokens": max_ctx,
            "live_asr_languages": ["en"],
            "sample_rate_in": cfg.sample_rate,
            "sample_rate_out": guard.codec.as_ref().map(|c| c.config().sampling_rate).unwrap_or(24_000),
            "interleaved_n_text": cfg.interleaved_n_text,
            "interleaved_n_audio": cfg.interleaved_n_audio,
            "audio_sampling_defaults": {
                "audio_temperature": cfg.audio_sampling.temperature,
                "audio_top_k": cfg.audio_sampling.top_k,
                "audio_repetition_penalty": cfg.audio_sampling.repetition_penalty,
                "audio_repetition_window": cfg.audio_sampling.repetition_window,
            },
            "languages": [
                "Arabic", "Danish", "German", "Greek", "British English", "American English",
                "Finnish", "French", "Hebrew", "Italian", "Japanese", "Korean",
                "Dutch", "Polish", "Portuguese", "Spanish", "Swedish", "Turkish", "Chinese"
            ],
            "constraints": {
                "live_asr_english_only": true,
                "voice_chat_english_only": true,
                "multilingual_asr_tts_translate": true,
                "audio_understanding": true,
                "vision_understanding": has_vision,
                "video_understanding": has_vision,
                "spoken_tool_calling": true,
                "reasoning_think_tags": true
            },
            "endpoints": [
                "POST /v1/chat/completions",
                "POST /v1/audio/transcriptions",
                "POST /v1/audio/speech",
                "POST /v1/audio/translate",
                "POST /v1/audio/live-asr/start",
                "POST /v1/audio/live-asr/chunk",
                "POST /v1/audio/live-asr/stop",
                "GET /v1/audio/capabilities",
                "GET /v1/omni/capabilities"
            ]
        }),
    )
}

#[get("/omni/capabilities")]
pub async fn handle_omni_capabilities(chat: &State<ChatState>) -> AudioApiResult {
    handle_audio_capabilities(chat).await
}

#[post("/audio/translate", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_audio_translate(
    body: Json<AudioTranslateRequest>,
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> AudioApiResult {
    let req = body.into_inner();
    let (loaded, audio) = match resolve_audio_model(chat.inner(), req.model.as_deref()).await {
        Ok(pair) => pair,
        Err(err) => return err,
    };

    let wav_bytes = if let Some(b64) = req.audio_base64 {
        let raw_b64 = b64.split_once(',').map_or(b64.as_str(), |(_, rest)| rest);
        match base64::engine::general_purpose::STANDARD.decode(raw_b64.trim()) {
            Ok(bytes) => Some(bytes),
            Err(err) => {
                return api_error(
                    Status::BadRequest,
                    Some("audio_base64"),
                    "invalid_request_error",
                    "invalid_base64",
                    err.to_string(),
                );
            },
        }
    } else if let Some(path) = req.audio_path {
        match tokio::fs::read(&path).await {
            Ok(bytes) => Some(bytes),
            Err(err) => {
                return api_error(
                    Status::BadRequest,
                    Some("audio_path"),
                    "invalid_request_error",
                    "file_read_error",
                    format!("failed to read '{path}': {err}"),
                );
            },
        }
    } else {
        None
    };

    let (sys_prompt, user_text) = if wav_bytes.is_some() {
        (
            format!(
                "Translate the following speech from {} to {} in text form.",
                req.source_language.trim(),
                req.target_language.trim()
            ),
            req.text.unwrap_or_default(),
        )
    } else if let Some(txt) = req.text.filter(|t| !t.trim().is_empty()) {
        (
            format!(
                "Translate the following {} text into {}.",
                req.source_language.trim(),
                req.target_language.trim()
            ),
            txt,
        )
    } else {
        return api_error(
            Status::BadRequest,
            Some("audio_base64"),
            "invalid_request_error",
            "missing_input",
            "provide 'audio_base64', 'audio_path', or 'text' to translate",
        );
    };

    let max_tokens = req.max_tokens.unwrap_or(512);
    let seed = req.seed.or(chat.seed);
    let lm = Arc::clone(&loaded.lm);
    let temp = req.temperature;

    let result = tokio::task::spawn_blocking(move || {
        let mut guard = audio.blocking_lock();
        guard.generate(
            &lm,
            wav_bytes.as_deref(),
            Some(&sys_prompt),
            &user_text,
            Some(GenerationMode::Text),
            false,
            max_tokens,
            temp,
            seed,
        )
    })
    .await;

    let out = match result {
        Ok(Ok(out)) => out,
        Ok(Err(err)) => {
            return api_error(
                Status::BadRequest,
                None,
                "invalid_request_error",
                "translation_error",
                err.to_string(),
            );
        },
        Err(err) => {
            return api_error(
                Status::InternalServerError,
                None,
                "server_error",
                "task_join_error",
                err.to_string(),
            );
        },
    };

    AudioApiResult::Json(
        Status::Ok,
        serde_json::json!({
            "text": out.text,
            "model": loaded.model_name,
            "source_language": req.source_language,
            "target_language": req.target_language,
            "usage": {
                "tokens": out.tokens.len(),
                "timing": AudioTiming {
                    prep_ms: out.prep_ms,
                    prefill_ms: out.prefill_ms,
                    depth_ms: out.depth_ms,
                    backbone_ms: out.backbone_ms,
                    codec_ms: out.codec_ms,
                }
            }
        }),
    )
}

#[post("/audio/live-asr/start", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_live_asr_start(
    body: Json<LiveAsrStartRequest>,
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> AudioApiResult {
    let req = body.into_inner();
    let (loaded, audio) = match resolve_audio_model(chat.inner(), req.model.as_deref()).await {
        Ok(pair) => pair,
        Err(err) => return err,
    };
    let sid = req
        .session_id
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("live_{}", uuid::Uuid::new_v4().simple()));
    let sid_ret = sid.clone();
    let lm = Arc::clone(&loaded.lm);

    let res = tokio::task::spawn_blocking(move || {
        let mut guard = audio.blocking_lock();
        guard.live_asr_start(&lm, sid)
    })
    .await;

    match res {
        Ok(Ok(())) => AudioApiResult::Json(
            Status::Ok,
            serde_json::json!({
                "session_id": sid_ret,
                "model": loaded.model_name,
                "language": "English",
                "note": "LiveASR currently supports English only."
            }),
        ),
        Ok(Err(err)) => api_error(
            Status::InternalServerError,
            None,
            "server_error",
            "live_asr_start_error",
            err.to_string(),
        ),
        Err(err) => api_error(
            Status::InternalServerError,
            None,
            "server_error",
            "task_join_error",
            err.to_string(),
        ),
    }
}

#[post("/audio/live-asr/chunk", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_live_asr_chunk(
    body: Json<LiveAsrChunkRequest>,
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> AudioApiResult {
    let req = body.into_inner();
    let (loaded, audio) = match resolve_audio_model(chat.inner(), req.model.as_deref()).await {
        Ok(pair) => pair,
        Err(err) => return err,
    };
    let audio_bytes = match req.audio_base64 {
        Some(b64) if !b64.trim().is_empty() => {
            let raw_b64 = b64.split_once(',').map_or(b64.as_str(), |(_, rest)| rest);
            match base64::engine::general_purpose::STANDARD.decode(raw_b64.trim()) {
                Ok(bytes) => bytes,
                Err(err) => {
                    return api_error(
                        Status::BadRequest,
                        Some("audio_base64"),
                        "invalid_request_error",
                        "invalid_base64",
                        err.to_string(),
                    );
                },
            }
        },
        _ => Vec::new(),
    };
    let is_final = req.is_final.unwrap_or(false);
    let max_new_tokens = req.max_new_tokens.unwrap_or(if is_final { 96 } else { 48 });
    let sid = req.session_id;
    let sid_ret = sid.clone();
    let format = req.format;
    let sample_rate = req.sample_rate;
    let temp = req.temperature;
    let seed = req.seed.or(chat.seed);
    let lm = Arc::clone(&loaded.lm);

    let res = tokio::task::spawn_blocking(move || {
        let mut guard = audio.blocking_lock();
        guard.live_asr_chunk(
            &lm,
            &sid,
            &audio_bytes,
            format.as_deref(),
            sample_rate,
            is_final,
            max_new_tokens,
            temp,
            seed,
        )
    })
    .await;

    match res {
        Ok(Ok((text, context_tokens))) => AudioApiResult::Json(
            Status::Ok,
            serde_json::json!({
                "session_id": sid_ret,
                "text": text,
                "is_final": is_final,
                "context_tokens": context_tokens
            }),
        ),
        Ok(Err(err)) => api_error(
            Status::BadRequest,
            None,
            "invalid_request_error",
            "live_asr_chunk_error",
            err.to_string(),
        ),
        Err(err) => api_error(
            Status::InternalServerError,
            None,
            "server_error",
            "task_join_error",
            err.to_string(),
        ),
    }
}

#[post("/audio/live-asr/stop", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_live_asr_stop(
    body: Json<LiveAsrStopRequest>,
    chat: &State<ChatState>,
    request_info: &RequestInfo,
) -> AudioApiResult {
    let req = body.into_inner();
    let (_, audio) = match resolve_audio_model(chat.inner(), req.model.as_deref()).await {
        Ok(pair) => pair,
        Err(err) => return err,
    };
    let sid = req.session_id;
    let mut guard = audio.lock().await;
    if let Err(err) = guard.live_asr_stop(&sid) {
        return api_error(
            Status::InternalServerError,
            None,
            "server_error",
            "live_asr_stop_error",
            err.to_string(),
        );
    }
    AudioApiResult::Json(Status::Ok, serde_json::json!({ "stopped": true, "session_id": sid }))
}
