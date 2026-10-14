use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use js_sys::{Function, Reflect, Uint8Array};
use serde::Serialize;
use serde_json::{Value, json};
use uzu_engine::{
    backends::webgpu::WebGpu,
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            state::LanguageModelState,
            stream::{LanguageModelStreamOptions, SamplingMethod},
        },
    },
};
use wasm_bindgen::{JsCast, prelude::*};

use crate::chat::{CodecConfig, ModelConfig, RunParams, sampling_method};

#[wasm_bindgen(inline_js = r#"
const pending = [];
const channel = new MessageChannel();
channel.port1.onmessage = () => pending.shift()?.();
export function yieldToBrowser() {
    // A normal task boundary lets the worker receive cancellation messages.
    // Nested timers add a delay per token, while scheduler.yield's boosted
    // continuations can starve pending cancellation tasks in a busy worker.
    return new Promise(resolve => {
        pending.push(resolve);
        channel.port2.postMessage(null);
    });
}
export function monotonicNow() {
    return performance.now();
}
"#)]
extern "C" {
    #[wasm_bindgen(js_name = yieldToBrowser)]
    pub(crate) async fn yield_to_browser();
    #[wasm_bindgen(js_name = monotonicNow)]
    pub(crate) fn monotonic_now() -> f64;
}

pub(crate) struct LoadedModel {
    pub(crate) encoded: Option<crate::encoding::EncodedState>,
    pub(crate) model: LanguageModel<WebGpu>,
    pub(crate) state: Option<LanguageModelState<WebGpu>>,
}

pub(crate) struct Inner {
    pub(crate) encoding_support: Option<crate::encoding::EncodingSupport>,
    loaded: RefCell<Option<LoadedModel>>,
    pub(crate) codec: CodecConfig,
    output_regex: Option<fancy_regex::Regex>,
    pub(crate) context_length: Option<u32>,
    seed: Option<u64>,
    running: Cell<bool>,
    pub(crate) cancelled: Cell<bool>,
    pub(crate) tool_abort: RefCell<Option<Function>>,
    disposed: Cell<bool>,
    #[cfg(feature = "kernel-tests")]
    pub(crate) last_tokens: RefCell<Vec<u32>>,
    #[cfg(feature = "kernel-tests")]
    pub(crate) reused_tokens: Cell<u32>,
}

impl Inner {
    pub(crate) fn new_state_seed(&self) -> u64 {
        self.seed.unwrap_or_else(|| js_sys::Date::now() as u64)
    }
}

/// Owns a loaded model. Only one run can use its KV cache at a time.
#[wasm_bindgen]
pub struct Session {
    inner: Rc<Inner>,
}

pub(crate) fn error(message: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}

fn bytes(
    options: &JsValue,
    key: &str,
    maximum: u32,
) -> Result<Vec<u8>, JsValue> {
    let value = Reflect::get(options, &JsValue::from_str(key))?;
    let array = value.dyn_into::<Uint8Array>().map_err(|_| error(format!("{key} must be a Uint8Array")))?;
    if array.length() == 0 || array.length() > maximum {
        return Err(error(format!("{key} is empty or exceeds the browser size limit")));
    }
    Ok(array.to_vec())
}

/// `options` contains config/tokenizer Uint8Arrays and safetensors as a Blob or
/// Uint8Array. Weights are read in bounded ranges outside WASM. Call this
/// inside a dedicated worker, on a secure cross-origin-isolated page.
#[wasm_bindgen(js_name = createSession)]
pub async fn create_session(options: JsValue) -> Result<Session, JsValue> {
    console_error_panic_hook::set_once();
    // Tokenizers' batch thread pool is unnecessary for one browser session.
    // It also cannot spawn OS threads on wasm32-unknown-unknown.
    tokenizers::utils::parallelism::set_parallelism(false);
    let config = bytes(&options, "config", 16 * 1024 * 1024)?;
    uzu_engine::backends::webgpu::validate_model_config(&config).map_err(error)?;
    let config_view: ModelConfig = serde_json::from_slice(&config).map_err(error)?;
    let output_regex =
        config_view.token_codec_config.output_regex().map(fancy_regex::Regex::new).transpose().map_err(error)?;
    let encoding = Reflect::get(&options, &JsValue::from_str("encoding"))?;
    let encoding: Option<hanashi::chat::EncodingConfig> = if encoding.is_undefined() || encoding.is_null() {
        None
    } else {
        Some(serde_wasm_bindgen::from_value(encoding).map_err(error)?)
    };
    let encoding_support = encoding.as_ref().map(crate::encoding::inspect).transpose().map_err(error)?;
    let tokenizer = bytes(&options, "tokenizer", 128 * 1024 * 1024)?;
    let weights =
        uzu_engine::backends::webgpu::WebGpuWeightSource::new(Reflect::get(&options, &JsValue::from_str("weights"))?)
            .and_then(|source| {
                source.with_cancellation(Reflect::get(&options, &JsValue::from_str("loadCancellation"))?)
            })
            .map_err(error)?;
    let requested = Reflect::get(&options, &JsValue::from_str("contextLength"))?;
    let requested = if requested.is_undefined() {
        None
    } else {
        let value = requested.as_f64().ok_or_else(|| error("contextLength must be a number"))?;
        if !value.is_finite() || value.fract() != 0.0 || !(1.0..=65536.0).contains(&value) {
            return Err(error("contextLength must be an integer between 1 and 65536"));
        }
        Some(value as u32)
    };
    let seed = Reflect::get(&options, &JsValue::from_str("seed"))?;
    let seed = if seed.is_undefined() || seed.is_null() {
        None
    } else {
        let value = seed.as_f64().ok_or_else(|| error("seed must be a nonnegative safe integer"))?;
        if !value.is_finite() || value.fract() != 0.0 || !(0.0..=9_007_199_254_740_991.0).contains(&value) {
            return Err(error("seed must be a nonnegative safe integer"));
        }
        Some(value as u64)
    };
    uzu_engine::backends::webgpu::prepare().await.map_err(error)?;
    let engine = Engine::<WebGpu>::new().map_err(error)?;
    let initialize = |context: &_, offset, size| {
        weights.initialize_buffer(context, offset, size).map_err(uzu_engine::ParameterLoaderError::BackendError)
    };
    let model = engine
        .load_language_model_from_source(&config, &tokenizer, &weights, Some(&initialize))
        .map_err(|cause| weights.check_cancellation().err().map(error).unwrap_or_else(|| error(cause)))?;
    weights.check_cancellation().map_err(error)?;
    let context_length = requested
        .map(|length| length.min(model.max_context_length().unwrap_or(length)))
        .or_else(|| model.recommended_context_length());
    let encoded = encoding
        .map(|config| {
            let hanashi::chat::EncodingConfig::Hanashi {
                config,
            } = config;
            hanashi::chat::hanashi::HanashiEncodingImpl::new(config, model.tokenizer().clone())
                .map(|encoder| crate::encoding::EncodedState {
                    encoder,
                    visible: vec![],
                    effort: None,
                    tools: vec![],
                })
                .map_err(error)
        })
        .transpose()?;
    Ok(Session {
        inner: Rc::new(Inner {
            encoding_support,
            loaded: RefCell::new(Some(LoadedModel {
                model,
                state: None,
                encoded,
            })),
            codec: config_view.token_codec_config,
            output_regex,
            context_length,
            seed,
            running: Cell::new(false),
            cancelled: Cell::new(false),
            tool_abort: RefCell::new(None),
            disposed: Cell::new(false),
            #[cfg(feature = "kernel-tests")]
            last_tokens: RefCell::new(Vec::new()),
            #[cfg(feature = "kernel-tests")]
            reused_tokens: Cell::new(0),
        }),
    })
}

// Own the model while yielding to JavaScript. Reentrant cancel/dispose calls
// only touch flags, and a second run is rejected before accessing the model.
struct RunGuard {
    inner: Rc<Inner>,
    loaded: Option<LoadedModel>,
}
impl Drop for RunGuard {
    fn drop(&mut self) {
        if !self.inner.disposed.get() {
            *self.inner.loaded.borrow_mut() = self.loaded.take();
        }
        self.inner.running.set(false);
    }
}

#[wasm_bindgen]
impl Session {
    /// Streams the same RunEvent objects consumed by the desktop chat UI.
    pub async fn run(
        &self,
        params: JsValue,
        on_event: Function,
        on_tool: Option<Function>,
    ) -> Result<(), JsValue> {
        let inner = self.inner.clone();
        if inner.disposed.get() {
            return Err(error("Session has been disposed"));
        }
        if inner.running.replace(true) {
            return Err(error("A generation is already running"));
        }
        let mut guard = RunGuard {
            inner: inner.clone(),
            loaded: inner.loaded.borrow_mut().take(),
        };
        inner.cancelled.set(false);
        #[cfg(feature = "kernel-tests")]
        inner.last_tokens.borrow_mut().clear();
        let params: RunParams = serde_wasm_bindgen::from_value(params).map_err(error)?;
        if inner.encoding_support.is_some() {
            return crate::encoded_run::run(
                &inner,
                guard.loaded.as_mut().ok_or_else(|| error("Session has been disposed"))?,
                params,
                &on_event,
                on_tool.as_ref(),
            )
            .await;
        }
        let now =
            chrono::DateTime::from_timestamp_millis(js_sys::Date::now() as i64).unwrap_or(chrono::DateTime::UNIX_EPOCH);
        let prompt = inner.codec.render(&params, now).map_err(error)?;
        let start = monotonic_now();
        let LoadedModel {
            model,
            state,
            ..
        } = guard.loaded.as_mut().ok_or_else(|| error("Session has been disposed"))?;
        let input: Vec<u64> = model
            .tokenizer()
            .encode(prompt.as_str(), false)
            .map_err(error)?
            .get_ids()
            .iter()
            .map(|&token| u64::from(token))
            .collect();
        if input.is_empty() {
            return Err(error("The model's chat template produced an empty prompt"));
        }
        if let Some(capacity) = inner.context_length
            && input.len() >= capacity as usize
        {
            return Err(error(format!(
                "The conversation uses {} tokens; this browser session has room for {}. Start a new chat or load a larger context.",
                input.len(),
                capacity
            )));
        }
        if state.as_ref().is_none_or(|state| !input.starts_with(state.tokens())) {
            *state = Some(model.create_empty_state(inner.context_length, inner.new_state_seed()).map_err(error)?);
        }
        let state = state.as_mut().unwrap();
        let sampling =
            sampling_method(params.sampling_policy.as_ref(), model.default_sampling_method()).map_err(error)?;
        let thinking_tag = inner.codec.thinking_tag();
        let opening_tag = inner.codec.thinking_opening_tag();
        let starts_thinking = opening_tag.as_ref().is_some_and(|tag| prompt.trim_end().ends_with(tag));
        let mut output = Output::new(starts_thinking, thinking_tag, opening_tag);
        output.publish(&on_event)?;
        let mut offset = state.tokens().len();
        #[cfg(feature = "kernel-tests")]
        inner.reused_tokens.set(offset as u32);
        // Yield between prefill batches as well as decoded tokens. Dropping a
        // prefill stream waits for GPU work, preserving the engine's state API.
        while input.len() - offset > 256 && !inner.cancelled.get() {
            let end = offset + 256;
            drop(
                model
                    .stream(
                        &input[offset..end],
                        state,
                        LanguageModelStreamOptions {
                            sampling_method: SamplingMethod::Greedy,
                            grammar: None,
                        },
                    )
                    .map_err(error)?,
            );
            offset = end;
            yield_to_browser().await;
        }
        let grammar = params
            .grammar
            .as_ref()
            .map(|request| {
                let trigger = if starts_thinking {
                    thinking_tag
                        .map(|tag| {
                            model
                                .tokenizer()
                                .encode(tag, false)
                                .map(|encoding| encoding.get_ids().iter().map(|&id| u64::from(id)).collect())
                                .map_err(error)
                        })
                        .transpose()?
                } else {
                    None
                };
                let stops: Vec<i32> = model.generation_config().stop_token_ids.iter().map(|&id| id as i32).collect();
                uzu_engine::engine::language_model::grammar::Grammar::new(
                    &request.config(),
                    model.tokenizer(),
                    trigger,
                    Some(&stops),
                )
                .map_err(error)
            })
            .transpose()?;
        let mut first_token = None;
        let mut count = 0u32;
        let mut finish_reason = "Cancelled";
        if !inner.cancelled.get() {
            let mut stream = model
                .stream(
                    &input[offset..],
                    state,
                    LanguageModelStreamOptions {
                        sampling_method: sampling,
                        grammar,
                    },
                )
                .map_err(error)?;
            let tokenizer = model.tokenizer();
            let mut decoder = tokenizer.decode_stream(false);
            loop {
                if inner.cancelled.get() {
                    finish_reason = "Cancelled";
                    break;
                }
                if params.max_output_tokens.is_some_and(|limit| count >= limit) {
                    finish_reason = "Length";
                    break;
                }
                let Some(token) = stream.next() else {
                    finish_reason = "ContextLimitReached";
                    break;
                };
                let token = token.map_err(error)?;
                first_token.get_or_insert_with(monotonic_now);
                if model.generation_config().stop_token_ids.contains(&token) {
                    finish_reason = "Stop";
                    break;
                }
                count += 1;
                #[cfg(feature = "kernel-tests")]
                inner.last_tokens.borrow_mut().push(token as u32);
                if let Some(delta) = decoder.step(token as u32).map_err(error)? {
                    output.push(&delta, &on_event)?;
                }
                yield_to_browser().await;
            }
        }
        output.finish(inner.output_regex.as_ref(), &on_event)?;
        let end = monotonic_now();
        let prefill = (first_token.unwrap_or(end) - start).max(0.0) / 1000.0;
        let generation = (end - first_token.unwrap_or(end)).max(0.0) / 1000.0;
        emit(
            &on_event,
            json!({
                "type":"done", "text":output.response, "finishReason":finish_reason,
                "parsed":{"response":output.response,"chainOfThought":output.thinking},
                "transcript":output.transcript(),
                "stats":{
                    "prefillStats":{"duration":prefill,"tokensCount":input.len(),"tokensPerSecond":rate(input.len() as u32,prefill)},
                    "generateStats":{"duration":generation,"tokensCount":count,"tokensPerSecond":rate(count,generation)},
                    "totalStats":{"duration":(end-start).max(0.0)/1000.0,"tokensCountInput":input.len(),"tokensCountOutput":count}
                }
            }),
        )
    }

    pub fn cancel(&self) {
        self.inner.cancelled.set(true);
        if let Some(resolve) = self.inner.tool_abort.borrow_mut().take() {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        }
    }

    /// Exposed only in numerical-test builds for CPU/GPU token parity checks.
    #[cfg(feature = "kernel-tests")]
    #[wasm_bindgen(js_name = debugTokenIds)]
    pub fn debug_token_ids(&self) -> Vec<u32> {
        self.inner.last_tokens.borrow().clone()
    }

    #[cfg(feature = "kernel-tests")]
    #[wasm_bindgen(js_name = debugReusedTokens)]
    pub fn debug_reused_tokens(&self) -> u32 {
        self.inner.reused_tokens.get()
    }

    #[cfg(feature = "kernel-tests")]
    #[wasm_bindgen(js_name = debugContextLength)]
    pub fn debug_context_length(&self) -> Option<u32> {
        self.inner.context_length
    }

    /// Disposal during generation requests cancellation and frees GPU buffers
    /// when the in-flight engine stream has completed.
    pub fn dispose(&self) {
        self.inner.disposed.set(true);
        self.cancel();
        if !self.inner.running.get() {
            self.inner.loaded.borrow_mut().take();
        }
    }
}

pub(crate) fn rate(
    count: u32,
    seconds: f64,
) -> f64 {
    if seconds > 0.0 {
        f64::from(count) / seconds
    } else {
        0.0
    }
}

pub(crate) fn emit(
    callback: &Function,
    event: Value,
) -> Result<(), JsValue> {
    let value = event.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(error)?;
    callback.call1(&JsValue::UNDEFINED, &value).map(|_| ())
}

struct Output<'a> {
    raw: String,
    response: String,
    thinking: Option<String>,
    reading_thinking: bool,
    pending: String,
    tag: Option<&'a str>,
    opening: Option<String>,
    initial: String,
    detect_opening: bool,
}

impl<'a> Output<'a> {
    fn new(
        thinking: bool,
        tag: Option<&'a str>,
        opening: Option<String>,
    ) -> Self {
        let detect_opening = !thinking && opening.is_some();
        Self {
            raw: String::new(),
            response: String::new(),
            thinking: thinking.then(String::new),
            reading_thinking: thinking,
            pending: String::new(),
            tag,
            opening,
            initial: String::new(),
            detect_opening,
        }
    }

    fn transcript(&self) -> Value {
        let mut items = Vec::new();
        if let Some(thinking) = &self.thinking {
            items.push(json!({"type":"thinking","text":thinking,"completed":!self.reading_thinking}));
        }
        if !self.reading_thinking {
            items.push(json!({"type":"text","text":self.response}));
        }
        json!(items)
    }

    fn publish(
        &self,
        callback: &Function,
    ) -> Result<(), JsValue> {
        emit(callback, json!({"type":"transcript","items":self.transcript()}))
    }

    fn append(
        &mut self,
        text: &str,
        callback: &Function,
    ) -> Result<(), JsValue> {
        if text.is_empty() {
            return Ok(());
        }
        let index = if self.reading_thinking {
            self.thinking.as_mut().unwrap().push_str(text);
            0
        } else {
            self.response.push_str(text);
            usize::from(self.thinking.is_some())
        };
        emit(callback, json!({"type":"transcriptDelta","index":index,"delta":text}))?;
        if !self.reading_thinking {
            emit(callback, json!({"type":"chunk","delta":text}))?;
        }
        Ok(())
    }

    fn push(
        &mut self,
        delta: &str,
        callback: &Function,
    ) -> Result<(), JsValue> {
        self.raw.push_str(delta);
        if self.detect_opening {
            self.initial.push_str(delta);
            let opening = self.opening.as_ref().unwrap();
            let initial = self.initial.trim_start();
            if opening.starts_with(initial) && initial.len() < opening.len() {
                return Ok(());
            }
            self.detect_opening = false;
            if let Some(tail) = initial.strip_prefix(opening) {
                let tail = tail.to_owned();
                self.initial.clear();
                self.thinking = Some(String::new());
                self.reading_thinking = true;
                self.publish(callback)?;
                return self.push_content(&tail, callback);
            }
            let initial = std::mem::take(&mut self.initial);
            return self.append(&initial, callback);
        }
        self.push_content(delta, callback)
    }

    fn push_content(
        &mut self,
        delta: &str,
        callback: &Function,
    ) -> Result<(), JsValue> {
        if !self.reading_thinking {
            return self.append(delta, callback);
        }
        self.pending.push_str(delta);
        let tag = self.tag.unwrap();
        if let Some(at) = self.pending.find(tag) {
            let pending = std::mem::take(&mut self.pending);
            self.append(&pending[..at], callback)?;
            self.reading_thinking = false;
            self.publish(callback)?;
            return self.append(&pending[at + tag.len()..], callback);
        }
        let keep = (1..=tag.len().min(self.pending.len()))
            .filter(|&n| tag.is_char_boundary(n) && self.pending.is_char_boundary(self.pending.len() - n))
            .rfind(|&n| self.pending.ends_with(&tag[..n]))
            .unwrap_or(0);
        let ready = self.pending.len() - keep;
        let tail = self.pending.split_off(ready);
        let text = std::mem::replace(&mut self.pending, tail);
        self.append(&text, callback)
    }

    fn finish(
        &mut self,
        regex: Option<&fancy_regex::Regex>,
        callback: &Function,
    ) -> Result<(), JsValue> {
        let initial = std::mem::take(&mut self.initial);
        self.append(&initial, callback)?;
        let pending = std::mem::take(&mut self.pending);
        self.append(&pending, callback)?;
        if let Some(regex) = regex
            && let Some(captures) = regex.captures(&self.raw).map_err(error)?
        {
            if let Some(response) = captures.name("response") {
                self.response = response.as_str().to_string();
            }
            if let Some(thinking) = captures.name("chain_of_thought").or_else(|| captures.name("thinking")) {
                self.thinking = Some(thinking.as_str().to_string());
            }
        }
        self.reading_thinking = false;
        self.publish(callback)
    }
}

/// Allows clients to detect stale engine builds before supplying a Blob.
#[wasm_bindgen(js_name = supportsBlobWeights)]
pub fn supports_blob_weights() -> bool {
    true
}

/// Read registry encoding capabilities without loading model weights or a GPU.
#[wasm_bindgen(js_name = inspectEncoding)]
pub fn inspect_encoding(value: JsValue) -> Result<JsValue, JsValue> {
    let config = serde_wasm_bindgen::from_value(value).map_err(error)?;
    let support = crate::encoding::inspect(&config).map_err(error)?;
    support.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(error)
}

/// Checks engine format and supported architecture without allocating a GPU.
#[wasm_bindgen(js_name = inspectModelConfig)]
pub fn inspect_model_config(config: Uint8Array) -> Result<(), JsValue> {
    if config.length() > 16 * 1024 * 1024 {
        return Err(error("Model config exceeds the 16 MB browser size limit"));
    }
    uzu_engine::backends::webgpu::validate_model_config(&config.to_vec()).map_err(error)
}
