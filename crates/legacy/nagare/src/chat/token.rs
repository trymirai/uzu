use std::{
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use futures::{Stream, StreamExt, stream};
use hanashi::{
    Encoding as EncodingTrait,
    chat::{Encoding, EncodingConfig, TokenizerLocation, hanashi::HanashiEncodingImpl, harmony::HarmonyEncodingImpl},
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use keisoku::KeisokuError;
use shoji::{
    traits::{
        State,
        backend::{
            Error,
            chat_message::{Output, ToolCallState},
            chat_token::{
                Backend, Instance as ChatTokenBackendInstance, StreamInput, StreamOutput, TokenStreamMetrics,
            },
        },
    },
    types::{
        basic::{SamplingParameters, TokenId},
        model::Model,
        session::chat::{
            ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyEnergy, ChatReplyFinishReason,
            ChatReplySpeculatorStats, ChatReplyStats,
        },
    },
};
use tokio_util::sync::CancellationToken;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::util::power::{EnergyRecorder, Error as EnergyError};
use crate::{chat::ChatSessionError, util::helpers::error_stream};

pub struct Session {
    instance: Arc<dyn ChatTokenBackendInstance>,
    state: Box<dyn State>,
    encoding: Encoding,
    input_tokens: Vec<u64>,
    stop_token_ids: Box<[u64]>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    energy_recorder: EnergyRecorder,
}

impl Session {
    pub async fn create_instance(
        backend: &dyn Backend,
        config: ChatConfig,
        reference: String,
    ) -> Result<Arc<dyn ChatTokenBackendInstance>, ChatSessionError> {
        let instance = backend.instance(reference, config).await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        Ok(Arc::from(instance))
    }

    pub async fn with_instance(
        instance: Arc<dyn ChatTokenBackendInstance>,
        reference: String,
        model: &Model,
    ) -> Result<Self, ChatSessionError> {
        let encoding_config: Option<EncodingConfig> = model
            .encoding
            .as_ref()
            .map(|value| serde_json::from_str::<EncodingConfig>(value.json.as_str()))
            .transpose()
            .map_err(|error| ChatSessionError::Loading {
                message: format!("Failed to parse encoding config: {error}"),
            })?;

        let tokenizer_location = TokenizerLocation::Directory {
            path: reference,
            name: None,
        };

        let encoding = match encoding_config {
            Some(EncodingConfig::Hanashi {
                config,
            }) => HanashiEncodingImpl::new(config, instance.tokenizer()).map(Encoding::Hanashi).map_err(|err| {
                ChatSessionError::Loading {
                    message: format!("can not create harmony encoding: {err}"),
                }
            }),
            Some(EncodingConfig::Harmony {
                config,
            }) => HarmonyEncodingImpl::new(config, tokenizer_location).map(Encoding::Harmony).map_err(|err| {
                ChatSessionError::Loading {
                    message: format!("can not create harmony encoding: {err}"),
                }
            }),
            None => Err(ChatSessionError::Loading {
                message: "can not get encoding config".to_string(),
            }),
        }?;
        let state = instance.state().await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        let stop_token_ids = instance.stop_token_ids().ok_or_else(|| ChatSessionError::Loading {
            message: "stop_token_ids is None".to_string(),
        })?;

        Ok(Self {
            instance,
            state,
            encoding,
            input_tokens: Vec::new(),
            stop_token_ids,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            energy_recorder: EnergyRecorder::new(),
        })
    }

    pub async fn reset(&mut self) -> Result<(), ChatSessionError> {
        self.encoding.reset().map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        self.state_reset().await?;
        Ok(())
    }

    pub async fn stream<'a>(
        &'a mut self,
        input: &'a [ChatMessage],
        config: ChatReplyConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
        let time_start = Instant::now();

        let curr_all_tokens = self.encoding.state().tokens.clone();
        let new_all_tokens = match self.build_input(input) {
            Ok(input) => input,
            Err(err) => {
                return error_stream(ChatSessionError::Backend {
                    message: err.to_string(),
                });
            },
        };

        // The engine state can only be kept whole or reset, so reuse it whenever the session's
        // text is a prefix of the newly rendered text — even if tokenizations differ, as sampled
        // replies are not canonically tokenized — and prefill only the raw-tokenized text suffix.
        let curr_text = curr_all_tokens.iter().fold(String::new(), |mut text, token| {
            text.push_str(&token.value);
            text
        });
        let new_text = self.encoding.state().tokens.iter().fold(String::new(), |mut text, token| {
            text.push_str(&token.value);
            text
        });
        let reset = !new_text.starts_with(&curr_text);
        let cached_tokens_input = if reset {
            0
        } else {
            curr_all_tokens.len()
        };
        self.input_tokens = if reset {
            if let Err(err) = self.state_reset().await {
                return error_stream(err);
            }
            new_all_tokens
        } else {
            match self.encoding.tokenize(&new_text[curr_text.len()..]) {
                Ok(suffix_tokens) => suffix_tokens.into_iter().map(u64::from).collect(),
                Err(err) => {
                    return error_stream(ChatSessionError::Backend {
                        message: err.to_string(),
                    });
                },
            }
        };

        let instance = self.instance.as_ref();
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            if let Err(error) = self.energy_recorder.begin() {
                return error_stream(error.into());
            }
        }
        let time_prefill_start = Instant::now();
        let stream = instance.stream(&self.input_tokens, self.state.as_mut(), config.clone(), cancel_token.clone());

        let stream_state = StreamingState {
            config: config.clone(),
            cancel_token,
            encoding: &mut self.encoding,
            max_context_length: self.instance.max_context_length(),
            stop_token_ids: self.stop_token_ids.clone(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            energy_recorder: &mut self.energy_recorder,

            time_start,
            time_last_token: None,
            time_prefill_start,
            time_first_token: None,
            input_energy: None,
            total_tokens_input: self.input_tokens.len(),
            cached_tokens_input,
            total_tokens_output: 0,
            memory_usage: None,
            metrics: None,
        };

        stream::unfold(
            (stream, stream_state, false, false),
            move |(mut inner, mut state, terminated, tail_done)| async move {
                if tail_done {
                    return None;
                }

                match inner.next().await {
                    Some(event) => {
                        state.metrics = inner.metrics();
                        state.memory_usage = instance.peak_memory_usage();
                        let output = Self::build_output(event, &mut state);
                        let terminated = terminated || matches!(&output, Ok(out) if out.finish_reason.is_some());
                        Some((output, (inner, state, terminated, false)))
                    },
                    None => {
                        if !terminated && state.cancel_token.is_cancelled() {
                            let output = Self::render_output(&mut state, Some(ChatReplyFinishReason::Cancelled));
                            Some((output, (inner, state, true, true)))
                        } else {
                            None
                        }
                    },
                }
            },
        )
        .boxed()
    }

    pub fn peak_memory_usage(&self) -> Option<usize> {
        self.instance.peak_memory_usage()
    }

    async fn state_reset(&mut self) -> Result<(), ChatSessionError> {
        self.state = self.instance.state().await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        Ok(())
    }

    fn build_input(
        &mut self,
        all_messages: &[ChatMessage],
    ) -> Result<StreamInput, ChatSessionError> {
        self.encoding.reset().map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;
        self.encoding.encode(all_messages.to_vec()).map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;
        let all_tokens = self.encoding.state().tokens.iter().map(|token| token.id as u64).collect::<Vec<u64>>();
        Ok(all_tokens)
    }

    fn build_output(
        event: Result<StreamOutput, Error>,
        state: &mut StreamingState<'_>,
    ) -> Result<Output, ChatSessionError> {
        let now = Instant::now();
        let result = event.map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;

        match result {
            StreamOutput::LimitReached => Self::render_output(state, Some(ChatReplyFinishReason::Length)),
            StreamOutput::Token(token) => {
                if state.total_tokens_output == 0 {
                    state.time_first_token = Some(now);
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    {
                        match state.energy_recorder.split() {
                            Ok(energy) => state.input_energy = Some(energy),
                            Err(EnergyError::Keisoku(KeisokuError::PowerReadingUnavailable)) => {},
                            Err(error) => return Err(error.into()),
                        }
                    }
                }
                state.total_tokens_output += 1;
                state.time_last_token = Some(now);

                if let Err(err) = state.encoding.decode(vec![token as TokenId]) {
                    return Err(ChatSessionError::Backend {
                        message: err.to_string(),
                    });
                }

                let finish_reason = state.get_finish_reason(token);
                Self::render_output(state, finish_reason)
            },
        }
    }

    fn render_output(
        state: &mut StreamingState<'_>,
        finish_reason: Option<ChatReplyFinishReason>,
    ) -> Result<Output, ChatSessionError> {
        let have_finish_reason = finish_reason.is_some();
        let stats = state.get_stats(have_finish_reason)?;
        let Some(message) = state.encoding.state().messages.last() else {
            return Ok(Output {
                finish_reason,
                stats,
                ..Default::default()
            });
        };

        let tool_calls = message
            .content
            .iter()
            .filter_map(|block| match block {
                ChatContentBlock::ToolCall {
                    value,
                } => Some(ToolCallState::Finished(value.clone())),
                ChatContentBlock::ToolCallCandidate {
                    value,
                } => Some(ToolCallState::Candidate(
                    // the block stores the candidate text as a JSON string document
                    serde_json::from_str::<String>(&value.json).unwrap_or_default(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();

        let finish_reason = if let Some(ChatReplyFinishReason::Stop) = finish_reason
            && !tool_calls.is_empty()
        {
            Some(ChatReplyFinishReason::ToolCalls)
        } else {
            finish_reason
        };

        Ok(Output {
            reasoning: message.reasoning(),
            text: message.text(),
            tool_calls,
            finish_reason,
            stats,
        })
    }

    pub fn supports_tool_calls(&self) -> bool {
        self.encoding.supports_tool_calls()
    }

    pub fn supports_multiple_tool_calls(&self) -> bool {
        self.encoding.supports_multiple_tool_calls()
    }

    pub fn sampling_defaults(&self) -> SamplingParameters {
        self.instance.sampling_defaults()
    }
}

struct StreamingState<'a> {
    config: ChatReplyConfig,
    cancel_token: CancellationToken,
    encoding: &'a mut Encoding,
    max_context_length: Option<usize>,
    stop_token_ids: Box<[u64]>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    energy_recorder: &'a mut EnergyRecorder,

    time_start: Instant,
    time_last_token: Option<Instant>,
    time_prefill_start: Instant,
    time_first_token: Option<Instant>,
    input_energy: Option<ChatReplyEnergy>,
    total_tokens_input: usize,
    cached_tokens_input: usize,
    total_tokens_output: usize,
    memory_usage: Option<usize>,
    metrics: Option<TokenStreamMetrics>,
}

impl StreamingState<'_> {
    fn get_finish_reason(
        &self,
        token: u64,
    ) -> Option<ChatReplyFinishReason> {
        let tokens_count = self.encoding.state().tokens.len();
        if self.cancel_token.is_cancelled() {
            Some(ChatReplyFinishReason::Cancelled)
        } else if self.stop_token_ids.contains(&token) {
            Some(ChatReplyFinishReason::Stop)
        } else if let Some(token_limit) = self.config.token_limit
            && self.total_tokens_output >= token_limit as usize
        {
            Some(ChatReplyFinishReason::Length)
        } else if let Some(length) = self.max_context_length
            && tokens_count >= length
        {
            Some(ChatReplyFinishReason::ContextLimitReached)
        } else {
            None
        }
    }

    fn get_stats(
        &mut self,
        last_stat: bool,
    ) -> Result<ChatReplyStats, ChatSessionError> {
        let speculator_stats = if let Some(metrics) = self.metrics.as_ref() {
            let num_forward_passes = metrics.num_prefill_forward_passes + metrics.num_decode_forward_passes;
            (num_forward_passes > 0).then(|| ChatReplySpeculatorStats {
                tokens_per_forward_pass: metrics.num_tokens_accepted as f64 / num_forward_passes as f64,
                num_decode_forward_passes: num_forward_passes as u32,
            })
        } else {
            None
        };

        let total_duration = self.time_last_token.unwrap_or(Instant::now()).duration_since(self.time_start);
        let ttft_duration =
            self.time_first_token.map(|time_first_token| time_first_token.duration_since(self.time_prefill_start));
        let prefill_tps = ttft_duration.and_then(|duration| {
            (self.total_tokens_input > 0 && !duration.is_zero())
                .then(|| self.total_tokens_input as f64 / duration.as_secs_f64())
        });

        let generate_duration = if let (Some(start), Some(finish)) = (self.time_first_token, self.time_last_token) {
            Some(finish.duration_since(start))
        } else {
            None
        };
        let generate_tps = calculate_rate(self.total_tokens_output, generate_duration);

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let completed_energy = if last_stat {
            match self.energy_recorder.finish() {
                Ok(energy) => Some(energy),
                Err(EnergyError::Keisoku(KeisokuError::PowerReadingUnavailable)) => None,
                Err(error) => return Err(error.into()),
            }
        } else {
            None
        };
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let completed_energy = None;
        let (input_energy, output_energy) = if self.time_first_token.is_some() {
            (self.input_energy.clone(), completed_energy)
        } else {
            (completed_energy, None)
        };
        Ok(ChatReplyStats {
            duration: total_duration.as_secs_f64(),
            time_to_first_token: ttft_duration.map(|time| time.as_secs_f64()),
            prefill_tokens_per_second: prefill_tps,
            generate_tokens_per_second: generate_tps,
            tokens_count_input: Some(self.total_tokens_input as u32),
            tokens_count_input_cached: Some(self.cached_tokens_input as u32),
            tokens_count_output: Some(self.total_tokens_output as u32),
            memory_used_bytes: last_stat.then(|| self.memory_usage.map(|bytes| bytes as i64)).flatten(),
            speculator_stats,
            input_energy,
            output_energy,
        })
    }
}

fn calculate_rate(
    tokens: usize,
    duration: Option<Duration>,
) -> Option<f64> {
    if tokens < 2 {
        return None;
    }

    let duration = duration?;
    if duration.is_zero() {
        return None;
    }

    Some((tokens - 1) as f64 / duration.as_secs_f64())
}
