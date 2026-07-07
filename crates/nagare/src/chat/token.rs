use std::{
    pin::Pin,
    time::{Duration, Instant},
};

use futures::{Stream, StreamExt, stream};
use hanashi::{Encoding as EncodingTrait, chat::Encoding};
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
        basic::TokenId,
        model::Model,
        session::chat::{
            ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyFinishReason,
            ChatReplySpeculatorStats, ChatReplyStats,
        },
    },
};
use tokio_util::sync::CancellationToken;

use crate::{
    chat::ChatSessionError,
    util::{
        helpers::{build_encoding, error_stream},
        power::PowerRecorder,
    },
};

pub struct Session {
    instance: Box<dyn ChatTokenBackendInstance>,
    state: Box<dyn State>,
    encoding: Encoding,
    input_tokens: Vec<u64>,
    stop_token_ids: Box<[u64]>,
    power_recorder: Box<dyn PowerRecorder>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        config: ChatConfig,
        reference: String,
        model: &Model,
    ) -> Result<Self, ChatSessionError> {
        let encoding = build_encoding(reference.clone(), model).map_err(|err| ChatSessionError::Loading {
            message: err.to_string(),
        })?;
        let tokenizer = encoding.tokenizer().ok_or_else(|| ChatSessionError::Loading {
            message: "tokenizer is empty".to_string(),
        })?;

        let instance =
            backend.instance(reference, config, tokenizer).await.map_err(|error| ChatSessionError::Backend {
                message: error.to_string(),
            })?;
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
            power_recorder: <dyn PowerRecorder>::create(),
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
        self.power_recorder.begin();

        let curr_all_tokens = self.encoding.state().tokens.clone();
        let new_all_tokens = match self.build_input(input) {
            Ok(input) => input,
            Err(err) => {
                return error_stream(ChatSessionError::Backend {
                    message: err.to_string(),
                });
            },
        };

        // if new_all_tokens = curr_all_tokens + suffix, then just encode suffix,
        // else reset state and encode all tokens
        let mut reset = new_all_tokens.len() <= curr_all_tokens.len();
        if !reset {
            for i in 0..curr_all_tokens.len() {
                if new_all_tokens[i] != curr_all_tokens[i].id as u64 {
                    reset = true;
                    break;
                }
            }
        }
        self.input_tokens = if reset {
            if let Err(err) = self.state_reset().await {
                return error_stream(err);
            }
            new_all_tokens
        } else {
            new_all_tokens[curr_all_tokens.len()..].to_vec()
        };

        let instance = self.instance.as_ref();
        let stream = instance.stream(&self.input_tokens, self.state.as_mut(), config.clone(), cancel_token.clone());

        let stream_state = StreamingState {
            config: config.clone(),
            cancel_token,
            encoding: &mut self.encoding,
            max_context_length: self.instance.max_context_length(),
            stop_token_ids: self.stop_token_ids.clone(),
            power_recorder: &mut *self.power_recorder,

            time_start,
            time_last_token: None,
            time_prefill_start: Instant::now(),
            time_first_token: None,
            total_tokens_input: self.input_tokens.len(),
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
                            let output = Ok(Self::render_output(&state, Some(ChatReplyFinishReason::Cancelled)));
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
        state: &mut StreamingState,
    ) -> Result<Output, ChatSessionError> {
        let now = Instant::now();
        let result = event.map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;

        match result {
            StreamOutput::LimitReached => Ok(Self::render_output(state, Some(ChatReplyFinishReason::Length))),
            StreamOutput::Token(token) => {
                if state.total_tokens_output == 0 {
                    state.time_first_token = Some(now)
                }
                state.total_tokens_output += 1;
                state.time_last_token = Some(now);

                if let Err(err) = state.encoding.decode(vec![token as TokenId]) {
                    return Err(ChatSessionError::Backend {
                        message: err.to_string(),
                    });
                }

                let finish_reason = state.get_finish_reason(token);
                Ok(Self::render_output(state, finish_reason))
            },
        }
    }

    fn render_output(
        state: &StreamingState,
        finish_reason: Option<ChatReplyFinishReason>,
    ) -> Output {
        let have_finish_reason = finish_reason.is_some();
        let Some(message) = state.encoding.state().messages.last() else {
            return Output {
                finish_reason,
                stats: state.get_stats(have_finish_reason),
                ..Default::default()
            };
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
                } => Some(ToolCallState::Candidate(value.json.clone())),
                _ => None,
            })
            .collect();

        Output {
            reasoning: message.reasoning(),
            text: message.text(),
            tool_calls,
            finish_reason,
            stats: state.get_stats(have_finish_reason),
        }
    }
}

struct StreamingState<'a> {
    config: ChatReplyConfig,
    cancel_token: CancellationToken,
    encoding: &'a mut Encoding,
    max_context_length: Option<usize>,
    stop_token_ids: Box<[u64]>,
    power_recorder: &'a mut dyn PowerRecorder,

    time_start: Instant,
    time_last_token: Option<Instant>,
    time_prefill_start: Instant,
    time_first_token: Option<Instant>,
    total_tokens_input: usize,
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
        &self,
        last_stat: bool,
    ) -> ChatReplyStats {
        let speculator_stats = if let Some(metrics) = self.metrics.as_ref()
            && metrics.num_forward_passes > 0
        {
            Some(ChatReplySpeculatorStats {
                tokens_per_forward_pass: metrics.num_tokens_accepted as f64 / metrics.num_forward_passes as f64,
            })
        } else {
            None
        };

        let power_stats = last_stat.then(|| self.power_recorder.finish()).flatten();

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

        ChatReplyStats {
            duration: total_duration.as_secs_f64(),
            time_to_first_token: ttft_duration.map(|time| time.as_secs_f64()),
            prefill_tokens_per_second: prefill_tps,
            generate_tokens_per_second: generate_tps,
            tokens_count_input: Some(self.total_tokens_input as u32),
            tokens_count_output: Some(self.total_tokens_output as u32),
            memory_used_bytes: last_stat.then(|| self.memory_usage.map(|bytes| bytes as i64)).flatten(),
            speculator_stats,
            power_stats,
        }
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
