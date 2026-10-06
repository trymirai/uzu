use tauri::{AppHandle, ipc::Channel};
use uzu::{
    session::chat::{ChatSessionError, ChatSessionStreamChunk},
    types::{
        basic::ReasoningEffort,
        session::chat::{ChatReply, ChatReplyConfig, ChatReplyFinishReason},
    },
};

use super::{
    ChatState, RunPlan,
    messages::{assistant_history_message, build_messages, build_tail_messages, parsed_from, sanitize},
    payloads::{Parsed, RunEvent, RunStreamPayload, map_reply_stats, sampling_policy},
    session::ensure_session,
};
use crate::error::{AppError, AppResult};

const DEFAULT_CHAT_TOKENS_LIMIT: u32 = 8_192;

// uzu turns LanguageModelStreamError::ContextOverflow into Backend { message },
// so the variant cannot tell an overflow from any other backend failure.
const ENGINE_CONTEXT_OVERFLOW: &str = "Context overflow";

fn stream_error(error: ChatSessionError) -> AppError {
    match &error {
        ChatSessionError::Backend {
            message,
        } if message == ENGINE_CONTEXT_OVERFLOW => {
            AppError::msg("Conversation is too long for this model's context window.")
        },
        _ => error.into(),
    }
}

pub(super) async fn run_stream_inner(
    app: &AppHandle,
    state: &ChatState,
    payload: &RunStreamPayload,
    on_event: &Channel<RunEvent>,
) -> AppResult<()> {
    let _run_guard = state.run_lock.lock().await;
    let (session, support) = ensure_session(app, state, &payload.repo_id).await?;
    if state.cancel_requested(&payload.run_id) {
        crate::logger::info("chat:run:cancelled-before-start", Some(serde_json::json!({ "runId": payload.run_id })));
        let _ = on_event.send(RunEvent::Done {
            text: String::new(),
            stats: map_reply_stats(None),
            finish_reason: Some(ChatReplyFinishReason::Cancelled),
            parsed: None,
        });
        return Ok(());
    }
    let effort = support.effective(payload.reasoning_effort);

    let reply_config = ChatReplyConfig::create()
        .with_token_limit(Some(DEFAULT_CHAT_TOKENS_LIMIT))
        .with_sampling_policy(sampling_policy(&payload.sampling_policy));

    let plan = state.plan_run(&payload.repo_id, &payload.messages, effort);
    crate::logger::info(
        "chat:run:start",
        Some(serde_json::json!({
            "runId": payload.run_id,
            "repoId": payload.repo_id,
            "requestedEffort": payload.reasoning_effort,
            "support": support,
            "effort": effort,
            "plan": match &plan { RunPlan::Continue { tail } => format!("continue:{}", tail.len()), RunPlan::Replay => "replay".to_string() },
            "messages": payload.messages.len(),
        })),
    );
    let messages = match plan {
        RunPlan::Continue {
            tail,
        } => build_tail_messages(&tail),
        RunPlan::Replay => {
            // Full replay requires resetting uzu's accumulated role validator.
            if let Err(error) = session.reset().await {
                state.evict(app, &payload.repo_id).await;
                return Err(error.into());
            }
            build_messages(&payload.messages, effort)
        },
    };
    state.forget_history();

    let stream = session.reply_with_stream(messages, reply_config).await;
    state.attach_cancel_token(&payload.run_id, stream.cancel_token());

    let should_promote_reasoning = effort == Some(ReasoningEffort::Disabled);
    let effective_channels = |reply: Option<&ChatReply>| -> (String, String) {
        let text = reply.and_then(|r| r.message.text()).unwrap_or_default();
        let reasoning = reply.and_then(|r| r.message.reasoning()).unwrap_or_default();
        if should_promote_reasoning && text.is_empty() && !reasoning.is_empty() {
            (reasoning, String::new())
        } else {
            (text, reasoning)
        }
    };

    let mut output: Option<ChatReply> = None;
    let mut last_parsed: Option<Parsed> = None;
    let mut prev_text_len = 0usize;
    let mut prev_reasoning_len = 0usize;

    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::Error {
                error,
            } => {
                return Err(stream_error(error));
            },
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                let reply = replies.into_iter().next();
                if reply.is_some() {
                    output = reply;
                }
                let (text, reasoning) = effective_channels(output.as_ref());
                last_parsed = parsed_from(&text, &reasoning);
                let text_grew = text.len() > prev_text_len;
                let reasoning_grew = reasoning.len() > prev_reasoning_len;
                // A previous length off a char boundary means a re-sent snapshot; skip
                // the delta, Done carries the full text.
                let delta = if text_grew {
                    text.get(prev_text_len..).unwrap_or("")
                } else {
                    ""
                };
                if text_grew {
                    prev_text_len = text.len();
                }
                if reasoning_grew {
                    prev_reasoning_len = reasoning.len();
                }
                if text_grew || reasoning_grew {
                    let parsed_patch = reasoning_grew.then(|| parsed_from("", &reasoning)).flatten();
                    let _ = on_event.send(RunEvent::Chunk {
                        delta: sanitize(delta),
                        parsed: parsed_patch,
                    });
                }
            },
        }
    }

    let raw_text = output.as_ref().and_then(|r| r.message.text()).unwrap_or_default();
    let raw_reasoning = output.as_ref().and_then(|r| r.message.reasoning()).unwrap_or_default();
    let promoted_reasoning = should_promote_reasoning && raw_text.is_empty() && !raw_reasoning.is_empty();
    let (text, reasoning) = effective_channels(output.as_ref());
    let finish_reason = output.as_ref().and_then(|r| r.finish_reason.as_ref());

    // Cancelled or truncated replies may diverge from what the client persists.
    if let (Some(ChatReplyFinishReason::Stop), Some(assistant)) =
        (finish_reason, assistant_history_message(&raw_text, &raw_reasoning, promoted_reasoning))
    {
        let mut history = payload.messages.clone();
        history.push(assistant);
        state.remember_history(&payload.repo_id, history, effort);
    }
    let finish_reason = finish_reason.cloned();
    crate::logger::info(
        "chat:run:done",
        Some(serde_json::json!({
            "runId": payload.run_id,
            "finishReason": finish_reason,
            "textLen": raw_text.len(),
            "reasoningLen": raw_reasoning.len(),
            "promoted": promoted_reasoning,
        })),
    );

    let _ = on_event.send(RunEvent::Done {
        text: sanitize(&text),
        stats: map_reply_stats(output.as_ref()),
        finish_reason,
        parsed: last_parsed.or_else(|| parsed_from(&text, &reasoning)),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(message: &str) -> ChatSessionError {
        ChatSessionError::Backend {
            message: message.to_string(),
        }
    }

    #[test]
    fn engine_context_overflow_gets_a_readable_message() {
        let error = stream_error(backend(ENGINE_CONTEXT_OVERFLOW));
        assert_eq!(error.to_string(), "Conversation is too long for this model's context window.");
    }

    #[test]
    fn other_backend_errors_pass_through() {
        let error = stream_error(backend("No seed token"));
        assert_eq!(error.to_string(), "Backend error: No seed token");
    }
}
