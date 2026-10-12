use tauri::{AppHandle, Manager, ipc::Channel};
use uzu::{
    session::chat::{ChatSessionError, ChatSessionStreamChunk},
    types::{
        basic::ReasoningEffort,
        session::chat::{ChatContentBlock, ChatReply, ChatReplyConfig, ChatReplyFinishReason},
    },
};

use super::{
    ChatState, RunPlan,
    messages::{assistant_history_message, build_messages, build_tail_messages, parsed_from, sanitize},
    naming::add_naming_instruction,
    payloads::{Parsed, RunEvent, RunStreamPayload, map_reply_stats, sampling_policy},
    session::ensure_session,
    transcript::Transcript,
};
use crate::{
    analytics::{AnalyticsState, Event},
    error::{AppError, AppResult},
};

// uzu turns LanguageModelStreamError::ContextOverflow into Backend { message },
// so the variant cannot tell an overflow from any other backend failure.
const ENGINE_CONTEXT_OVERFLOW: &str = "Context overflow";

#[derive(Default)]
struct ReplyText {
    completed_text: String,
    completed_reasoning: String,
    text: String,
    reasoning: String,
    tool_turn_finished: bool,
}

fn join_turns(
    first: &str,
    second: &str,
) -> String {
    match (first.is_empty(), second.is_empty()) {
        (true, _) => second.to_string(),
        (_, true) => first.to_string(),
        _ => format!("{first}\n\n{second}"),
    }
}

impl ReplyText {
    fn update(
        &mut self,
        reply: &ChatReply,
    ) {
        // Each automatic tool continuation starts a new snapshot at zero.
        // Preserve visible content from the completed assistant turn first.
        if self.tool_turn_finished {
            self.completed_text = join_turns(&self.completed_text, &self.text);
            self.completed_reasoning = join_turns(&self.completed_reasoning, &self.reasoning);
        }
        self.text = reply.message.text().unwrap_or_default();
        self.reasoning = reply.message.reasoning().unwrap_or_default();
        self.tool_turn_finished = reply.finish_reason == Some(ChatReplyFinishReason::ToolCalls);
    }

    fn raw(&self) -> (String, String) {
        (join_turns(&self.completed_text, &self.text), join_turns(&self.completed_reasoning, &self.reasoning))
    }

    fn effective(
        &self,
        promote_reasoning: bool,
    ) -> (String, String) {
        if promote_reasoning {
            // Tool-call reasoning is internal even when a model ignored the
            // disabled-reasoning preference for its final answer.
            (join_turns(&self.completed_text, &self.reasoning), self.completed_reasoning.clone())
        } else {
            self.raw()
        }
    }
}

fn final_parsed(
    text: &str,
    reasoning: &str,
    promoted_reasoning: bool,
) -> Option<Parsed> {
    let mut parsed = parsed_from(text, reasoning);
    if promoted_reasoning && let Some(parsed) = parsed.as_mut() {
        // The frontend merges parsed patches. Explicitly clear reasoning that
        // was streamed before we knew it belonged in the final answer.
        parsed.chain_of_thought = Some(sanitize(reasoning.trim()));
    }
    parsed
}

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
    let (session, support, naming) = ensure_session(
        app,
        state,
        &payload.repo_id,
        payload.model_chat_naming_enabled,
        payload.date_time_tool_enabled,
        payload.chart_tool_enabled,
    )
    .await?;
    if state.cancel_requested(&payload.run_id) {
        crate::logger::info("chat:run:cancelled-before-start", Some(serde_json::json!({ "runId": payload.run_id })));
        let _ = on_event.send(RunEvent::Done {
            text: String::new(),
            stats: map_reply_stats(None),
            finish_reason: Some(ChatReplyFinishReason::Cancelled),
            parsed: None,
            chat_name: None,
            transcript: Vec::new(),
        });
        return Ok(());
    }
    let effort = support.effective(payload.reasoning_effort);

    let reply_config = ChatReplyConfig::create().with_sampling_policy(sampling_policy(&payload.sampling_policy));

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
            let mut messages = build_messages(&payload.messages, effort);
            if naming.is_some() {
                add_naming_instruction(&mut messages);
            }
            messages
        },
    };
    state.forget_history();
    let naming_run = naming.as_ref().map(|naming| naming.begin(on_event.clone()));
    let (model_id, tools_enabled) = state
        .session
        .lock()
        .await
        .as_ref()
        .map(|resident| {
            (
                Some(resident.identifier.clone()),
                resident.model_chat_naming_enabled || resident.date_time_tool_enabled || resident.chart_tool_enabled,
            )
        })
        .unwrap_or_default();
    let tools_enabled = tools_enabled && session.supports_tool_calls().await;
    if let Some(model_id) = &model_id {
        app.state::<AnalyticsState>().report(|| Event::InferenceStarted {
            model_id: model_id.clone(),
        });
    }
    let stream = session.reply_with_stream(messages, reply_config).await;
    state.attach_cancel_token(&payload.run_id, stream.cancel_token());

    let should_promote_reasoning = effort == Some(ReasoningEffort::Disabled);
    let mut output: Option<ChatReply> = None;
    let mut reply_text = ReplyText::default();
    let mut transcript = Transcript::default();

    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::Error {
                error,
            } => {
                transcript.flush_reply(output.as_ref());
                let _ = on_event.send(RunEvent::Transcript {
                    items: transcript.items(true),
                });
                return Err(stream_error(error));
            },
            ChatSessionStreamChunk::ToolResults {
                messages,
            } => {
                transcript.tool_results(messages);
                let _ = on_event.send(RunEvent::Transcript {
                    items: transcript.items(false),
                });
            },
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                if let Some(reply) = replies.into_iter().next() {
                    let has_text = reply
                        .message
                        .content
                        .iter()
                        .any(|block| matches!(block, ChatContentBlock::Text { value } if !value.is_empty()));
                    let has_reasoning = reply
                        .message
                        .content
                        .iter()
                        .any(|block| matches!(block, ChatContentBlock::Reasoning { value } if !value.is_empty()));
                    let promote_while_streaming =
                        !tools_enabled && should_promote_reasoning && !has_text && has_reasoning;
                    if let Some(event) = transcript.stream_update(output.as_ref(), &reply, promote_while_streaming) {
                        let _ = on_event.send(event);
                    }
                    if reply.finish_reason.is_some() {
                        reply_text.update(&reply);
                    }
                    output = Some(reply);
                }
            },
        }
    }

    transcript.flush_reply(output.as_ref());
    if let Some(reply) = output.as_ref().filter(|reply| reply.finish_reason.is_none()) {
        reply_text.update(reply);
    }
    let finish_reason = output.as_ref().and_then(|r| r.finish_reason.as_ref());
    let (raw_text, raw_reasoning) = reply_text.raw();
    let promoted_reasoning = should_promote_reasoning
        && reply_text.text.is_empty()
        && !reply_text.reasoning.is_empty()
        && finish_reason.is_some_and(|reason| *reason != ChatReplyFinishReason::ToolCalls);
    let (text, reasoning) = reply_text.effective(promoted_reasoning);

    // Cancelled or truncated replies may diverge from what the client persists.
    if !stream.cancel_token().is_cancelled()
        && let (Some(ChatReplyFinishReason::Stop), Some(assistant)) =
            (finish_reason, assistant_history_message(&raw_text, &raw_reasoning, promoted_reasoning))
    {
        let mut history = payload.messages.clone();
        history.push(assistant);
        state.remember_history(&payload.repo_id, history, effort);
    }
    // Cancellation between a tool result and the next token has no assistant
    // snapshot of its own; the last reply still says ToolCalls.
    let finish_reason = if stream.cancel_token().is_cancelled() {
        Some(ChatReplyFinishReason::Cancelled)
    } else {
        finish_reason.cloned()
    };
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

    transcript.set_reasoning_promotion(promoted_reasoning);
    if let (Some(model_id), Some(reply)) = (model_id, &output) {
        app.state::<AnalyticsState>().report(|| Event::InferenceFinished {
            model_id,
            stats: Box::new(reply.stats.clone().into()),
        });
    }
    let _ = on_event.send(RunEvent::Done {
        text: sanitize(&text),
        stats: map_reply_stats(output.as_ref()),
        finish_reason,
        parsed: final_parsed(&text, &reasoning, promoted_reasoning),
        chat_name: naming_run.as_ref().and_then(|run| run.name()),
        transcript: transcript.items(true),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use uzu::types::session::chat::ChatMessage;

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

    fn reply(
        text: &str,
        reasoning: &str,
        finish_reason: Option<ChatReplyFinishReason>,
    ) -> ChatReply {
        ChatReply {
            message: ChatMessage::assistant().with_text(text.to_string()).with_reasoning(reasoning.to_string()),
            stats: Default::default(),
            finish_reason,
        }
    }

    #[test]
    fn final_text_preserves_completed_tool_turns_and_partial_last_turn() {
        let snapshots = [
            reply("Let", "First thought", None),
            reply("Let me check.", "First thought", Some(ChatReplyFinishReason::ToolCalls)),
            reply("", "Second", None),
            reply("", "Second thought", Some(ChatReplyFinishReason::ToolCalls)),
            reply("The", "Final thought", None),
            reply("The answer.", "Final thought", Some(ChatReplyFinishReason::Stop)),
        ];
        let mut text = ReplyText::default();
        for snapshot in &snapshots {
            if snapshot.finish_reason.is_some() {
                text.update(snapshot);
            }
        }
        assert_eq!(
            text.raw(),
            (
                "Let me check.\n\nThe answer.".to_string(),
                "First thought\n\nSecond thought\n\nFinal thought".to_string()
            )
        );

        let mut interrupted = ReplyText::default();
        for snapshot in &snapshots[..5] {
            if snapshot.finish_reason.is_some() {
                interrupted.update(snapshot);
            }
        }
        interrupted.update(&snapshots[4]);
        assert_eq!(
            interrupted.raw(),
            ("Let me check.\n\nThe".to_string(), "First thought\n\nSecond thought\n\nFinal thought".to_string())
        );
    }

    #[test]
    fn reasoning_promotion_keeps_tool_reasoning_out_of_the_answer() {
        let mut text = ReplyText::default();
        text.update(&reply("", "Name the chat", Some(ChatReplyFinishReason::ToolCalls)));
        text.update(&reply("", "Actual answer", Some(ChatReplyFinishReason::Stop)));
        assert_eq!(text.effective(true), ("Actual answer".to_string(), "Name the chat".to_string()));
        let parsed = final_parsed("Actual answer", "Name the chat", true).unwrap();
        assert_eq!(parsed.chain_of_thought.as_deref(), Some("Name the chat"));
        let parsed = final_parsed("Actual answer", "", true).unwrap();
        assert_eq!(parsed.chain_of_thought.as_deref(), Some(""));
        assert_eq!(parsed.response.as_deref(), Some("Actual answer"));
    }
}
