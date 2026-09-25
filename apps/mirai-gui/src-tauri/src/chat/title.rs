use tauri::AppHandle;
use uzu::{
    session::chat::ChatSessionStreamChunk,
    types::{
        basic::SamplingPolicy,
        session::chat::{ChatReply, ChatReplyConfig},
    },
};

use super::{
    ChatState,
    messages::{build_messages, sanitize},
    payloads::TitleGenPayload,
    session::ensure_session,
};
use crate::error::AppResult;

const DEFAULT_TITLE_GEN_TOKENS_LIMIT: u32 = 200;
pub(super) const TITLE_GEN_RUN_ID: &str = "title-gen";

pub(super) async fn title_gen_inner(
    app: &AppHandle,
    state: &ChatState,
    payload: &TitleGenPayload,
) -> AppResult<String> {
    let _run_guard = state.run_lock.lock().await;
    let (session, support) = ensure_session(app, state, &payload.repo_id).await?;
    if state.cancel_requested(TITLE_GEN_RUN_ID) {
        return Ok(String::new());
    }

    let messages = build_messages(&payload.messages, support.cheapest());

    let reply_config = ChatReplyConfig::create()
        .with_token_limit(Some(DEFAULT_TITLE_GEN_TOKENS_LIMIT))
        .with_sampling_policy(SamplingPolicy::Default {});

    // reset() invalidates the KV prefix tracked for chat runs.
    state.forget_history();
    if let Err(error) = session.reset().await {
        state.evict(app, &payload.repo_id).await;
        return Err(error.into());
    }
    let stream = session.reply_with_stream(messages, reply_config).await;
    state.attach_cancel_token(TITLE_GEN_RUN_ID, stream.cancel_token());

    let mut output: Option<ChatReply> = None;
    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::Error {
                error,
            } => return Err(error.into()),
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                if let Some(reply) = replies.into_iter().next() {
                    output = Some(reply);
                }
            },
        }
    }
    let text = output.as_ref().and_then(|r| r.message.text()).unwrap_or_default();
    let reasoning = output.as_ref().and_then(|r| r.message.reasoning()).unwrap_or_default();
    let title = if text.is_empty() {
        reasoning
    } else {
        text
    };
    Ok(sanitize(title.trim()))
}
