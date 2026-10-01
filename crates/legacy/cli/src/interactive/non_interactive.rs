use std::io::{self, Write};

use anyhow::{Context, Result, bail};
use iocraft::prelude::*;
use nagare::chat::ChatSessionStreamChunk;
use shoji::types::{
    basic::ReasoningEffort,
    session::chat::{ChatMessage, ChatReplyConfig},
};
use uzu::{
    engine::{Engine, EngineConfig},
    storage::DownloadPhase,
};

use crate::{
    common::thinking::ThinkingSupport,
    interactive::{
        APP_IDENTIFIER,
        components::{ApplicationState, TranscriptItem, chat_transcript_component},
        sessions::chat::{build_transcript, create_session},
    },
};

pub async fn run_non_interactive(
    model: String,
    message: String,
    reasoning_effort: Option<ReasoningEffort>,
    seed: Option<i64>,
) -> Result<()> {
    let preferences = ApplicationState::load_preferences().unwrap_or_default();
    let config = EngineConfig::default().with_application_identifier(APP_IDENTIFIER.to_string());
    let engine = Engine::new(config).await?;
    let model = engine.model(model.clone()).await?.with_context(|| format!("Model not found: {model}"))?;
    if !model.is_chat_capable() {
        bail!("--message requires a chat model");
    }

    if model.is_downloadable() {
        let download = engine.download(&model).await?;
        let mut announced = false;
        while let Some(update) = download.next().await {
            if !announced && update.is_in_progress() {
                eprintln!("Downloading {}", model.identifier);
                announced = true;
            }
        }
        if !engine.download_state(&model).await.is_some_and(|state| matches!(state.phase, DownloadPhase::Downloaded {}))
        {
            bail!("Model download did not complete: {}", model.identifier);
        }
    }

    let support = ThinkingSupport::for_model(&model);
    let effort = match reasoning_effort {
        Some(effort) => support
            .fulfill_requested_effort(effort)
            .map_err(|error| anyhow::anyhow!("Invalid --reasoning-effort: {error}"))?,
        None => support.with_preference(&preferences.thinking).reasoning_effort(),
    };
    let session = create_session(&engine, &model, seed, true).await?;
    let mut messages = Vec::new();
    if let Some(effort) = effort {
        messages.push(ChatMessage::system().with_reasoning_effort(effort));
    }
    messages.push(ChatMessage::user().with_text(message));
    let config = ChatReplyConfig::default().with_sampling_policy(preferences.sampling.policy());
    let stream = session.reply_with_stream(messages, config).await;
    let cancel_token = stream.cancel_token();
    let theme = preferences.theme;
    let result = async move {
        let mut stdout = io::stdout();
        let mut plain = PlainTranscript::default();
        let mut stats = None;
        while let Some(chunk) = stream.next().await {
            match chunk {
                ChatSessionStreamChunk::Replies {
                    replies,
                } => {
                    if let Some(reply) = replies.last() {
                        stats = Some(reply.stats.clone());
                    }
                    let items = build_transcript(&session.messages().await, 0);
                    plain.write(&mut stdout, items, false)?;
                },
                ChatSessionStreamChunk::Error {
                    error,
                } => return Err(error.into()),
            }
        }

        let stats = stats.context("No response generated")?;
        let items = build_transcript(&session.messages().await, 0);
        plain.write(&mut stdout, items, true)?;
        writeln!(stdout)?;
        chat_transcript_component(
            Vec::new(),
            Some(stats),
            theme.subtitle_color,
            theme.overlay_color(),
            theme.padding(),
            false,
        )
        .write(&mut stdout)?;
        stdout.flush()?;
        Ok(())
    }
    .await;
    cancel_token.cancel();
    result
}

#[derive(Default)]
struct PlainTranscript {
    reasoning: String,
    text: String,
    last_was_thinking: Option<bool>,
}

impl PlainTranscript {
    fn write(
        &mut self,
        output: &mut impl Write,
        items: Vec<TranscriptItem>,
        finished: bool,
    ) -> io::Result<()> {
        let mut reasoning = String::new();
        let mut text = String::new();
        for item in items {
            match item {
                TranscriptItem::Thinking(value) => reasoning.push_str(&value),
                TranscriptItem::Text(value) => text.push_str(&value),
                TranscriptItem::ToolCall {
                    ..
                } => {},
            }
        }
        if self.last_was_thinking == Some(false)
            && reasoning.trim_end_matches(|ch: char| ch == '\u{fffd}' || ch.is_whitespace()).len()
                > self.reasoning.len()
        {
            self.write_part(output, &text, false, true)?;
        }
        self.write_part(output, &reasoning, true, finished)?;
        if self.last_was_thinking == Some(true) && text.trim_end_matches('\u{fffd}').len() > self.text.len() {
            self.write_part(output, &reasoning, true, true)?;
        }
        self.write_part(output, &text, false, finished)?;
        output.flush()
    }

    fn write_part(
        &mut self,
        output: &mut impl Write,
        text: &str,
        is_thinking: bool,
        finished: bool,
    ) -> io::Result<()> {
        let previous = if is_thinking {
            &mut self.reasoning
        } else {
            &mut self.text
        };
        let delta = text.strip_prefix(previous.as_str()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Model output changed text that was already written")
        })?;
        let delta = if finished {
            delta
        } else {
            delta.trim_end_matches(|ch: char| ch == '\u{fffd}' || (is_thinking && ch.is_whitespace()))
        };
        if delta.is_empty() {
            return Ok(());
        }
        if self.last_was_thinking.is_some_and(|last| last != is_thinking) {
            writeln!(output)?;
        }
        output.write_all(delta.as_bytes())?;
        previous.push_str(delta);
        self.last_was_thinking = Some(is_thinking);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../unit/interactive/non_interactive_test.rs"]
mod tests;
