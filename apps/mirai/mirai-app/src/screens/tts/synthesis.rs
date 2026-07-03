use futures::{StreamExt, channel::mpsc};
use gpui::Context;
use gpui_tokio::Tokio;
use uzu::{
    player::Player,
    session::text_to_speech::TextToSpeechSessionStreamChunk,
    types::{basic::PcmBatch, model::Model},
};

use super::{
    pending_generation::PendingGeneration,
    synthesis_message::SynthesisMessage,
    view::{CHAR_LIMIT, TtsView},
};
use crate::{engine, tts_history};

impl TtsView {
    fn generate(
        &mut self,
        text: String,
        cx: &mut Context<Self>,
    ) {
        if self.generating {
            return;
        }
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        if text.chars().count() > CHAR_LIMIT {
            self.error = Some(format!("Text exceeds the {CHAR_LIMIT}-character limit."));
            cx.notify();
            return;
        }
        let Some(model) = self.resolved_model(cx) else {
            self.error = Some("Download and select a voice model first.".to_string());
            cx.notify();
            return;
        };

        if let Some(player) = &self.player {
            player.stop();
        }
        self.selected = Some(model.clone());
        self.generating = true;
        self.generation_id = self.generation_id.wrapping_add(1);
        let generation_id = self.generation_id;
        self.error = None;
        self.playing_id = None;
        self.pending_batches.clear();
        self.pending_gen = Some(PendingGeneration {
            text: text.clone(),
            model: model.clone(),
            vendor: self
                .store
                .read(cx)
                .rows
                .iter()
                .find(|r| r.model.identifier == model.identifier)
                .and_then(|r| r.vendor())
                .unwrap_or_else(|| "Other".to_string()),
        });
        cx.notify();

        let Some(engine) = engine::try_engine(cx) else {
            self.generating = false;
            self.error = Some("engine unavailable".to_string());
            cx.notify();
            return;
        };

        let (tx, mut rx) = mpsc::unbounded::<SynthesisMessage>();
        Tokio::spawn(cx, async move {
            let session = match engine.text_to_speech(model).await {
                Ok(session) => session,
                Err(err) => {
                    let _ = tx.unbounded_send(SynthesisMessage::Error(err.to_string()));
                    return;
                },
            };
            let stream = session.synthesize_stream(text).await;
            let _ = tx.unbounded_send(SynthesisMessage::Started(stream.cancel_token()));
            while let Some(event) = stream.next().await {
                match event {
                    TextToSpeechSessionStreamChunk::Output {
                        output,
                    } => {
                        let _ = tx.unbounded_send(SynthesisMessage::Batch(output.pcm_batch));
                    },
                    TextToSpeechSessionStreamChunk::Error {
                        error,
                    } => {
                        let _ = tx.unbounded_send(SynthesisMessage::Error(format!("{error}")));
                    },
                }
            }
            let _ = tx.unbounded_send(SynthesisMessage::Done);
        })
        .detach();

        cx.spawn(async move |this, cx| {
            while let Some(message) = rx.next().await {
                let still_current = this.update(cx, |view, cx| {
                    if view.generation_id != generation_id {
                        return false;
                    }
                    view.apply(message, cx);
                    true
                });
                if !matches!(still_current, Ok(true)) {
                    break;
                }
            }
        })
        .detach();
    }

    fn apply(
        &mut self,
        msg: SynthesisMessage,
        cx: &mut Context<Self>,
    ) {
        match msg {
            SynthesisMessage::Started(token) => self.cancel = Some(token),
            SynthesisMessage::Batch(batch) => {
                if !self.generating {
                    return;
                }
                self.pending_batches.push(batch.clone());
                if let Err(err) = self.append_pcm(batch) {
                    self.error = Some(err);
                    self.generating = false;
                }
            },
            SynthesisMessage::Done => {
                self.generating = false;
                self.cancel = None;
                if self.error.is_none()
                    && let Some(pending) = self.pending_gen.as_ref()
                    && tts_history::save_generation(
                        &pending.model,
                        &pending.vendor,
                        &pending.text,
                        &self.pending_batches,
                    )
                    .is_some()
                {
                    self.reload_history();
                }
                self.clear_gen();
                cx.notify();
            },
            SynthesisMessage::Error(err) => {
                self.error = Some(err);
                self.generating = false;
                self.cancel = None;
                self.clear_gen();
                cx.notify();
            },
        }
    }

    pub(super) fn resolved_model(
        &self,
        cx: &Context<Self>,
    ) -> Option<Model> {
        self.store.read(cx).resolve_installed(self.selected.as_ref())
    }

    pub(super) fn append_pcm(
        &mut self,
        batch: PcmBatch,
    ) -> Result<(), String> {
        if self.player.is_none() {
            self.player = Some(Player::new().map_err(|e| format!("audio: {e}"))?);
        }
        self.player.as_ref().expect("player just opened").append_pcm_batch(batch).map_err(|e| format!("audio: {e}"))
    }

    fn clear_gen(&mut self) {
        self.pending_batches.clear();
        self.pending_gen = None;
    }

    pub(super) fn generate_from_button(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let text = self.input.read(cx).text();
        self.generate(text, cx);
    }

    pub(super) fn stop(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        self.stop_playback(cx);
        self.generating = false;
        self.generation_id = self.generation_id.wrapping_add(1);
        self.cancel = None;
        self.clear_gen();
        cx.notify();
    }
}
