//! The chat inference effect. `send`/`regenerate` kick off a reply; a Tokio
//! producer streams cumulative chunks back through an `mpsc` channel, and
//! `apply_stream` (the reducer) folds each into the trailing assistant message.

use futures::{StreamExt, channel::mpsc};
use gpui::Context;
use uzu::{
    session::chat::{ChatSession, ChatSessionStreamChunk},
    types::{
        basic::CancelToken,
        model::Model,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

use super::{
    conversation::{ChatMsg, Role, Version, conversation_for_request},
    sampling::sampling_method,
    view::{ChatEvent, ChatView},
};
use crate::{engine, persistence, title_gen};

/// Messages bridged from the Tokio reply stream to the UI.
pub(super) enum StreamMsg {
    Started(CancelToken),
    Session(ChatSession),
    DropSession,
    Update {
        text: String,
        reasoning: Option<String>,
        tps: Option<f32>,
        tokens: Option<u32>,
    },
    Done,
    Error(String),
}

impl ChatView {
    pub(super) fn send(&mut self, text: String, cx: &mut Context<Self>) {
        if self.state.streaming {
            return;
        }
        let text = text.trim().to_string();
        if text.is_empty() && self.state.attached_files.is_empty() {
            return;
        }
        // Append attached files as fenced code blocks (Electron parity).
        let full_text = if self.state.attached_files.is_empty() {
            text
        } else {
            let mut s = text;
            for (name, ext, content) in self.state.attached_files.drain(..) {
                s.push_str(&format!("\n\n```{ext}\n# {name}\n{content}\n```"));
            }
            s
        };
        let first_user = !self.state.messages.iter().any(|m| m.role == Role::User);
        self.state.messages.push(ChatMsg::user(full_text));
        if first_user {
            self.state.title_pending = true;
        }
        self.run_inference(cx);
    }

    /// Start a fresh assistant reply for the latest turn.
    fn run_inference(&mut self, cx: &mut Context<Self>) {
        let model_name = self.state.model.as_ref().map(|m| m.name());
        self.state.messages.push(ChatMsg::assistant(Version { model_name, ..Default::default() }));
        self.state.streaming = true;
        self.state.waiting_for_model = true;
        self.spawn_reply(cx);
    }

    /// Re-run an assistant turn (optionally switching model) as a new version.
    pub(super) fn regenerate_at_with_model(
        &mut self,
        msg_idx: usize,
        model: Option<Model>,
        cx: &mut Context<Self>,
    ) {
        if self.state.streaming {
            return;
        }
        let Some(msg) = self.state.messages.get(msg_idx) else {
            return;
        };
        if msg.role != Role::Assistant {
            return;
        }
        if let Some(model) = model {
            self.state.model = Some(model);
            self.clear_session();
        }
        self.state.messages.truncate(msg_idx + 1);
        let model_name = self.state.model.as_ref().map(|m| m.name());
        if let Some(last) = self.state.messages.last_mut() {
            last.versions.push(Version { model_name, ..Default::default() });
            last.current = last.versions.len() - 1;
        }
        self.close_popovers();
        self.state.streaming = true;
        self.spawn_reply(cx);
    }

    /// Resolve the model, build the request from the conversation (excluding the
    /// trailing assistant placeholder), and stream into its current version.
    fn spawn_reply(&mut self, cx: &mut Context<Self>) {
        let Some(model) = self.resolved_model(cx) else {
            if let Some(last) = self.state.messages.last_mut() {
                let v = last.cur_mut();
                v.text = "No local model installed yet. Open Local Models to download one."
                    .to_string();
                v.error = true;
            }
            self.state.streaming = false;
            cx.notify();
            return;
        };
        self.state.model = Some(model.clone());
        self.state.loaded_model = Some(model.name());

        // Global instructions + prior messages, excluding the trailing assistant
        // placeholder being filled and any errored turns.
        let mut history: Vec<ChatMessage> = Vec::new();
        let instructions = persistence::global_instructions();
        if !instructions.trim().is_empty() {
            history.push(ChatMessage::system().with_text(instructions));
        }
        history.extend(conversation_for_request(&self.state.messages).into_iter().map(
            |(role, text)| match role {
                Role::User => ChatMessage::user().with_text(text),
                Role::Assistant => ChatMessage::assistant().with_text(text),
            },
        ));

        let Some(engine) = engine::try_engine(cx) else {
            self.apply_stream(StreamMsg::Error("engine unavailable".to_string()), cx);
            return;
        };

        let method = sampling_method(
            self.state.sampling_mode,
            self.state.temperature,
            self.state.top_k,
            self.state.top_p,
            self.state.min_p,
        );
        let mut reply_config = ChatReplyConfig::default();
        if let Some(method) = method {
            reply_config = reply_config.with_sampling_method(method);
        }
        let reply_config = reply_config
            .with_token_limit((self.state.max_tokens > 0).then_some(self.state.max_tokens));

        let model_id = model.identifier.clone();
        let cached_session = self.cached_session(&model_id);

        let (tx, mut rx) = mpsc::unbounded::<StreamMsg>();

        // Producer: run uzu on the Tokio runtime, never touching view state.
        gpui_tokio::Tokio::spawn(cx, async move {
            let session = match cached_session {
                Some(session) => session,
                None => match engine.chat(model.clone(), ChatConfig::default()).await {
                    Ok(session) => {
                        let _ = tx.unbounded_send(StreamMsg::Session(session.clone()));
                        session
                    }
                    Err(err) => {
                        let _ = tx.unbounded_send(StreamMsg::Error(err.to_string()));
                        return;
                    }
                },
            };
            if let Err(err) = session.reset().await {
                let _ = tx.unbounded_send(StreamMsg::DropSession);
                let _ = tx.unbounded_send(StreamMsg::Error(format!("{err:?}")));
                return;
            }
            let stream = session.reply_with_stream(history, reply_config).await;
            let _ = tx.unbounded_send(StreamMsg::Started(stream.cancel_token()));
            while let Some(chunk) = stream.next().await {
                match chunk {
                    ChatSessionStreamChunk::Replies { replies } => {
                        if let Some(reply) = replies.into_iter().next() {
                            let _ = tx.unbounded_send(StreamMsg::Update {
                                text: reply.message.text().unwrap_or_default(),
                                reasoning: reply.message.reasoning(),
                                tps: reply.stats.generate_tokens_per_second.map(|v| v as f32),
                                tokens: reply.stats.tokens_count_output,
                            });
                        }
                    }
                    ChatSessionStreamChunk::Error { error } => {
                        let _ = tx.unbounded_send(StreamMsg::Error(format!("{error:?}")));
                    }
                }
            }
            let _ = tx.unbounded_send(StreamMsg::Done);
        })
        .detach();

        // Consumer: fold updates into the trailing assistant message.
        cx.spawn(async move |this, cx| {
            while let Some(msg) = rx.next().await {
                if this.update(cx, |view, cx| view.apply_stream(msg, cx)).is_err() {
                    break;
                }
            }
        })
        .detach();

        self.pin_to_bottom();
        cx.notify();
    }

    fn apply_stream(&mut self, msg: StreamMsg, cx: &mut Context<Self>) {
        match msg {
            StreamMsg::Started(token) => {
                self.state.cancel = Some(token);
                self.state.waiting_for_model = false;
            }
            StreamMsg::Session(session) => {
                if let Some(id) = self.state.model.as_ref().map(|m| m.identifier.clone()) {
                    self.store_session(session, &id);
                }
            }
            StreamMsg::DropSession => self.clear_session(),
            StreamMsg::Update {
                text,
                reasoning,
                tps,
                tokens,
            } => {
                if let Some(last) = self.state.messages.last_mut() {
                    if last.role == Role::Assistant {
                        let had_text = !last.cur().text.is_empty();
                        let v = last.cur_mut();
                        v.text = text;
                        v.reasoning = reasoning;
                        v.tps = tps;
                        v.tokens = tokens;
                        // Auto-collapse reasoning once the reply body starts
                        // arriving (mirrors Electron behaviour).
                        if !had_text && !last.cur().text.is_empty() {
                            last.reasoning_collapsed = true;
                        }
                    }
                }
                if self.should_auto_scroll() {
                    self.pin_to_bottom();
                }
                // Keep the live reasoning panel scrolled to its newest line.
                self.reasoning_scroll.scroll_to_bottom();
                cx.notify();
            }
            StreamMsg::Done => {
                self.state.streaming = false;
                self.state.waiting_for_model = false;
                self.state.cancel = None;
                // If the model produced no text, show a notice rather than an
                // empty bubble stuck on "…".
                if let Some(last) = self.state.messages.last_mut() {
                    if last.role == Role::Assistant {
                        let v = last.cur_mut();
                        if !v.error && v.text.is_empty() {
                            v.text = "(The model returned no text.)".to_string();
                            v.error = true;
                        }
                    }
                }
                self.save();
                self.maybe_generate_title(cx);
                cx.notify();
            }
            StreamMsg::Error(err) => {
                if let Some(last) = self.state.messages.last_mut() {
                    if last.role == Role::Assistant {
                        let v = last.cur_mut();
                        v.text = format!("Error: {err}");
                        v.error = true;
                    }
                }
                self.state.streaming = false;
                self.state.waiting_for_model = false;
                self.state.cancel = None;
                crate::toast::push(cx, "Inference failed", crate::toast::ToastKind::Error);
                cx.notify();
            }
        }
    }

    pub(super) fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(token) = &self.state.cancel {
            token.cancel();
        }
        self.state.streaming = false;
        self.state.waiting_for_model = false;
        self.state.cancel = None;
        cx.notify();
    }

    fn maybe_generate_title(&mut self, cx: &mut Context<Self>) {
        if !self.state.title_pending {
            return;
        }
        let Some(model) = self.state.model.clone() else {
            return;
        };
        let Some(user_text) = self.state
            .messages
            .iter()
            .find(|m| m.role == Role::User)
            .map(|m| m.cur().text.clone())
        else {
            return;
        };
        self.state.title_pending = false;

        let Some(engine) = engine::try_engine(cx) else {
            return;
        };
        let (tx, mut rx) = mpsc::unbounded::<Result<String, String>>();
        gpui_tokio::Tokio::spawn(cx, async move {
            let _ = tx.unbounded_send(title_gen::run(&engine, model, &user_text).await);
        })
        .detach();
        cx.spawn(async move |this, cx| {
            if let Some(Ok(title)) = rx.next().await {
                let _ = this.update(cx, |view, cx| {
                    view.state.chat_title = title;
                    view.save();
                    cx.emit(ChatEvent::Updated);
                    cx.notify();
                });
            }
        })
        .detach();
    }
}
