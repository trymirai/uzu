use std::fmt::Write;

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
    chat_turn::ChatTurn, conversation::conversation_for_request, event::ChatEvent, role::Role,
    sampling::sampling_method, version::Version, view::ChatView,
};
use crate::{components::markdown, engine, persistence, title_gen};

const REVEAL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);
const MIN_REVEAL_CHARS: usize = 3;
const REVEAL_CATCHUP_DIVISOR: usize = 8;

fn safe_split(prefix: &str) -> usize {
    let mut boundary = 0;
    let mut index = 0;
    while let Some(offset) = prefix[index..].find("\n\n") {
        let candidate = index + offset + 2;
        if prefix[..candidate].matches("```").count().is_multiple_of(2) {
            boundary = candidate;
        }
        index = candidate;
    }
    boundary
}

pub(super) enum StreamMsg {
    Started(CancelToken),
    Session(ChatSession, String),
    DropSession,
    Update {
        text: String,
        reasoning: Option<String>,
        tps: Option<f32>,
        tokens: Option<u32>,
        ttft: Option<f32>,
        total_time: Option<f32>,
    },
    Done,
    Error(String),
}

impl ChatView {
    pub(super) fn send(
        &mut self,
        text: String,
        cx: &mut Context<Self>,
    ) {
        if self.state.streaming {
            return;
        }
        self.last_active = std::time::Instant::now();
        let text = text.trim().to_string();
        if text.is_empty() && self.state.attached_files.is_empty() {
            return;
        }

        let full_text = if self.state.attached_files.is_empty() {
            text
        } else {
            let mut combined = text;
            for (name, extension, content) in self.state.attached_files.drain(..) {
                let _ = write!(combined, "\n\n```{extension}\n# {name}\n{content}\n```");
            }
            combined
        };
        let first_user = !self.state.messages.iter().any(|m| m.role == Role::User);
        self.state.messages.push(ChatTurn::user(full_text));
        if first_user {
            self.state.title_pending = true;
        }
        self.run_inference(cx);
    }

    fn run_inference(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let model_name = self.state.model.as_ref().map(|m| m.name());
        self.state.messages.push(ChatTurn::assistant(Version {
            model_name,
            ..Default::default()
        }));
        self.state.streaming = true;
        self.state.waiting_for_model = true;
        self.spawn_reply(cx);
    }

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
            last.versions.push(Version {
                model_name,
                ..Default::default()
            });
            last.current = last.versions.len() - 1;
        }
        self.close_popovers();
        self.state.streaming = true;
        self.state.waiting_for_model = true;
        self.spawn_reply(cx);
    }

    fn spawn_reply(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let Some(model) = self.resolved_model(cx) else {
            if let Some(last) = self.state.messages.last_mut() {
                let v = last.cur_mut();
                v.text = "No local model installed yet. Open Local Models to download one.".to_string();
                v.error = true;
            }
            self.state.streaming = false;
            cx.notify();
            return;
        };
        self.state.model = Some(model.clone());
        self.state.loaded_model = Some(model.name());

        let mut history: Vec<ChatMessage> = Vec::new();
        let instructions = persistence::global_instructions();
        if !instructions.trim().is_empty() {
            history.push(ChatMessage::system().with_text(instructions));
        }
        history.extend(conversation_for_request(&self.state.messages).into_iter().map(|(role, text)| match role {
            Role::User => ChatMessage::user().with_text(text),
            Role::Assistant => ChatMessage::assistant().with_text(text),
        }));

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
        let reply_config = reply_config.with_token_limit((self.state.max_tokens > 0).then_some(self.state.max_tokens));

        let model_id = model.identifier.clone();
        let cached_session = self.cached_session(&model_id);

        self.state.stream_gen = self.state.stream_gen.wrapping_add(1);
        let gen_id = self.state.stream_gen;
        self.state.revealed_chars = 0;
        self.state.stream_parsed = None;
        self.state.stream_stable_len = 0;
        self.state.stream_parse_in_flight = false;
        self.state.stream_parse_pending = false;

        let (tx, mut rx) = mpsc::unbounded::<StreamMsg>();

        gpui_tokio::Tokio::spawn(cx, async move {
            let session = match cached_session {
                Some(session) => session,
                None => match engine.chat(model.clone(), ChatConfig::default()).await {
                    Ok(session) => {
                        let _ = tx.unbounded_send(StreamMsg::Session(session.clone(), model.identifier.clone()));
                        session
                    },
                    Err(err) => {
                        let _ = tx.unbounded_send(StreamMsg::Error(err.to_string()));
                        return;
                    },
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
                    ChatSessionStreamChunk::Replies {
                        replies,
                    } => {
                        if let Some(reply) = replies.into_iter().next() {
                            let _ = tx.unbounded_send(StreamMsg::Update {
                                text: reply.message.text().unwrap_or_default(),
                                reasoning: reply.message.reasoning(),
                                tps: reply.stats.generate_tokens_per_second.map(|v| v as f32),
                                tokens: reply.stats.tokens_count_output,
                                ttft: reply.stats.time_to_first_token.map(|v| v as f32),
                                total_time: Some(reply.stats.duration as f32),
                            });
                        }
                    },
                    ChatSessionStreamChunk::Error {
                        error,
                    } => {
                        let _ = tx.unbounded_send(StreamMsg::Error(format!("{error:?}")));
                    },
                }
            }
            let _ = tx.unbounded_send(StreamMsg::Done);
        })
        .detach();

        cx.spawn(async move |this, cx| {
            while let Some(message) = rx.next().await {
                let keep = this.update(cx, |view, cx| {
                    if view.state.stream_gen != gen_id {
                        return false;
                    }
                    view.apply_stream(message, cx);
                    true
                });
                if !matches!(keep, Ok(true)) {
                    break;
                }
            }
        })
        .detach();

        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(REVEAL_INTERVAL).await;
                let keep = this.update(cx, |view, cx| {
                    if view.state.stream_gen != gen_id {
                        return false;
                    }
                    view.advance_reveal(cx)
                });
                if !matches!(keep, Ok(true)) {
                    break;
                }
            }
        })
        .detach();

        self.pin_to_bottom();
        cx.notify();
    }

    fn last_assistant_mut(&mut self) -> Option<&mut ChatTurn> {
        self.state.messages.last_mut().filter(|message| message.role == Role::Assistant)
    }

    fn advance_reveal(
        &mut self,
        cx: &mut Context<Self>,
    ) -> bool {
        let full = {
            let Some(message) = self.state.messages.last() else {
                return false;
            };
            if message.role != Role::Assistant {
                return false;
            }
            message.cur().text.chars().count()
        };
        if self.state.revealed_chars < full {
            let remaining = full - self.state.revealed_chars;
            let step = (remaining / REVEAL_CATCHUP_DIVISOR).max(MIN_REVEAL_CHARS);
            self.state.revealed_chars = self.state.revealed_chars.saturating_add(step).min(full);
            self.request_stream_parse(cx);
            cx.notify();
            return true;
        }
        self.state.streaming
    }

    fn request_stream_parse(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        if self.state.stream_parse_in_flight {
            self.state.stream_parse_pending = true;
            return;
        }
        let Some(message) = self.state.messages.last() else {
            return;
        };
        if message.role != Role::Assistant {
            return;
        }
        let revealed: String = message.cur().text.chars().take(self.state.revealed_chars).collect();
        let boundary = safe_split(&revealed);
        if boundary == 0 {
            return;
        }
        let stable = revealed[..boundary].to_string();
        self.state.stream_parse_in_flight = true;
        let gen_id = self.state.stream_gen;
        let parse_task = cx.background_executor().spawn(async move { markdown::parse(&stable) });
        cx.spawn(async move |this, cx| {
            let parsed = parse_task.await;
            let _ = this.update(cx, |view, cx| {
                if view.state.stream_gen != gen_id {
                    return;
                }
                view.state.stream_parsed = Some(parsed);
                view.state.stream_stable_len = boundary;
                view.state.stream_parse_in_flight = false;
                cx.notify();
                if view.state.stream_parse_pending {
                    view.state.stream_parse_pending = false;
                    view.request_stream_parse(cx);
                }
            });
        })
        .detach();
    }

    fn apply_stream(
        &mut self,
        msg: StreamMsg,
        cx: &mut Context<Self>,
    ) {
        match msg {
            StreamMsg::Started(token) => {
                self.state.cancel = Some(token);
                self.state.waiting_for_model = false;
            },
            StreamMsg::Session(session, model_id) => {
                self.store_session(session, &model_id);
            },
            StreamMsg::DropSession => self.clear_session(),
            StreamMsg::Update {
                text,
                reasoning,
                tps,
                tokens,
                ttft,
                total_time,
            } => {
                if let Some(message) = self.last_assistant_mut() {
                    let had_text = !message.cur().text.is_empty();
                    let version = message.cur_mut();
                    version.text = text;
                    version.reasoning = reasoning;
                    version.tps = tps;
                    version.tokens = tokens;
                    version.ttft = ttft;
                    version.total_time = total_time;
                    if !had_text && !message.cur().text.is_empty() {
                        message.reasoning_collapsed = true;
                    }
                }
                if self.should_auto_scroll() {
                    self.pin_to_bottom();
                }
                self.reasoning_scroll.scroll_to_bottom();
                cx.notify();
            },
            StreamMsg::Done => {
                self.state.streaming = false;
                self.state.waiting_for_model = false;
                self.state.cancel = None;
                self.last_active = std::time::Instant::now();
                if let Some(message) = self.last_assistant_mut() {
                    let version = message.cur_mut();
                    if !version.error && version.text.is_empty() {
                        version.text = "(The model returned no text.)".to_string();
                        version.error = true;
                    }
                }
                self.save();
                cx.emit(ChatEvent::Updated);
                self.maybe_generate_title(cx);
                cx.notify();
            },
            StreamMsg::Error(err) => {
                if let Some(message) = self.last_assistant_mut() {
                    let version = message.cur_mut();
                    version.text = format!("Error: {err}");
                    version.error = true;
                }
                self.state.streaming = false;
                self.state.waiting_for_model = false;
                self.state.cancel = None;
                self.save();
                cx.emit(ChatEvent::Updated);
                crate::toast::push(cx, "Inference failed", crate::toast::ToastKind::Error);
                cx.notify();
            },
        }
    }

    pub(super) fn stop(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.cancel_stream();
        cx.notify();
    }

    fn maybe_generate_title(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        if !self.state.title_pending {
            return;
        }
        let Some(model) = self.state.model.clone() else {
            return;
        };
        let Some(user_text) = self.state.messages.iter().find(|m| m.role == Role::User).map(|m| m.cur().text.clone())
        else {
            return;
        };
        self.state.title_pending = false;
        let chat_id = self.state.chat_id.clone();

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
                    if view.state.chat_id != chat_id {
                        return;
                    }
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
