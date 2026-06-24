//! Chat screen: streaming local inference with a reasoning panel, perf stats,
//! and a composer. Streaming follows the Tokio→channel→foreground pattern: the
//! uzu reply stream runs on the Tokio runtime and pushes cumulative updates back
//! to the UI entity.

use futures::{StreamExt, channel::mpsc};
use gpui::{
    Context, Entity, FontWeight, IntoElement, Render, ScrollHandle, Window, div, prelude::*, px,
};
use uzu::{
    session::chat::ChatSessionStreamChunk,
    types::{
        basic::{CancelToken, SamplingMethod},
        model::Model,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

use crate::{
    components::{Icon, IconButton, IconEl, InputEvent, Loader, SegmentedControl, TextInput},
    engine,
    models_store::ModelsStore,
    persistence::{self, StoredChat, StoredMessage},
    settings_state,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
}

/// Sampling mode, mirroring mirai-chat's Default / Argmax / Stochastic segmented
/// control. `Default` defers to the model's generation config.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SamplingMode {
    Default,
    Argmax,
    Stochastic,
}

struct ChatMsg {
    role: Role,
    text: String,
    reasoning: Option<String>,
    tps: Option<f32>,
    tokens: Option<u32>,
    error: bool,
}

/// Messages bridged from the Tokio reply stream to the UI.
enum StreamMsg {
    Started(CancelToken),
    Update {
        text: String,
        reasoning: Option<String>,
        tps: Option<f32>,
        tokens: Option<u32>,
    },
    Done,
    Error(String),
}

pub struct ChatView {
    store: Entity<ModelsStore>,
    input: Entity<TextInput>,
    messages: Vec<ChatMsg>,
    model: Option<Model>,
    streaming: bool,
    cancel: Option<CancelToken>,
    chat_id: Option<String>,
    created_at: u64,
    scroll: ScrollHandle,
    model_picker_open: bool,
    /// Name of the model the UI considers "loaded" (set once a chat runs).
    /// uzu has no unload API, so eject is a UI-level indicator/deselect.
    loaded_model: Option<String>,
    // Generation settings.
    gen_settings_open: bool,
    sampling_mode: SamplingMode,
    temperature: f32,
    /// Top-K; 0 = off (None).
    top_k: u32,
    /// Top-P nucleus; 0.0 = off (None).
    top_p: f32,
    /// Min-P; 0.0 = off (None).
    min_p: f32,
    /// Max output tokens; 0 = unlimited.
    max_tokens: u32,
}

impl ChatView {
    pub fn new(store: Entity<ModelsStore>, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| TextInput::new(cx, "Message Mirai…"));
        cx.subscribe(&input, |this, _input, event, cx| match event {
            InputEvent::Submit(text) => this.send(text.clone(), cx),
        })
        .detach();
        cx.observe(&store, |_, _, cx| cx.notify()).detach();
        settings_state::observe(cx, |_, cx| cx.notify()).detach();
        Self {
            store,
            input,
            messages: Vec::new(),
            model: None,
            streaming: false,
            cancel: None,
            chat_id: None,
            created_at: persistence::now_ms(),
            scroll: ScrollHandle::new(),
            model_picker_open: false,
            loaded_model: None,
            gen_settings_open: false,
            sampling_mode: SamplingMode::Default,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            max_tokens: 0,
        }
    }

    /// The model the footer shows as loaded (None → "No model loaded").
    pub fn loaded_model_name(&self) -> Option<String> {
        self.loaded_model.clone()
    }

    /// "Eject" the loaded model: stop any generation and clear the loaded
    /// indicator. Note: uzu exposes no unload API, so this does not free GPU
    /// memory — it's a UI deselect. The picked model (`self.model`) is kept, so
    /// the next message reloads it.
    pub fn eject(&mut self, cx: &mut Context<Self>) {
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        self.streaming = false;
        self.cancel = None;
        self.loaded_model = None;
        cx.notify();
    }

    /// Overlay listing installed local models; picking one pins it for this chat.
    fn model_picker_overlay(&self, cx: &mut Context<Self>) -> Option<gpui::AnyElement> {
        if !self.model_picker_open {
            return None;
        }
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let installed: Vec<(String, String)> = self
            .store
            .read(cx)
            .rows
            .iter()
            .filter(|r| r.is_installed())
            .map(|r| (r.id().to_string(), r.name()))
            .collect();

        let mut list = div()
            .id("model-picker-list")
            .flex()
            .flex_col()
            .gap_1()
            .max_h(px(360.))
            .overflow_y_scroll();
        if installed.is_empty() {
            list = list.child(
                div()
                    .p_3()
                    .text_sm()
                    .text_color(theme.text_muted)
                    .child("No installed models — download one in Local Models."),
            );
        } else {
            for (id, name) in installed {
                list = list.child(
                    div()
                        .id(gpui::SharedString::from(format!("pick-{id}")))
                        .flex()
                        .items_center()
                        .h(px(40.))
                        .px_3()
                        .rounded_md()
                        .text_color(theme.text)
                        .cursor(gpui::CursorStyle::PointingHand)
                        .hover(move |s| s.bg(hover))
                        .child(name)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if let Some(model) = this
                                .store
                                .read(cx)
                                .rows
                                .iter()
                                .find(|r| r.id() == id)
                                .map(|r| r.model.clone())
                            {
                                this.model = Some(model);
                            }
                            this.model_picker_open = false;
                            cx.notify();
                        })),
                );
            }
        }

        Some(
            div()
                .id("model-picker-backdrop")
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .bg(gpui::black().opacity(0.5))
                .occlude()
                .on_click(cx.listener(|this, _, _, cx| {
                    this.model_picker_open = false;
                    cx.notify();
                }))
                .child(
                    div()
                        .occlude()
                        .w(px(360.))
                        .flex()
                        .flex_col()
                        .gap_1()
                        .p_3()
                        .rounded_xl()
                        .bg(theme.card)
                        .border_1()
                        .border_color(theme.border)
                        .child(
                            div()
                                .pb_1()
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(theme.text)
                                .child("Choose a model"),
                        )
                        .child(list),
                )
                .into_any_element(),
        )
    }

    /// Reset to a fresh, unsaved conversation (keeps the selected model).
    pub fn start_new(&mut self, cx: &mut Context<Self>) {
        self.messages.clear();
        self.chat_id = None;
        self.created_at = persistence::now_ms();
        self.streaming = false;
        self.cancel = None;
        cx.notify();
    }

    /// Start a fresh chat pinned to a specific model (e.g. a cloud model picked
    /// on the Cloud Models screen).
    pub fn use_model(&mut self, model: Model, cx: &mut Context<Self>) {
        self.model = Some(model);
        self.start_new(cx);
    }

    /// Load a previously saved chat for viewing/continuing.
    pub fn load_stored(&mut self, stored: StoredChat, cx: &mut Context<Self>) {
        self.messages = stored
            .messages
            .into_iter()
            .map(|m| ChatMsg {
                role: if m.role == "assistant" {
                    Role::Assistant
                } else {
                    Role::User
                },
                text: m.text,
                reasoning: m.reasoning,
                tps: m.tps,
                tokens: m.tokens,
                error: false,
            })
            .collect();
        self.chat_id = Some(stored.id);
        self.created_at = stored.created_at;
        self.model = None; // re-resolved on next send
        self.streaming = false;
        self.cancel = None;
        cx.notify();
    }

    fn save(&mut self) {
        if !self.messages.iter().any(|m| m.role == Role::User) {
            return;
        }
        let id = self
            .chat_id
            .clone()
            .unwrap_or_else(|| format!("chat-{}", self.created_at));
        self.chat_id = Some(id.clone());
        let title = self
            .messages
            .iter()
            .find(|m| m.role == Role::User)
            .map(|m| truncate(&m.text, 48))
            .unwrap_or_else(|| "New chat".to_string());
        let messages = self
            .messages
            .iter()
            .filter(|m| !m.error && !m.text.is_empty())
            .map(|m| StoredMessage {
                role: match m.role {
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                },
                text: m.text.clone(),
                reasoning: m.reasoning.clone(),
                tps: m.tps,
                tokens: m.tokens,
            })
            .collect();
        persistence::save_chat(&StoredChat {
            id,
            title,
            model_name: self.model.as_ref().map(|m| m.name()),
            created_at: self.created_at,
            updated_at: persistence::now_ms(),
            messages,
        });
    }

    /// Generation-settings overlay: sampling mode + (Stochastic) params + max tokens.
    fn gen_settings_overlay(&self, cx: &mut Context<Self>) -> Option<gpui::AnyElement> {
        if !self.gen_settings_open {
            return None;
        }
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let border = theme.border;
        let fg = theme.text;

        let mode_row = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(div().text_sm().text_color(fg).child("Sampling"))
            .child(
                SegmentedControl::new("sampling-mode", self.sampling_mode as usize)
                    .segment(
                        "Default",
                        cx.listener(|this, _, _, cx| {
                            this.sampling_mode = SamplingMode::Default;
                            cx.notify();
                        }),
                    )
                    .segment(
                        "Argmax",
                        cx.listener(|this, _, _, cx| {
                            this.sampling_mode = SamplingMode::Argmax;
                            cx.notify();
                        }),
                    )
                    .segment(
                        "Stochastic",
                        cx.listener(|this, _, _, cx| {
                            this.sampling_mode = SamplingMode::Stochastic;
                            cx.notify();
                        }),
                    ),
            );

        let mut card = div()
            .occlude()
            .w(px(320.))
            .flex()
            .flex_col()
            .gap_3()
            .p_4()
            .rounded_xl()
            .bg(theme.card)
            .border_1()
            .border_color(theme.border)
            .child(
                div()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child("Generation settings"),
            )
            .child(mode_row);

        if self.sampling_mode == SamplingMode::Stochastic {
            let off = |on: bool, v: String| if on { v } else { "off".to_string() };
            card = card
                .child(param_row(
                    "Temperature",
                    format!("{:.2}", self.temperature),
                    "temp-dec",
                    "temp-inc",
                    border,
                    fg,
                    hover,
                    cx.listener(|this, _, _, cx| {
                        this.temperature = round2((this.temperature - 0.05).max(0.0));
                        cx.notify();
                    }),
                    cx.listener(|this, _, _, cx| {
                        this.temperature = round2((this.temperature + 0.05).min(1.0));
                        cx.notify();
                    }),
                ))
                .child(param_row(
                    "Top K",
                    off(self.top_k > 0, self.top_k.to_string()),
                    "topk-dec",
                    "topk-inc",
                    border,
                    fg,
                    hover,
                    cx.listener(|this, _, _, cx| {
                        this.top_k = this.top_k.saturating_sub(8);
                        cx.notify();
                    }),
                    cx.listener(|this, _, _, cx| {
                        this.top_k = (this.top_k + 8).min(500);
                        cx.notify();
                    }),
                ))
                .child(param_row(
                    "Top P",
                    off(self.top_p > 0.0, format!("{:.2}", self.top_p)),
                    "topp-dec",
                    "topp-inc",
                    border,
                    fg,
                    hover,
                    cx.listener(|this, _, _, cx| {
                        this.top_p = round2((this.top_p - 0.05).max(0.0));
                        cx.notify();
                    }),
                    cx.listener(|this, _, _, cx| {
                        this.top_p = round2((this.top_p + 0.05).min(1.0));
                        cx.notify();
                    }),
                ))
                .child(param_row(
                    "Min P",
                    off(self.min_p > 0.0, format!("{:.2}", self.min_p)),
                    "minp-dec",
                    "minp-inc",
                    border,
                    fg,
                    hover,
                    cx.listener(|this, _, _, cx| {
                        this.min_p = round2((this.min_p - 0.01).max(0.0));
                        cx.notify();
                    }),
                    cx.listener(|this, _, _, cx| {
                        this.min_p = round2((this.min_p + 0.01).min(1.0));
                        cx.notify();
                    }),
                ));
        }

        let tokens_str = if self.max_tokens == 0 {
            "∞".to_string()
        } else {
            self.max_tokens.to_string()
        };
        card = card.child(param_row(
            "Max tokens",
            tokens_str,
            "tok-dec",
            "tok-inc",
            border,
            fg,
            hover,
            cx.listener(|this, _, _, cx| {
                this.max_tokens = this.max_tokens.saturating_sub(128);
                cx.notify();
            }),
            cx.listener(|this, _, _, cx| {
                this.max_tokens = (this.max_tokens + 128).min(8192);
                cx.notify();
            }),
        ));

        Some(
            div()
                .id("gen-settings-backdrop")
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .bg(gpui::black().opacity(0.5))
                .occlude()
                .on_click(cx.listener(|this, _, _, cx| {
                    this.gen_settings_open = false;
                    cx.notify();
                }))
                .child(card)
                .into_any_element(),
        )
    }

    fn resolved_model(&self, cx: &Context<Self>) -> Option<Model> {
        self.model.clone().or_else(|| {
            self.store
                .read(cx)
                .rows
                .iter()
                .find(|r| r.is_installed())
                .map(|r| r.model.clone())
        })
    }

    fn send(&mut self, text: String, cx: &mut Context<Self>) {
        if self.streaming {
            return;
        }
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.messages.push(ChatMsg {
            role: Role::User,
            text,
            reasoning: None,
            tps: None,
            tokens: None,
            error: false,
        });
        self.run_inference(cx);
    }

    /// Re-run the last turn, replacing the trailing assistant reply.
    fn regenerate(&mut self, cx: &mut Context<Self>) {
        if self.streaming {
            return;
        }
        if matches!(self.messages.last().map(|m| m.role), Some(Role::Assistant)) {
            self.messages.pop();
        }
        if !self.messages.iter().any(|m| m.role == Role::User) {
            return;
        }
        self.run_inference(cx);
    }

    /// Resolve the model, build the request from current history, and stream a
    /// reply into a fresh trailing assistant message.
    fn run_inference(&mut self, cx: &mut Context<Self>) {
        let Some(model) = self.resolved_model(cx) else {
            self.messages.push(ChatMsg {
                role: Role::Assistant,
                text: "No local model installed yet. Open Local Models to download one."
                    .to_string(),
                reasoning: None,
                tps: None,
                tokens: None,
                error: true,
            });
            cx.notify();
            return;
        };
        self.model = Some(model.clone());
        self.loaded_model = Some(model.name());

        // Build the conversation: optional global-instructions system message,
        // then everything so far, excluding errors.
        let mut history: Vec<ChatMessage> = Vec::new();
        let instructions = persistence::global_instructions();
        if !instructions.trim().is_empty() {
            history.push(ChatMessage::system().with_text(instructions));
        }
        history.extend(self.messages.iter().filter(|m| !m.error).map(|m| match m.role {
            Role::User => ChatMessage::user().with_text(m.text.clone()),
            Role::Assistant => ChatMessage::assistant().with_text(m.text.clone()),
        }));

        // Placeholder assistant message that fills in as tokens stream.
        self.messages.push(ChatMsg {
            role: Role::Assistant,
            text: String::new(),
            reasoning: None,
            tps: None,
            tokens: None,
            error: false,
        });
        self.streaming = true;

        let Some(engine) = engine::try_engine(cx) else {
            self.apply_stream(StreamMsg::Error("engine unavailable".to_string()), cx);
            return;
        };

        let sampling_method = match self.sampling_mode {
            SamplingMode::Default => None,
            SamplingMode::Argmax => Some(SamplingMethod::Greedy {}),
            SamplingMode::Stochastic => Some(SamplingMethod::Stochastic {
                temperature: Some(self.temperature as f64),
                top_k: (self.top_k > 0).then_some(self.top_k as i64),
                top_p: (self.top_p > 0.0).then_some(self.top_p as f64),
                min_p: (self.min_p > 0.0).then_some(self.min_p as f64),
                repetition_penalty: None,
                suffix_repetition_length: None,
            }),
        };
        let mut reply_config = ChatReplyConfig::default();
        if let Some(method) = sampling_method {
            reply_config = reply_config.with_sampling_method(method);
        }
        let reply_config = reply_config
            .with_token_limit((self.max_tokens > 0).then_some(self.max_tokens));

        let (tx, mut rx) = mpsc::unbounded::<StreamMsg>();

        // Producer: run uzu on the Tokio runtime, never touching view state.
        gpui_tokio::Tokio::spawn(cx, async move {
            let session = match engine.chat(model, ChatConfig::default()).await {
                Ok(session) => session,
                Err(err) => {
                    let _ = tx.unbounded_send(StreamMsg::Error(err.to_string()));
                    return;
                }
            };
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

        self.scroll.scroll_to_bottom();
        cx.notify();
    }

    fn apply_stream(&mut self, msg: StreamMsg, cx: &mut Context<Self>) {
        match msg {
            StreamMsg::Started(token) => self.cancel = Some(token),
            StreamMsg::Update {
                text,
                reasoning,
                tps,
                tokens,
            } => {
                if let Some(last) = self.messages.last_mut() {
                    if last.role == Role::Assistant {
                        last.text = text;
                        last.reasoning = reasoning;
                        last.tps = tps;
                        last.tokens = tokens;
                    }
                }
                self.scroll.scroll_to_bottom();
                cx.notify();
            }
            StreamMsg::Done => {
                self.streaming = false;
                self.cancel = None;
                // If the model produced no text, show a notice rather than an
                // empty bubble stuck on "…".
                if let Some(last) = self.messages.last_mut() {
                    if last.role == Role::Assistant && !last.error && last.text.is_empty() {
                        last.text = "(The model returned no text.)".to_string();
                        last.error = true;
                    }
                }
                self.save();
                cx.notify();
            }
            StreamMsg::Error(err) => {
                if let Some(last) = self.messages.last_mut() {
                    if last.role == Role::Assistant {
                        last.text = format!("Error: {err}");
                        last.error = true;
                    }
                }
                self.streaming = false;
                self.cancel = None;
                cx.notify();
            }
        }
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        self.streaming = false;
        self.cancel = None;
        cx.notify();
    }

    fn submit_from_button(&mut self, cx: &mut Context<Self>) {
        let text = self.input.read(cx).text();
        self.input.update(cx, |input, cx| input.clear(cx));
        self.send(text, cx);
    }
}

fn truncate(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max {
        trimmed.to_string()
    } else {
        let cut: String = trimmed.chars().take(max).collect();
        format!("{cut}…")
    }
}

/// Small square −/+ stepper button.
fn step_button(
    id: &'static str,
    symbol: &'static str,
    border: gpui::Hsla,
    fg: gpui::Hsla,
    hover: gpui::Hsla,
    handler: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .w(px(24.))
        .h(px(24.))
        .flex()
        .items_center()
        .justify_center()
        .rounded_md()
        .border_1()
        .border_color(border)
        .text_color(fg)
        .cursor(gpui::CursorStyle::PointingHand)
        .hover(move |s| s.bg(hover))
        .on_click(handler)
        .child(symbol)
}

/// Round a 0–1 sampling value to 2 decimals (avoids float drift across steps).
fn round2(v: f32) -> f32 {
    (v * 100.0).round() / 100.0
}

/// A labeled −/+ stepper row used in the generation-settings panel.
#[allow(clippy::too_many_arguments)]
fn param_row(
    label: &str,
    value: String,
    dec_id: &'static str,
    inc_id: &'static str,
    border: gpui::Hsla,
    fg: gpui::Hsla,
    hover: gpui::Hsla,
    on_dec: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
    on_inc: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .child(div().text_sm().text_color(fg).child(label.to_string()))
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(step_button(dec_id, "−", border, fg, hover, on_dec))
                .child(div().w(px(44.)).text_color(fg).child(value))
                .child(step_button(inc_id, "+", border, fg, hover, on_inc)),
        )
}

impl Render for ChatView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let streaming = self.streaming;
        let show_reasoning = settings_state::current(cx).reasoning;
        let model_name = self
            .resolved_model(cx)
            .map(|m| m.name())
            .unwrap_or_else(|| "No model".to_string());

        // Message column.
        let mut column = div().flex().flex_col().gap_5().py_6();
        if self.messages.is_empty() {
            column = column.child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .justify_center()
                    .gap_3()
                    .pt_16()
                    .child(IconEl::new(Icon::Logo, theme.text).size(40.))
                    .child(
                        div()
                            .text_xl()
                            .font_weight(FontWeight::MEDIUM)
                            .child("Ask anything"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text_muted)
                            .child(format!("Chatting with {model_name}")),
                    ),
            );
        } else {
            let last_idx = self.messages.len().saturating_sub(1);
            for (idx, msg) in self.messages.iter().enumerate() {
                let is_last = idx == last_idx;
                column = column.child(match msg.role {
                    Role::User => div()
                        .flex()
                        .justify_end()
                        .child(
                            div()
                                .max_w(px(560.))
                                .px_3()
                                .py_2()
                                .rounded_lg()
                                .bg(theme.bg_hover)
                                .text_color(theme.text)
                                .child(msg.text.clone()),
                        )
                        .into_any_element(),
                    Role::Assistant => {
                        let mut block = div().flex().flex_col().gap_2().w_full();

                        // Reasoning ("thinking") panel (hidden when the setting is off).
                        if show_reasoning {
                            if let Some(reasoning) = &msg.reasoning {
                                if !reasoning.trim().is_empty() {
                                block = block.child(
                                    div()
                                        .flex()
                                        .flex_col()
                                        .gap_1()
                                        .p_3()
                                        .rounded_lg()
                                        .bg(theme.bg_sub)
                                        .border_l_2()
                                        .border_color(theme.border)
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_1()
                                                .text_xs()
                                                .text_color(theme.text_muted)
                                                .child(IconEl::new(Icon::Thinking, theme.text_muted).size(13.))
                                                .child("Thinking"),
                                        )
                                        .child(
                                            div()
                                                .text_sm()
                                                .text_color(theme.text_muted)
                                                .child(reasoning.clone()),
                                        ),
                                );
                                }
                            }
                        }

                        // Body (markdown for assistant prose + code blocks).
                        let body = if msg.error {
                            div()
                                .text_color(theme.error)
                                .child(msg.text.clone())
                                .into_any_element()
                        } else if msg.text.is_empty() && streaming {
                            Loader::new().label("Generating…").into_any_element()
                        } else {
                            div()
                                .text_color(theme.text)
                                .child(crate::components::markdown::markdown(&msg.text, &theme))
                                .into_any_element()
                        };
                        block = block.child(body);

                        // Perf stats footer.
                        if !streaming && (msg.tokens.is_some() || msg.tps.is_some()) {
                            let tps = msg.tps.map(|t| format!("{t:.0} tok/s")).unwrap_or_default();
                            let toks = msg
                                .tokens
                                .map(|t| format!("{t} tokens"))
                                .unwrap_or_default();
                            block = block.child(
                                div()
                                    .flex()
                                    .gap_2()
                                    .text_xs()
                                    .text_color(theme.text_muted)
                                    .child(toks)
                                    .child(tps),
                            );
                        }

                        // Actions (copy always; regenerate on the last message).
                        if !streaming && !msg.error && !msg.text.is_empty() {
                            let copy_text = msg.text.clone();
                            let mut actions = div().flex().items_center().gap_3().pt_1();
                            actions = actions.child(
                                div()
                                    .id(gpui::SharedString::from(format!("copy-{idx}")))
                                    .flex()
                                    .items_center()
                                    .gap_1()
                                    .text_xs()
                                    .text_color(theme.text_muted)
                                    .cursor(gpui::CursorStyle::PointingHand)
                                    .on_click(cx.listener(move |_this, _, _, cx| {
                                        cx.write_to_clipboard(gpui::ClipboardItem::new_string(
                                            copy_text.clone(),
                                        ));
                                    }))
                                    .child(IconEl::new(Icon::Copy, theme.text_muted).size(12.))
                                    .child("Copy"),
                            );
                            if is_last {
                                actions = actions.child(
                                    div()
                                        .id("regenerate")
                                        .flex()
                                        .items_center()
                                        .gap_1()
                                        .text_xs()
                                        .text_color(theme.text_muted)
                                        .cursor(gpui::CursorStyle::PointingHand)
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.regenerate(cx)),
                                        )
                                        .child("↻ Regenerate"),
                                );
                            }
                            block = block.child(actions);
                        }

                        block.into_any_element()
                    }
                });
            }
        }

        // Composer send/stop button.
        let send_button = if streaming {
            IconButton::new("chat-stop", Icon::Stop)
                .color(theme.text)
                .on_click(cx.listener(|this, _, _, cx| this.stop(cx)))
        } else {
            IconButton::new("chat-send", Icon::Send)
                .color(theme.accent)
                .on_click(cx.listener(|this, _, _, cx| this.submit_from_button(cx)))
        };

        div()
            .size_full()
            .relative()
            .flex()
            .flex_col()
            .items_center()
            // Scrollable message area.
            .child(
                div()
                    .id("chat-scroll")
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .flex()
                    .flex_col()
                    .items_center()
                    .child(
                        div()
                            .w_full()
                            .max_w(px(CONTENT_MAX_WIDTH))
                            .px_6()
                            .child(column),
                    ),
            )
            // Composer.
            .child(
                div()
                    .w_full()
                    .flex()
                    .flex_col()
                    .items_center()
                    .pb_4()
                    .child(
                        div()
                            .w_full()
                            .max_w(px(CONTENT_MAX_WIDTH))
                            .px_6()
                            .flex()
                            .flex_col()
                            .gap_1()
                            // Model selector + generation-settings gear.
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        div()
                                            .id("model-trigger")
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .cursor(gpui::CursorStyle::PointingHand)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.model_picker_open = !this.model_picker_open;
                                                cx.notify();
                                            }))
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(theme.text_muted)
                                                    .child(format!("Model: {model_name}")),
                                            )
                                            .child(
                                                IconEl::new(Icon::ChevronDown, theme.text_muted)
                                                    .size(13.),
                                            ),
                                    )
                                    .child(
                                        IconButton::new("gen-settings", Icon::Settings)
                                            .color(theme.text_muted)
                                            .icon_size(14.)
                                            .hit_size(22.)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.gen_settings_open = !this.gen_settings_open;
                                                cx.notify();
                                            })),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .w_full()
                                    .px_3()
                                    .py_2()
                                    .rounded_xl()
                                    .border_1()
                                    .border_color(theme.border)
                                    .bg(theme.card)
                                    .child(div().flex_1().child(self.input.clone()))
                                    .child(send_button),
                            ),
                    ),
            )
            .children(self.model_picker_overlay(cx))
            .children(self.gen_settings_overlay(cx))
    }
}
