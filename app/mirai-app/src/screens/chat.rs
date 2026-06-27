//! Chat screen: streaming local inference with a reasoning panel, perf stats,
//! and a composer. Streaming follows the Tokio→channel→foreground pattern: the
//! uzu reply stream runs on the Tokio runtime and pushes cumulative updates back
//! to the UI entity.

use futures::{StreamExt, channel::mpsc};
use gpui::{
    Anchor, App, Context, CursorStyle, Entity, EventEmitter, FontWeight, IntoElement,
    Render, ScrollHandle, SharedString, Window, anchored, deferred, div, prelude::*, px,
};
use uzu::{
    session::chat::{ChatSession, ChatSessionStreamChunk},
    types::{
        basic::{CancelToken, SamplingMethod},
        model::Model,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

use crate::{
    components::{
        Icon, IconButton, IconEl, InputEvent, Loader, SegmentedControl, Slider, TextInput, Toggle,
        VendorIcon,
    },
    engine,
    models_store::ModelsStore,
    persistence::{self, StoredChat, StoredMessage},
    settings_state,
    theme::{ActiveTheme, FONT_MONO, Theme, layout::CONTENT_MAX_WIDTH},
    title_gen,
    toast::{self, ToastKind},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
}

/// Sampling mode (mirrors ui-kit). `Default` uses the model's own config.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SamplingMode {
    Default,
    Argmax,
    Stochastic,
}

/// One generated version of an assistant reply (regenerate keeps the prior
/// ones so they can be paged through). User messages have exactly one.
#[derive(Clone, Default)]
struct Version {
    text: String,
    reasoning: Option<String>,
    tps: Option<f32>,
    tokens: Option<u32>,
    error: bool,
    /// Display name of the model that produced this version.
    model_name: Option<String>,
}

struct ChatMsg {
    role: Role,
    versions: Vec<Version>,
    current: usize,
    /// Whether the reasoning panel is collapsed. Starts expanded (false) while
    /// streaming; auto-collapses once the reply body text starts arriving.
    reasoning_collapsed: bool,
}

impl ChatMsg {
    fn user(text: String) -> Self {
        Self {
            role: Role::User,
            versions: vec![Version { text, ..Default::default() }],
            current: 0,
            reasoning_collapsed: false,
        }
    }

    fn assistant(version: Version) -> Self {
        Self {
            role: Role::Assistant,
            versions: vec![version],
            current: 0,
            reasoning_collapsed: false,
        }
    }

    fn cur(&self) -> &Version {
        &self.versions[self.current]
    }

    fn cur_mut(&mut self) -> &mut Version {
        &mut self.versions[self.current]
    }
}

/// The conversation to send for a reply: prior turns only — the trailing
/// assistant placeholder being filled is dropped, as are any errored turns —
/// each as its current version's `(role, text)`.
fn conversation_for_request(messages: &[ChatMsg]) -> Vec<(Role, String)> {
    let upto = messages.len().saturating_sub(1);
    messages[..upto]
        .iter()
        .filter(|m| !m.cur().error)
        .map(|m| (m.role, m.cur().text.clone()))
        .collect()
}

/// Map the UI sampling mode + params to a uzu `SamplingMethod`. `Default`
/// leaves it to the model's own config (`None`); a param of 0 means "off".
fn sampling_method(
    mode: SamplingMode,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
) -> Option<SamplingMethod> {
    match mode {
        SamplingMode::Default => None,
        SamplingMode::Argmax => Some(SamplingMethod::Greedy {}),
        SamplingMode::Stochastic => Some(SamplingMethod::Stochastic {
            temperature: Some(temperature as f64),
            top_k: (top_k > 0).then_some(top_k as i64),
            top_p: (top_p > 0.0).then_some(top_p as f64),
            min_p: (min_p > 0.0).then_some(min_p as f64),
            repetition_penalty: None,
            suffix_repetition_length: None,
        }),
    }
}

/// Messages bridged from the Tokio reply stream to the UI.
enum StreamMsg {
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

pub enum ChatEvent {
    Updated,
    OpenLocalModels,
}

pub struct ChatView {
    store: Entity<ModelsStore>,
    /// Cloud chat models, shown alongside local ones in the model picker.
    cloud_store: Entity<ModelsStore>,
    input: Entity<TextInput>,
    messages: Vec<ChatMsg>,
    model: Option<Model>,
    streaming: bool,
    /// True between "send" and the first `Started` token — model is loading.
    waiting_for_model: bool,
    cancel: Option<CancelToken>,
    chat_id: Option<String>,
    created_at: u64,
    scroll: ScrollHandle,
    /// Auto-scrolls the streaming reasoning panel to its latest line.
    reasoning_scroll: ScrollHandle,
    /// Frames remaining to keep re-pinning the scroll to the bottom. A single
    /// `scroll_to_bottom()` lands short because wrapped-text height is only
    /// final after the second layout pass; re-asserting for a few frames lets
    /// the offset converge to the true bottom as `content_size` settles.
    pin_bottom_frames: u8,
    model_picker_open: bool,
    /// Message index whose per-message model menu is open.
    msg_model_picker_open: Option<usize>,
    /// Message index whose performance popover is open (`None` = closed).
    perf_open_msg: Option<usize>,
    file_upload_open: bool,
    /// Files attached to the next outgoing message: (display_name, extension, content).
    /// Appended as fenced code blocks when the message is sent, then cleared.
    attached_files: Vec<(String, String, String)>,
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
    chat_title: String,
    title_pending: bool,
    session: Option<ChatSession>,
    session_model_id: Option<String>,
}

impl EventEmitter<ChatEvent> for ChatView {}

/// (id, model, display_name, vendor_name, icon_url)
type ModelEntry = (String, Model, String, String, Option<String>);

impl ChatView {
    pub fn new(
        store: Entity<ModelsStore>,
        cloud_store: Entity<ModelsStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(|cx| TextInput::new(cx, "Add message…").multiline(true, 1, 8));
        cx.subscribe(&input, |this, _input, event, cx| match event {
            InputEvent::Submit(text) => this.send(text.clone(), cx),
            InputEvent::Changed(_) => {}
        })
        .detach();
        cx.observe(&store, |_, _, cx| cx.notify()).detach();
        cx.observe(&cloud_store, |_, _, cx| cx.notify()).detach();
        settings_state::observe(cx, |_, cx| cx.notify()).detach();
        Self {
            store,
            cloud_store,
            input,
            messages: Vec::new(),
            model: None,
            streaming: false,
            waiting_for_model: false,
            cancel: None,
            chat_id: None,
            created_at: persistence::now_ms(),
            scroll: ScrollHandle::new(),
            reasoning_scroll: ScrollHandle::new(),
            pin_bottom_frames: 0,
            model_picker_open: false,
            msg_model_picker_open: None,
            perf_open_msg: None,
            file_upload_open: false,
            attached_files: Vec::new(),
            loaded_model: None,
            gen_settings_open: false,
            sampling_mode: SamplingMode::Default,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            max_tokens: 0,
            chat_title: String::new(),
            title_pending: false,
            session: None,
            session_model_id: None,
        }
    }

    fn clear_session(&mut self) {
        self.session = None;
        self.session_model_id = None;
    }

    fn cached_session(&self, model_id: &str) -> Option<ChatSession> {
        self.session
            .as_ref()
            .filter(|_| self.session_model_id.as_deref() == Some(model_id))
            .cloned()
    }

    fn store_session(&mut self, session: ChatSession, model_id: &str) {
        self.session = Some(session);
        self.session_model_id = Some(model_id.to_string());
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
        self.clear_session();
        cx.notify();
    }

    fn close_popovers(&mut self) {
        self.model_picker_open = false;
        self.msg_model_picker_open = None;
        self.perf_open_msg = None;
        self.file_upload_open = false;
    }

    /// Open a native file-picker, read the selected file(s) as UTF-8 text and
    /// add them to `attached_files`. Max 5 files, 256 KB each (matching Electron).
    fn pick_file(&mut self, cx: &mut Context<Self>) {
        const MAX_SIZE: u64 = 256 * 1024; // 256 KB
        const MAX_FILES: usize = 5;
        const SUPPORTED: &[&str] = &[
            "txt", "md", "markdown", "json", "csv", "tsv", "yaml", "yml",
            "py", "js", "ts", "tsx", "jsx", "rs", "html", "css", "xml",
        ];

        if self.attached_files.len() >= MAX_FILES {
            toast::push(cx, "Maximum 5 files per message", ToastKind::Info);
            return;
        }

        let rx = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Attach text file".into()),
        });

        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return; };
            for path in paths.iter().take(MAX_FILES) {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("txt")
                    .to_lowercase();
                if !SUPPORTED.contains(&ext.as_str()) {
                    let _ = this.update(cx, |_, cx| {
                        toast::push(cx, format!("Unsupported file type: .{ext}"), ToastKind::Info);
                    });
                    continue;
                }
                let meta = std::fs::metadata(path);
                if meta.map(|m| m.len()).unwrap_or(0) > MAX_SIZE {
                    let _ = this.update(cx, |_, cx| {
                        toast::push(cx, "File too large (max 256 KB)", ToastKind::Info);
                    });
                    continue;
                }
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        let name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("file")
                            .to_string();
                        let _ = this.update(cx, |this, cx| {
                            if this.attached_files.len() < MAX_FILES {
                                this.attached_files.push((name, ext, content));
                                cx.notify();
                            }
                        });
                    }
                    Err(_) => {
                        let _ = this.update(cx, |_, cx| {
                            toast::push(cx, "Could not read file", ToastKind::Info);
                        });
                    }
                }
            }
        })
        .detach();
    }

    fn collect_model_entries(&self, cx: &App) -> (Vec<ModelEntry>, Vec<ModelEntry>) {
        let local = self
            .store
            .read(cx)
            .rows
            .iter()
            .filter(|r| r.is_installed())
            .map(|r| {
                (
                    r.id().to_string(),
                    r.model.clone(),
                    r.name(),
                    r.vendor().unwrap_or_default(),
                    r.icon_url(true),
                )
            })
            .collect();
        let cloud = self
            .cloud_store
            .read(cx)
            .rows
            .iter()
            .map(|r| {
                (
                    r.id().to_string(),
                    r.model.clone(),
                    r.name(),
                    r.vendor().unwrap_or_default(),
                    r.icon_url(true),
                )
            })
            .collect();
        (local, cloud)
    }

    /// One row in a model picker (`model-selector.tsx` ListboxOption layout).
    /// Wraps `panel` in `deferred(anchored(corner).snap_to_window())` with a
    /// mouse-down-out close handler — shared by all three popovers.
    fn anchored_popover(
        panel: impl gpui::IntoElement + 'static,
        corner: Anchor,
        close: impl Fn(&mut Self, &gpui::MouseDownEvent, &mut gpui::Window, &mut gpui::Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        deferred(
            anchored()
                .anchor(corner)
                .snap_to_window_with_margin(px(8.))
                .child(div().on_mouse_down_out(cx.listener(close)).child(panel)),
        )
        .priority(1)
        .into_any_element()
    }

    fn picker_row(
        &self,
        cx: &mut Context<Self>,
        id: String,
        model: Model,
        name: String,
        vendor: String,
        icon_url: Option<String>,
        is_local: bool,
        hover: gpui::Hsla,
        for_message: Option<usize>,
    ) -> gpui::AnyElement {
        let theme = cx.theme().clone();
        // Plain muted label — matches Electron's model-selector styling.
        let badge = div()
            .flex_none()
            .text_size(crate::tokens::font::SMALL)
            .text_color(theme.text_muted)
            .child(if is_local { "Local" } else { "Cloud" });

        div()
            .id(gpui::SharedString::from(format!("pick-{id}")))
            .flex()
            .items_center()
            .gap_3()
            .w_full()
            .px_4()
            .h(px(48.))
            .cursor(gpui::CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .child(VendorIcon::new(vendor).size(crate::tokens::icon::XXL).icon_url(icon_url))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .text_sm()
                    .text_color(theme.text)
                    .child(name),
            )
            .child(badge)
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(msg_idx) = for_message {
                    this.regenerate_at_with_model(msg_idx, Some(model.clone()), cx);
                } else {
                    this.model = Some(model.clone());
                    this.clear_session();
                    this.close_popovers();
                    cx.notify();
                }
            }))
            .into_any_element()
    }

    /// Shared model list + footer (`model-selector.tsx` ListboxOptions body).
    fn model_picker_panel(
        &self,
        cx: &mut Context<Self>,
        for_message: Option<usize>,
    ) -> gpui::AnyElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let (local, cloud) = self.collect_model_entries(cx);

        let mut list = div()
            .id("model-picker-list")
            .flex()
            .flex_col()
            .gap_2()
            .max_h(px(360.))
            .overflow_y_scroll();
        if local.is_empty() && cloud.is_empty() {
            list = list.child(
                div()
                    .px(px(14.))
                    .py_2()
                    .text_sm()
                    .text_color(theme.text_muted)
                    .child("No models available"),
            );
        } else {
            for (id, model, name, vendor, icon_url) in local {
                list = list.child(self.picker_row(
                    cx, id, model, name, vendor, icon_url, true, hover, for_message,
                ));
            }
            for (id, model, name, vendor, icon_url) in cloud {
                list = list.child(self.picker_row(
                    cx, id, model, name, vendor, icon_url, false, hover, for_message,
                ));
            }
        }

        div()
            .occlude()
            .w(px(520.))
            .flex()
            .flex_col()
            .rounded_xl()
            .bg(theme.card)
            .border_1()
            .border_color(theme.border)
            .child(
                div()
                    .py_2()
                    .child(list),
            )
            .child(div().h_px().bg(theme.border))
            .child(
                div()
                    .id("model-picker-more")
                    .flex()
                    .items_center()
                    .justify_between()
                    .w_full()
                    .px_4()
                    .h(px(48.))
                    .cursor(gpui::CursorStyle::PointingHand)
                    .hover(move |s| s.bg(hover))
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.close_popovers();
                        cx.emit(ChatEvent::OpenLocalModels);
                        cx.notify();
                    }))
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text)
                            .child("More local models"),
                    )
                    .child(IconEl::new(Icon::ChevronRight, theme.text_muted).size(crate::tokens::icon::MD)),
            )
            .into_any_element()
    }

    fn performance_panel(
        &self,
        msg_idx: usize,
        cur: &Version,
        cx: &mut Context<Self>,
    ) -> Option<gpui::AnyElement> {
        if self.perf_open_msg != Some(msg_idx) {
            return None;
        }
        let theme = cx.theme().clone();
        let tps = cur
            .tps
            .filter(|t| t.is_finite() && *t > 0.0)
            .map(|t| format!("{}", t.round() as i64))
            .unwrap_or_else(|| "—".into());
        let stat = |value: String, label: &'static str| {
            div()
                .flex()
                .flex_col()
                .items_center()
                .gap_1()
                .child(
                    div()
                        .text_size(crate::tokens::font::HEADING)
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(theme.text)
                        .child(value),
                )
                .child(
                    div()
                        .text_size(crate::tokens::font::CAPTION)
                        .text_color(theme.text_muted)
                        .child(label),
                )
        };
        let sep = || div().w_px().h(px(40.)).bg(theme.border).self_center();
        Some(
            div()
                .occlude()
                .py_4()
                .px_6()
                .rounded_xl()
                .bg(theme.card)
                .border_1()
                .border_color(theme.border)
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_6()
                        .child(stat("—".into(), "T to 1st token"))
                        .child(sep())
                        .child(stat(tps, "№ of t/s"))
                        .child(sep())
                        .child(stat("—".into(), "Total time")),
                )
                .into_any_element(),
        )
    }

    fn file_upload_panel(&self, cx: &mut Context<Self>) -> Option<gpui::AnyElement> {
        if !self.file_upload_open {
            return None;
        }
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        Some(
            div()
                .occlude()
                .rounded_md()
                .bg(theme.card)
                .border_1()
                .border_color(theme.border)
                .p(px(6.))
                .child(
                    div()
                        .id("file-upload-option")
                        .flex()
                        .items_center()
                        .gap_2()
                        .w(px(300.))
                        .px(px(14.))
                        .py_2()
                        .rounded_md()
                        .cursor(gpui::CursorStyle::PointingHand)
                        .hover(move |s| s.bg(hover))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.file_upload_open = false;
                            this.pick_file(cx);
                        }))
                        .child(IconEl::new(Icon::Rename, theme.text_muted).size(crate::tokens::icon::MD))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(theme.text)
                                        .child("Upload a file"),
                                )
                                .child(
                                    div()
                                        .text_size(crate::tokens::font::CAPTION)
                                        .text_color(theme.text_muted)
                                        .child("TXT · MD · JSON · CSV · YAML · PY · JS · TS · RS"),
                                ),
                        ),
                )
                .into_any_element(),
        )
    }

    /// Open the model picker (used by the trigger and visual tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn open_model_picker(&mut self, cx: &mut Context<Self>) {
        self.close_popovers();
        self.model_picker_open = true;
        cx.notify();
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn open_perf_panel(&mut self, msg_idx: usize, cx: &mut Context<Self>) {
        self.close_popovers();
        self.perf_open_msg = Some(msg_idx);
        cx.notify();
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn open_gen_settings(&mut self, cx: &mut Context<Self>) {
        self.gen_settings_open = true;
        cx.notify();
    }

    #[cfg(test)]
    pub fn set_stochastic(&mut self, cx: &mut Context<Self>) {
        self.sampling_mode = SamplingMode::Stochastic;
        cx.notify();
    }

    #[cfg(test)]
    pub fn expand_reasoning(&mut self, msg_idx: usize, cx: &mut Context<Self>) {
        if let Some(m) = self.messages.get_mut(msg_idx) {
            m.reasoning_collapsed = false;
            cx.notify();
        }
    }

    /// Reset to a fresh, unsaved conversation (keeps the selected model).
    pub fn start_new(&mut self, cx: &mut Context<Self>) {
        self.messages.clear();
        self.chat_id = None;
        self.created_at = persistence::now_ms();
        self.chat_title.clear();
        self.title_pending = false;
        self.streaming = false;
        self.cancel = None;
        self.attached_files.clear();
        self.clear_session();
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
                versions: vec![Version {
                    text: m.text,
                    reasoning: m.reasoning,
                    tps: m.tps,
                    tokens: m.tokens,
                    error: false,
                    ..Default::default()
                }],
                current: 0,
                // Loaded chats collapse reasoning by default — they're already done.
                reasoning_collapsed: true,
            })
            .collect();
        self.chat_id = Some(stored.id);
        self.created_at = stored.created_at;
        self.chat_title = stored.title;
        self.title_pending = false;
        self.model = None;
        self.streaming = false;
        self.cancel = None;
        self.clear_session();
        // Opening a saved chat lands on its most recent message.
        self.pin_to_bottom();
        cx.notify();
    }

    /// Keep the message list pinned to the bottom for the next few frames so
    /// the scroll offset converges to the true bottom once wrapped-text height
    /// has settled (a single `scroll_to_bottom` lands short — see the field).
    fn pin_to_bottom(&mut self) {
        self.scroll.scroll_to_bottom();
        self.pin_bottom_frames = 8;
    }

    /// True when the user is at (or near) the bottom — used to avoid fighting
    /// manual scroll during streaming.
    fn should_auto_scroll(&self) -> bool {
        let offset = self.scroll.offset();
        let max = self.scroll.max_offset();
        if max.y <= px(0.) {
            return true;
        }
        offset.y <= -(max.y - px(32.))
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
        let title = if title_gen::is_placeholder(&self.chat_title) {
            self.messages
                .iter()
                .find(|m| m.role == Role::User)
                .map(|m| truncate(&m.cur().text, 48))
                .unwrap_or_else(|| "New chat".to_string())
        } else {
            self.chat_title.clone()
        };
        let messages = self
            .messages
            .iter()
            .filter(|m| !m.cur().error && !m.cur().text.is_empty())
            .map(|m| StoredMessage {
                role: match m.role {
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                },
                text: m.cur().text.clone(),
                reasoning: m.cur().reasoning.clone(),
                tps: m.cur().tps,
                tokens: m.cur().tokens,
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

        // Title bar + current-model row + reasoning toggle (Electron drawer).
        let resolved = self.resolved_model(cx);
        let model_name =
            resolved.as_ref().map(|m| m.name()).unwrap_or_else(|| "No model".to_string());
        let vendor = resolved
            .as_ref()
            .and_then(|m| m.family.as_ref().map(|f| f.vendor.name()))
            .unwrap_or_default();
        let icon_url = resolved.as_ref().and_then(|m| {
            let icons = &m.family.as_ref()?.vendor.metadata.icons;
            icons
                .iter()
                .find(|i| i.theme == uzu::types::basic::ImageTheme::Dark)
                .or_else(|| icons.first())
                .map(|i| i.url.clone())
        });
        let reasoning_on = settings_state::current(cx).reasoning;

        let header = div()
            .flex()
            .items_center()
            .justify_between()
            .child(
                div()
                    .text_lg()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(fg)
                    .child("Edit parameters"),
            )
            .child(
                IconButton::new("gen-close", Icon::Close)
                    .color(theme.text_muted)
                    .icon_size(16.)
                    .hit_size(28.)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.gen_settings_open = false;
                        cx.notify();
                    })),
            );

        let model_row = div()
            .flex()
            .items_center()
            .gap_2()
            .child(VendorIcon::new(vendor).size(crate::tokens::icon::XL).icon_url(icon_url))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_sm()
                    .text_color(fg)
                    .child(model_name),
            )
            .child(IconEl::new(Icon::ChevronDown, theme.text_muted).size(crate::tokens::icon::MD));

        let reasoning_row = div()
            .flex()
            .items_center()
            .justify_between()
            .child(div().text_sm().text_color(fg).child("Reasoning"))
            .child(Toggle::new("gen-reasoning", reasoning_on).on_click(|_, _, cx| {
                let mut s = settings_state::current(cx);
                s.reasoning = !s.reasoning;
                settings_state::set(cx, s);
            }));

        let mut card = div()
            .occlude()
            .absolute()
            .top_0()
            .right_0()
            .bottom_0()
            .w(px(380.))
            .flex()
            .flex_col()
            .gap_4()
            .p_5()
            .bg(theme.bg)
            .border_l_1()
            .border_color(theme.border)
            .child(header)
            .child(model_row)
            .child(mode_row);

        if self.sampling_mode == SamplingMode::Stochastic {
            let view = cx.entity();
            // Temperature (0–2).
            let v = view.clone();
            card = card.child(slider_param(
                "Temperature",
                None,
                format!("{}", round2(self.temperature)),
                (self.temperature / 2.0).clamp(0., 1.),
                "temp-slider",
                &theme,
                move |frac, _, cx| {
                    v.update(cx, |this, cx| {
                        this.temperature = round2(frac * 2.0);
                        cx.notify();
                    });
                },
            ));
            // Top K (0–200).
            let v = view.clone();
            card = card.child(slider_param(
                "Top K",
                None,
                self.top_k.to_string(),
                (self.top_k as f32 / 200.0).clamp(0., 1.),
                "topk-slider",
                &theme,
                move |frac, _, cx| {
                    v.update(cx, |this, cx| {
                        this.top_k = (frac * 200.0).round() as u32;
                        cx.notify();
                    });
                },
            ));
            // Top P (0–1) with on/off checkbox.
            let v = view.clone();
            let topp_box = param_checkbox("topp-cb", self.top_p > 0.0, &theme, {
                let v = view.clone();
                move |_, _, cx| {
                    v.update(cx, |this, cx| {
                        this.top_p = if this.top_p > 0.0 { 0.0 } else { 0.95 };
                        cx.notify();
                    });
                }
            });
            card = card.child(slider_param(
                "Top P",
                Some(topp_box.into_any_element()),
                format!("{}", round2(self.top_p)),
                self.top_p.clamp(0., 1.),
                "topp-slider",
                &theme,
                move |frac, _, cx| {
                    v.update(cx, |this, cx| {
                        this.top_p = round2(frac);
                        cx.notify();
                    });
                },
            ));
            // Min P (0–1) with on/off checkbox.
            let v = view.clone();
            let minp_box = param_checkbox("minp-cb", self.min_p > 0.0, &theme, {
                let v = view.clone();
                move |_, _, cx| {
                    v.update(cx, |this, cx| {
                        this.min_p = if this.min_p > 0.0 { 0.0 } else { 0.05 };
                        cx.notify();
                    });
                }
            });
            card = card.child(slider_param(
                "Min P",
                Some(minp_box.into_any_element()),
                format!("{}", round2(self.min_p)),
                self.min_p.clamp(0., 1.),
                "minp-slider",
                &theme,
                move |frac, _, cx| {
                    v.update(cx, |this, cx| {
                        this.min_p = round2(frac);
                        cx.notify();
                    });
                },
            ));
        }

        // Divider + Reasoning toggle (always shown, like Electron).
        card = card.child(div().h_px().bg(border)).child(reasoning_row);

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
                .bg(gpui::black().opacity(0.4))
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
        if text.is_empty() && self.attached_files.is_empty() {
            return;
        }
        // Append attached files as fenced code blocks (Electron parity).
        let full_text = if self.attached_files.is_empty() {
            text
        } else {
            let mut s = text;
            for (name, ext, content) in self.attached_files.drain(..) {
                s.push_str(&format!("\n\n```{ext}\n# {name}\n{content}\n```"));
            }
            s
        };
        let first_user = !self.messages.iter().any(|m| m.role == Role::User);
        self.messages.push(ChatMsg::user(full_text));
        if first_user {
            self.title_pending = true;
        }
        self.run_inference(cx);
    }

    /// Start a fresh assistant reply for the latest turn.
    fn run_inference(&mut self, cx: &mut Context<Self>) {
        let model_name = self.model.as_ref().map(|m| m.name());
        self.messages.push(ChatMsg::assistant(Version { model_name, ..Default::default() }));
        self.streaming = true;
        self.waiting_for_model = true;
        self.spawn_reply(cx);
    }

    /// Re-run an assistant turn (optionally switching model) as a new version.
    fn regenerate_at_with_model(
        &mut self,
        msg_idx: usize,
        model: Option<Model>,
        cx: &mut Context<Self>,
    ) {
        if self.streaming {
            return;
        }
        let Some(msg) = self.messages.get(msg_idx) else {
            return;
        };
        if msg.role != Role::Assistant {
            return;
        }
        if let Some(model) = model {
            self.model = Some(model);
            self.clear_session();
        }
        self.messages.truncate(msg_idx + 1);
        let model_name = self.model.as_ref().map(|m| m.name());
        if let Some(last) = self.messages.last_mut() {
            last.versions.push(Version { model_name, ..Default::default() });
            last.current = last.versions.len() - 1;
        }
        self.close_popovers();
        self.streaming = true;
        self.spawn_reply(cx);
    }

    /// Resolve the model, build the request from the conversation (excluding the
    /// trailing assistant placeholder), and stream into its current version.
    fn spawn_reply(&mut self, cx: &mut Context<Self>) {
        let Some(model) = self.resolved_model(cx) else {
            if let Some(last) = self.messages.last_mut() {
                let v = last.cur_mut();
                v.text = "No local model installed yet. Open Local Models to download one."
                    .to_string();
                v.error = true;
            }
            self.streaming = false;
            cx.notify();
            return;
        };
        self.model = Some(model.clone());
        self.loaded_model = Some(model.name());

        // Global instructions + prior messages, excluding the trailing assistant
        // placeholder being filled and any errored turns.
        let mut history: Vec<ChatMessage> = Vec::new();
        let instructions = persistence::global_instructions();
        if !instructions.trim().is_empty() {
            history.push(ChatMessage::system().with_text(instructions));
        }
        history.extend(conversation_for_request(&self.messages).into_iter().map(
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
            self.sampling_mode,
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
        );
        let mut reply_config = ChatReplyConfig::default();
        if let Some(method) = method {
            reply_config = reply_config.with_sampling_method(method);
        }
        let reply_config = reply_config
            .with_token_limit((self.max_tokens > 0).then_some(self.max_tokens));

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
                self.cancel = Some(token);
                self.waiting_for_model = false;
            }
            StreamMsg::Session(session) => {
                if let Some(id) = self.model.as_ref().map(|m| m.identifier.clone()) {
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
                if let Some(last) = self.messages.last_mut() {
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
                self.streaming = false;
                self.waiting_for_model = false;
                self.cancel = None;
                // If the model produced no text, show a notice rather than an
                // empty bubble stuck on "…".
                if let Some(last) = self.messages.last_mut() {
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
                if let Some(last) = self.messages.last_mut() {
                    if last.role == Role::Assistant {
                        let v = last.cur_mut();
                        v.text = format!("Error: {err}");
                        v.error = true;
                    }
                }
                self.streaming = false;
                self.waiting_for_model = false;
                self.cancel = None;
                crate::toast::push(cx, "Inference failed", crate::toast::ToastKind::Error);
                cx.notify();
            }
        }
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        self.streaming = false;
        self.waiting_for_model = false;
        self.cancel = None;
        cx.notify();
    }

    fn maybe_generate_title(&mut self, cx: &mut Context<Self>) {
        if !self.title_pending {
            return;
        }
        let Some(model) = self.model.clone() else {
            return;
        };
        let Some(user_text) = self
            .messages
            .iter()
            .find(|m| m.role == Role::User)
            .map(|m| m.cur().text.clone())
        else {
            return;
        };
        self.title_pending = false;

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
                    view.chat_title = title;
                    view.save();
                    cx.emit(ChatEvent::Updated);
                    cx.notify();
                });
            }
        })
        .detach();
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

/// Read-only numeric value box on the right of a parameter row.
fn value_box(text: String, theme: &Theme) -> impl IntoElement {
    div()
        .min_w(px(76.))
        .px_3()
        .py_1p5()
        .rounded_md()
        .border_1()
        .border_color(theme.border)
        .bg(theme.bg)
        .text_sm()
        .text_color(theme.text)
        .child(text)
}

/// Small square checkbox used to enable/disable an optional sampling param.
fn param_checkbox(
    id: &'static str,
    checked: bool,
    theme: &Theme,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    let mut b = div()
        .id(id)
        .size(px(18.))
        .flex_none()
        .rounded(crate::tokens::radius::SM)
        .border_1()
        .border_color(theme.border)
        .flex()
        .items_center()
        .justify_center()
        .cursor(gpui::CursorStyle::PointingHand)
        .on_click(on_click);
    if checked {
        b = b.bg(theme.text).child(IconEl::new(Icon::Check, theme.bg).size(crate::tokens::icon::XS));
    }
    b
}

/// A sampling parameter: label (+ optional checkbox) and value box on top, a
/// slider below. `frac` is the slider's normalized position; `on_change`
/// receives the new fraction.
#[allow(clippy::too_many_arguments)]
fn slider_param(
    label: &'static str,
    checkbox: Option<gpui::AnyElement>,
    value_text: String,
    frac: f32,
    slider_id: &'static str,
    theme: &Theme,
    on_change: impl Fn(f32, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(div().text_sm().text_color(theme.text).child(label))
                        .children(checkbox),
                )
                .child(value_box(value_text, theme)),
        )
        .child(Slider::new(slider_id, frac).on_change(on_change))
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
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let streaming = self.streaming;

        // Re-pin the scroll to the bottom for a few frames after content
        // changes: the first layout underestimates wrapped-text height, so a
        // single `scroll_to_bottom` lands short. Each frame re-asserts and
        // schedules the next until the offset converges to the true bottom.
        if self.pin_bottom_frames > 0 {
            self.pin_bottom_frames -= 1;
            self.scroll.scroll_to_bottom();
            let entity = cx.entity();
            window.on_next_frame(move |_, cx| entity.update(cx, |_, cx| cx.notify()));
        }
        let show_reasoning = settings_state::current(cx).reasoning;
        let resolved = self.resolved_model(cx);
        let model_name = resolved
            .as_ref()
            .map(|m| m.name())
            .unwrap_or_else(|| "Select model".to_string());
        let has_model = resolved.is_some();
        // Icon URL for the trigger badge (prefer dark theme logo).
        let trigger_icon_url: Option<String> = resolved.as_ref().and_then(|m| {
            let icons = &m.family.as_ref()?.vendor.metadata.icons;
            icons
                .iter()
                .find(|i| i.theme == uzu::types::basic::ImageTheme::Dark)
                .or_else(|| icons.first())
                .map(|i| i.url.clone())
        });
        let trigger_vendor = resolved.as_ref().and_then(|m| m.family.as_ref().map(|f| f.vendor.name())).unwrap_or_default();

        // Message column (`gap-4`, `pt-4`).
        let mut column = div().flex().flex_col().gap_4().pt_4().w_full();
        if self.messages.is_empty() {
            column = column.child(
                div()
                    .flex()
                    .items_center()
                    .justify_center()
                    .gap_2()
                    .pt(px(220.))
                    .text_sm()
                    .text_color(theme.text_muted)
                    .child("Start your new private and local conversation")
                    .child(IconEl::new(Icon::Logo, theme.text_muted).size(13.)),
            );
        } else {
            let msg_count = self.messages.len();
            for (idx, msg) in self.messages.iter().enumerate() {
                let streaming_here = streaming && idx + 1 == msg_count;
                let cur = msg.cur();
                column = column.child(match msg.role {
                    Role::User => div()
                            .flex()
                            .w_full()
                            .justify_end()
                            .child(
                                div()
                                    .max_w(px(560.))
                                    .flex_shrink_0()
                                    .px_3()
                                    .py_2()
                                    .rounded_lg()
                                    .bg(theme.bg_hover)
                                    .text_color(theme.text)
                                    .child(crate::components::markdown::markdown(
                                        &cur.text, &theme, idx,
                                    )),
                            )
                        .into_any_element(),
                    Role::Assistant => {
                        let mut block = div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .w_full()
                            .min_w_0()
                            .pb_3();

                        // Reasoning ("thinking") panel (hidden when the setting is off).
                        if show_reasoning {
                            if let Some(reasoning) = &cur.reasoning {
                                if !reasoning.trim().is_empty() {
                                    let collapsed = msg.reasoning_collapsed;
                                    let chevron = if collapsed {
                                        Icon::ChevronDown
                                    } else {
                                        Icon::ChevronUp
                                    };
                                    // Header: label on the left, collapse chevron on the right.
                                    let header = div()
                                        .id(SharedString::from(format!("think-hdr-{idx}")))
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .text_xs()
                                        .text_color(theme.text_muted)
                                        .cursor(CursorStyle::PointingHand)
                                        .on_click(cx.listener(move |this, _, _, cx| {
                                            if let Some(m) = this.messages.get_mut(idx) {
                                                m.reasoning_collapsed = !m.reasoning_collapsed;
                                                cx.notify();
                                            }
                                        }))
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_1()
                                                .child(IconEl::new(Icon::Thinking, theme.text_muted).size(13.))
                                                .child(if streaming_here {
                                                    "thinking…"
                                                } else {
                                                    "Thoughts"
                                                }),
                                        )
                                        .child(IconEl::new(chevron, theme.text_muted).size(crate::tokens::icon::XS));

                                    let mut panel = div()
                                        .flex()
                                        .flex_col()
                                        .gap_2()
                                        .p_3()
                                        .rounded_lg()
                                        .bg(theme.bg_sub)
                                        .border_l_2()
                                        .border_color(theme.border)
                                        .child(header);

                                    if !collapsed {
                                        // Monospace body capped at 180 px, scrollable, with a
                                        // bottom fade so the cut-off blends into the panel.
                                        // The live message auto-scrolls to its newest line.
                                        let mut body = div()
                                            .id(SharedString::from(format!("think-body-{idx}")))
                                            .max_h(px(180.))
                                            .overflow_y_scroll()
                                            .font_family(FONT_MONO)
                                            .text_size(crate::tokens::font::SMALL)
                                            .text_color(theme.text_muted)
                                            .child(reasoning.clone());
                                        if streaming_here {
                                            body = body.track_scroll(&self.reasoning_scroll);
                                        }
                                        panel = panel.child(
                                            div()
                                                .relative()
                                                .child(body)
                                                .child(
                                                    div()
                                                        .absolute()
                                                        .bottom_0()
                                                        .left_0()
                                                        .right_0()
                                                        .h(px(28.))
                                                        .bg(gpui::linear_gradient(
                                                            180.,
                                                            gpui::linear_color_stop(theme.bg_sub.opacity(0.), 0.),
                                                            gpui::linear_color_stop(theme.bg_sub, 1.),
                                                        )),
                                                ),
                                        );
                                    }
                                    block = block.child(panel);
                                }
                            }
                        }

                        // Body (markdown for assistant prose + code blocks).
                        let body = if cur.error {
                            div()
                                .p_3()
                                .rounded_md()
                                .border_1()
                                .border_color(theme.error.opacity(0.3))
                                .bg(theme.error.opacity(0.08))
                                .text_sm()
                                .text_color(theme.error)
                                .child(cur.text.clone())
                                .into_any_element()
                        } else if cur.text.is_empty() && streaming {
                            let label = if self.waiting_for_model {
                                "Waiting for model…"
                            } else {
                                "Generating…"
                            };
                            Loader::new().label(label).into_any_element()
                        } else {
                            div()
                                .w_full()
                                .min_w_0()
                                .text_color(theme.text)
                                .child(crate::components::markdown::markdown(&cur.text, &theme, idx))
                                .into_any_element()
                        };
                        block = block.child(body);

                        // Actions (`mt-6 pr-6`, copy left, model + performance right).
                        let has_reasoning = cur
                            .reasoning
                            .as_ref()
                            .is_some_and(|r| !r.trim().is_empty());
                        let show_actions = !streaming
                            && (msg.versions.len() > 1
                                || !cur.text.is_empty()
                                || cur.error
                                || has_reasoning);
                        if show_actions {
                            let copy_text = cur.text.clone();
                            let show_perf = cur
                                .tps
                                .is_some_and(|t| t.is_finite() && t > 0.0)
                                || cur.tokens.is_some_and(|t| t > 0);

                            let mut left = div().flex().items_center().gap_1();
                            if msg.versions.len() > 1 {
                                let total = msg.versions.len();
                                let cur_n = msg.current + 1;
                                left = left
                                    .child(
                                        div()
                                            .id(gpui::SharedString::from(format!("ver-prev-{idx}")))
                                            .cursor(gpui::CursorStyle::PointingHand)
                                            .text_xs()
                                            .text_color(theme.text_muted)
                                            .child("◀")
                                            .on_click(cx.listener(move |this, _, _, cx| {
                                                if let Some(m) = this.messages.get_mut(idx) {
                                                    m.current = m.current.saturating_sub(1);
                                                }
                                                cx.notify();
                                            })),
                                    )
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .gap_1()
                                            .text_xs()
                                            .text_color(theme.text_muted)
                                            .child(format!("{cur_n}/{total}"))
                                            .when_some(cur.model_name.clone(), |el, name| {
                                                el.child(
                                                    div()
                                                        .text_color(theme.text_muted.opacity(0.6))
                                                        .child(format!("· {name}")),
                                                )
                                            }),
                                    )
                                    .child(
                                        div()
                                            .id(gpui::SharedString::from(format!("ver-next-{idx}")))
                                            .cursor(gpui::CursorStyle::PointingHand)
                                            .text_xs()
                                            .text_color(theme.text_muted)
                                            .child("▶")
                                            .on_click(cx.listener(move |this, _, _, cx| {
                                                if let Some(m) = this.messages.get_mut(idx) {
                                                    let max = m.versions.len().saturating_sub(1);
                                                    m.current = (m.current + 1).min(max);
                                                }
                                                cx.notify();
                                            })),
                                    );
                            }
                            if !cur.text.is_empty() || cur.error {
                                left = left.child(
                                    IconButton::new(
                                        gpui::SharedString::from(format!("copy-{idx}")),
                                        Icon::Copy,
                                    )
                                    .color(theme.text_muted)
                                    .icon_size(14.)
                                    .hit_size(24.)
                                    .on_click(cx.listener(move |_this, _, _, cx| {
                                        cx.write_to_clipboard(gpui::ClipboardItem::new_string(
                                            copy_text.clone(),
                                        ));
                                    })),
                                );
                            }

                            let mut right = div().flex().items_center().gap_4();
                            right = right.child(
                                {
                                    let is_open = self.msg_model_picker_open == Some(idx);
                                    let msg_picker = self.model_picker_panel(cx, Some(idx));
                                    let btn_bg = if is_open { theme.bg_hover } else { gpui::transparent_black() };
                                    div()
                                        .relative()
                                        .child(
                                            div()
                                                .id(gpui::SharedString::from(format!("msg-model-{idx}")))
                                                .flex()
                                                .items_center()
                                                .gap_2()
                                                .px(px(6.))
                                                .py_1()
                                                .rounded_md()
                                                .bg(btn_bg)
                                                .cursor(gpui::CursorStyle::PointingHand)
                                                .hover(|s| s.bg(theme.bg_hover))
                                                .on_click(cx.listener(move |this, _, _, cx| {
                                                    let opening = this.msg_model_picker_open != Some(idx);
                                                    this.close_popovers();
                                                    if opening { this.msg_model_picker_open = Some(idx); }
                                                    cx.notify();
                                                }))
                                                .child(IconEl::new(Icon::ModelMenu, theme.text_muted).size(crate::tokens::icon::SM))
                                                .child(
                                                    div()
                                                        .text_size(crate::tokens::font::COMPACT)
                                                        .text_color(theme.text_muted)
                                                        .child("Model"),
                                                ),
                                        )
                                        .when(is_open, |el| {
                                            el.child(Self::anchored_popover(
                                                msg_picker,
                                                Anchor::BottomRight,
                                                |this, _, _, cx| { this.msg_model_picker_open = None; cx.notify(); },
                                                cx,
                                            ))
                                        })
                                },
                            );
                            if show_perf {
                                let perf_open = self.perf_open_msg == Some(idx);
                                let perf_btn_bg = if perf_open { theme.bg_hover } else { gpui::transparent_black() };
                                let perf_panel = self.performance_panel(idx, cur, cx);
                                right = right.child(
                                    div()
                                        .relative()
                                        .child(
                                            div()
                                                .id(gpui::SharedString::from(format!("msg-perf-{idx}")))
                                                .flex()
                                                .items_center()
                                                .gap_2()
                                                .px(px(6.))
                                                .py_1()
                                                .rounded_md()
                                                .bg(perf_btn_bg)
                                                .cursor(gpui::CursorStyle::PointingHand)
                                                .hover(|s| s.bg(theme.bg_hover))
                                                .on_click(cx.listener(move |this, _, _, cx| {
                                                    let opening = this.perf_open_msg != Some(idx);
                                                    this.close_popovers();
                                                    if opening { this.perf_open_msg = Some(idx); }
                                                    cx.notify();
                                                }))
                                                .child(IconEl::new(Icon::Performance, theme.text_muted).size(crate::tokens::icon::SM))
                                                .child(
                                                    div()
                                                        .text_size(crate::tokens::font::COMPACT)
                                                        .text_color(theme.text_muted)
                                                        .child("Performance"),
                                                ),
                                        )
                                        .when(perf_open, |el| {
                                            el.children(perf_panel.map(|panel| {
                                                Self::anchored_popover(
                                                    panel,
                                                    Anchor::BottomRight,
                                                    |this, _, _, cx| { this.perf_open_msg = None; cx.notify(); },
                                                    cx,
                                                )
                                            }))
                                        }),
                                );
                            }

                            block = block.child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .pt_6()
                                    .pr_6()
                                    .child(left)
                                    .child(right),
                            );
                        }

                        block.into_any_element()
                    }
                });
            }
        }

        // Composer send/stop (`message-input.tsx`: filled label-title button, not accent).
        let send_button = if streaming {
            div()
                .id("chat-stop")
                .flex()
                .items_center()
                .justify_center()
                .size(px(24.))
                .rounded_md()
                .bg(theme.text)
                .cursor(gpui::CursorStyle::PointingHand)
                .hover(|s| s.bg(theme.button_border_hover))
                .on_click(cx.listener(|this, _, _, cx| this.stop(cx)))
                .child(IconEl::new(Icon::Stop, theme.text_inverse).size(crate::tokens::icon::SM))
                .into_any_element()
        } else {
            div()
                .id("chat-send")
                .flex()
                .items_center()
                .justify_center()
                .size(px(24.))
                .rounded_md()
                .bg(theme.text)
                .cursor(gpui::CursorStyle::PointingHand)
                .hover(|s| s.bg(theme.button_border_hover))
                .on_click(cx.listener(|this, _, _, cx| this.submit_from_button(cx)))
                .child(IconEl::new(Icon::Send, theme.text_inverse).size(crate::tokens::icon::MD))
                .into_any_element()
        };

        div()
            .size_full()
            .min_h_0()
            .relative()
            .flex()
            .flex_col()
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .flex()
                    .flex_col()
                    .px_5()
                    .pt_4()
                    .pb_5()
                    .items_center()
                    // Scrollable message area (same flex pattern as `chats.rs` list).
                    .child(
                        div()
                            .w_full()
                            .max_w(px(CONTENT_MAX_WIDTH))
                            .flex_1()
                            .min_h_0()
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .id("chat-scroll")
                                    .flex_1()
                                    .min_h_0()
                                    .min_w_0()
                                    .w_full()
                                    .overflow_y_scroll()
                                    .track_scroll(&self.scroll)
                                    .child(column),
                            ),
                    )
                    // Composer (`flex-shrink-0`, max 800px — mirai-chat ChatPage).
                    .child(
                        div()
                            .w_full()
                            .flex_shrink_0()
                            .flex()
                            .flex_col()
                            .items_center()
                            .child(
                                div()
                                    .w_full()
                                    .max_w(px(CONTENT_MAX_WIDTH))
                                    .flex()
                                    .flex_col()
                                    .items_end()
                                    .gap_2()
                                    // `MessageInput`: `gap-4 p-4 rounded-[8px]`.
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_4()
                                            .w_full()
                                            .p_4()
                                            .rounded_lg()
                                            .border_1()
                                            .border_color(theme.border)
                                            .bg(theme.card)
                                            .child(div().w_full().child(self.input.clone()))
                                            // Attached-file chips — each shows the filename
                                            // and an × to remove it before sending.
                                            .when(!self.attached_files.is_empty(), |el| {
                                                let chips = self.attached_files.iter().enumerate().map(|(i, (name, ext, _))| {
                                                    let label = format!("{name} .{ext}");
                                                    div()
                                                        .id(SharedString::from(format!("attach-{i}")))
                                                        .flex()
                                                        .items_center()
                                                        .gap_1()
                                                        .px_2()
                                                        .py_0p5()
                                                        .rounded_md()
                                                        .bg(theme.bg_hover)
                                                        .border_1()
                                                        .border_color(theme.border)
                                                        .text_size(crate::tokens::font::CAPTION)
                                                        .text_color(theme.text_muted)
                                                        .child(IconEl::new(Icon::Rename, theme.text_muted).size(11.))
                                                        .child(label)
                                                        .child(
                                                            div()
                                                                .id(SharedString::from(format!("attach-rm-{i}")))
                                                                .cursor(gpui::CursorStyle::PointingHand)
                                                                .text_color(theme.text_muted)
                                                                .child("×")
                                                                .on_click(cx.listener(move |this, _, _, cx| {
                                                                    if i < this.attached_files.len() {
                                                                        this.attached_files.remove(i);
                                                                        cx.notify();
                                                                    }
                                                                })),
                                                        )
                                                        .into_any_element()
                                                }).collect::<Vec<_>>();
                                                el.child(
                                                    div()
                                                        .flex()
                                                        .flex_wrap()
                                                        .gap_1()
                                                        .children(chips),
                                                )
                                            })
                                            .child(
                                                div()
                                                    .flex()
                                                    .items_center()
                                                    .justify_between()
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .flex_col()
                                                            .items_start()
                                                            .gap_2()
                                                            .children(self.file_upload_panel(cx))
                                                            .child(
                                                                div()
                                                                    .id("file-upload-trigger")
                                                                    .flex()
                                                                    .items_center()
                                                                    .px(px(6.))
                                                                    .py_1()
                                                                    .rounded_md()
                                                                    .cursor(gpui::CursorStyle::PointingHand)
                                                                    .hover(|s| s.bg(theme.bg_hover))
                                                                    .on_click(cx.listener(|this, _, _, cx| {
                                                                        this.file_upload_open =
                                                                            !this.file_upload_open;
                                                                        if this.file_upload_open {
                                                                            this.model_picker_open = false;
                                                                            this.msg_model_picker_open = None;
                                                                            this.perf_open_msg = None;
                                                                        }
                                                                        cx.notify();
                                                                    }))
                                                                    .child(
                                                                        IconEl::new(Icon::Plus, theme.text_muted)
                                                                            .size(crate::tokens::icon::MD),
                                                                    ),
                                                            ),
                                                    )
                                                    .child({
                                                        let mut controls =
                                                            div().flex().items_center().gap(px(10.));
                                                        if has_model {
                                                            controls = controls.child(
                                                                IconButton::new(
                                                                    "gen-settings",
                                                                    Icon::Settings,
                                                                )
                                                                .color(theme.text_muted)
                                                                .icon_size(18.)
                                                                .hit_size(32.)
                                                                .on_click(cx.listener(|this, _, _, cx| {
                                                                    this.gen_settings_open =
                                                                        !this.gen_settings_open;
                                                                    cx.notify();
                                                                })),
                                                            );
                                                        }
                                                        let picker_panel = self
                                                            .model_picker_panel(cx, None);
                                                        controls = controls
                                                            .child(
                                                                div()
                                                                    .relative()
                                                                    .child(
                                                                        div()
                                                                            .id("model-trigger")
                                                                            .flex()
                                                                            .items_center()
                                                                            .gap_1()
                                                                            .px(px(6.))
                                                                            .cursor(gpui::CursorStyle::PointingHand)
                                                                            .on_click(cx.listener(|this, _, _, cx| {
                                                                                let opening = !this.model_picker_open;
                                                                                this.close_popovers();
                                                                                if opening { this.model_picker_open = true; }
                                                                                cx.notify();
                                                                            }))
                                                                            .when(has_model, |el| {
                                                                                el.child(
                                                                                    VendorIcon::new(trigger_vendor.clone())
                                                                                        .size(crate::tokens::icon::MD)
                                                                                        .icon_url(trigger_icon_url.clone()),
                                                                                )
                                                                            })
                                                                            .child(
                                                                                div()
                                                                                    .text_size(crate::tokens::font::COMPACT)
                                                                                    .text_color(theme.text_muted)
                                                                                    .child(model_name.clone()),
                                                                            )
                                                                            .child(
                                                                                IconEl::new(Icon::ChevronDown, theme.text_muted)
                                                                                    .size(crate::tokens::icon::XS),
                                                                            ),
                                                                    )
                                                                    .when(self.model_picker_open, |el| {
                                                                        el.child(Self::anchored_popover(
                                                                            picker_panel,
                                                                            Anchor::BottomRight,
                                                                            |this, _, _, cx| { this.model_picker_open = false; cx.notify(); },
                                                                            cx,
                                                                        ))
                                                                    }),
                                                            )
                                                            .child(send_button);
                                                        controls
                                                    }),
                                            ),
                                    ),
                            ),
                    ),
            )
            .children(self.gen_settings_overlay(cx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, text: &str, error: bool) -> ChatMsg {
        ChatMsg {
            role,
            versions: vec![Version { text: text.into(), error, ..Default::default() }],
            current: 0,
            reasoning_collapsed: false,
        }
    }

    #[test]
    fn user_message_has_single_version() {
        let m = ChatMsg::user("hi".into());
        assert_eq!(m.versions.len(), 1);
        assert_eq!(m.current, 0);
        assert_eq!(m.cur().text, "hi");
    }

    // Regenerate pushes a new version and points `current` at it, keeping priors
    // reachable via the pager.
    #[test]
    fn regenerate_keeps_prior_versions() {
        let mut m = ChatMsg::assistant(Version { text: "v0".into(), ..Default::default() });
        m.versions.push(Version { text: "v1".into(), ..Default::default() });
        m.current = m.versions.len() - 1;
        assert_eq!(m.versions.len(), 2);
        assert_eq!(m.cur().text, "v1");
        m.current = 0;
        assert_eq!(m.cur().text, "v0");
    }

    #[test]
    fn request_excludes_trailing_placeholder() {
        let messages = vec![
            msg(Role::User, "q1", false),
            msg(Role::Assistant, "a1", false),
            msg(Role::User, "q2", false),
            msg(Role::Assistant, "", false), // placeholder being filled
        ];
        let convo = conversation_for_request(&messages);
        assert_eq!(convo.len(), 3);
        assert_eq!(convo[0].1, "q1");
        assert_eq!(convo[2].1, "q2");
    }

    #[test]
    fn request_drops_errored_turns() {
        let messages = vec![
            msg(Role::User, "q1", false),
            msg(Role::Assistant, "boom", true), // errored — excluded from history
            msg(Role::User, "q2", false),
            msg(Role::Assistant, "", false),
        ];
        let convo = conversation_for_request(&messages);
        assert_eq!(convo.len(), 2);
        assert_eq!(convo[0].1, "q1");
        assert_eq!(convo[1].1, "q2");
    }

    #[test]
    fn request_empty_when_only_placeholder() {
        assert!(conversation_for_request(&[msg(Role::Assistant, "", false)]).is_empty());
    }

    #[test]
    fn default_mode_leaves_sampling_to_model() {
        assert!(sampling_method(SamplingMode::Default, 0.7, 0, 0.0, 0.0).is_none());
    }

    #[test]
    fn argmax_mode_is_greedy() {
        assert!(matches!(
            sampling_method(SamplingMode::Argmax, 0.7, 40, 0.9, 0.1),
            Some(SamplingMethod::Greedy {})
        ));
    }

    #[test]
    fn stochastic_zero_params_are_off() {
        match sampling_method(SamplingMode::Stochastic, 0.7, 0, 0.0, 0.0) {
            Some(SamplingMethod::Stochastic { temperature, top_k, top_p, min_p, .. }) => {
                assert_eq!(temperature, Some(0.7f32 as f64));
                assert_eq!(top_k, None);
                assert_eq!(top_p, None);
                assert_eq!(min_p, None);
            }
            _ => panic!("expected stochastic"),
        }
    }

    #[test]
    fn stochastic_nonzero_params_pass_through() {
        match sampling_method(SamplingMode::Stochastic, 0.8, 40, 0.9, 0.05) {
            Some(SamplingMethod::Stochastic { top_k, top_p, min_p, .. }) => {
                assert_eq!(top_k, Some(40));
                assert_eq!(top_p, Some(0.9f32 as f64));
                assert!(min_p.unwrap() > 0.0);
            }
            _ => panic!("expected stochastic"),
        }
    }
}
