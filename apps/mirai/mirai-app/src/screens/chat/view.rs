use std::time::{Duration, Instant};

use gpui::{
    Anchor, Animation, AnimationExt, Context, CursorStyle, Entity, EventEmitter, FontWeight, IntoElement, Render,
    ScrollHandle, SharedString, Window, div, prelude::*, px,
};
use uzu::{session::chat::ChatSession, types::model::Model};

use super::{
    chat_turn::ChatTurn,
    event::ChatEvent,
    params::{param_checkbox, param_row, round2, slider_param},
    role::Role,
    sampling::SamplingMode,
    state::ChatState,
    version::Version,
};
use crate::{
    components::{Icon, IconButton, IconEl, InputEvent, Loader, SegmentedControl, TextInput, Toggle, VendorIcon},
    models_store::ModelsStore,
    persistence::{self, StoredChat, StoredMessage},
    settings_state,
    theme::{ActiveTheme, FONT_MONO, layout::CONTENT_MAX_WIDTH},
    title_gen,
};

fn dark_icon_url(model: &Model) -> Option<String> {
    let icons = &model.family.as_ref()?.vendor.metadata.icons;
    icons
        .iter()
        .find(|i| i.theme == uzu::types::basic::ImageTheme::Dark)
        .or_else(|| icons.first())
        .map(|i| i.url.clone())
}

pub struct ChatView {
    pub(super) state: ChatState,
    pub(super) store: Entity<ModelsStore>,

    pub(super) cloud_store: Entity<ModelsStore>,
    pub(super) input: Entity<TextInput>,
    pub(super) scroll: ScrollHandle,

    pub(super) reasoning_scroll: ScrollHandle,

    pub(super) pin_bottom_frames: u8,

    pub(super) last_active: Instant,
}

impl EventEmitter<ChatEvent> for ChatView {}

impl ChatView {
    pub fn new(
        store: Entity<ModelsStore>,
        cloud_store: Entity<ModelsStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(|cx| TextInput::new(cx, "Add message…").multiline(true, 1, 8));
        cx.subscribe(&input, |this, input, event, cx| match event {
            InputEvent::Submit(text) => {
                if this.state.streaming {
                    return;
                }
                let text = text.clone();
                input.update(cx, |input, cx| input.clear(cx));
                this.send(text, cx);
            },
            InputEvent::Changed(_) => {},
        })
        .detach();
        cx.observe(&store, |_, _, cx| cx.notify()).detach();
        cx.observe(&cloud_store, |_, _, cx| cx.notify()).detach();
        settings_state::observe(cx, |_, cx| cx.notify()).detach();

        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(Duration::from_secs(30)).await;
                if this.update(cx, |view, cx| view.maybe_auto_eject(cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        Self {
            state: ChatState {
                messages: Vec::new(),
                model: None,
                streaming: false,
                waiting_for_model: false,
                cancel: None,
                stream_gen: 0,
                chat_id: None,
                created_at: persistence::now_ms(),
                model_picker_open: false,
                msg_model_picker_open: None,
                pending_regen: None,
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
            },
            store,
            cloud_store,
            input,
            scroll: ScrollHandle::new(),
            reasoning_scroll: ScrollHandle::new(),
            pin_bottom_frames: 0,
            last_active: Instant::now(),
        }
    }

    fn maybe_auto_eject(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let settings = settings_state::current(cx);
        if !settings.auto_eject || self.state.streaming || self.state.loaded_model.is_none() {
            return;
        }
        let idle = self.last_active.elapsed().as_secs();
        if idle >= u64::from(settings.idle_timeout_minutes.max(1)) * 60 {
            self.eject(cx);
        }
    }

    pub(super) fn clear_session(&mut self) {
        self.state.session = None;
        self.state.session_model_id = None;
    }

    pub(super) fn cancel_stream(&mut self) {
        if let Some(token) = self.state.cancel.take() {
            token.cancel();
        }

        self.state.stream_gen = self.state.stream_gen.wrapping_add(1);
        self.state.streaming = false;
        self.state.waiting_for_model = false;
    }

    pub(super) fn cached_session(
        &self,
        model_id: &str,
    ) -> Option<ChatSession> {
        self.state.session.as_ref().filter(|_| self.state.session_model_id.as_deref() == Some(model_id)).cloned()
    }

    pub(super) fn store_session(
        &mut self,
        session: ChatSession,
        model_id: &str,
    ) {
        self.state.session = Some(session);
        self.state.session_model_id = Some(model_id.to_string());
    }

    pub fn loaded_model_name(&self) -> Option<String> {
        self.state.loaded_model.clone()
    }

    pub fn eject(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.cancel_stream();
        self.state.loaded_model = None;
        self.clear_session();
        cx.notify();
    }

    pub(super) fn close_popovers(&mut self) {
        self.state.model_picker_open = false;
        self.state.msg_model_picker_open = None;
        self.state.perf_open_msg = None;
        self.state.file_upload_open = false;
    }

    pub fn start_new(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.state.messages.clear();
        self.state.chat_id = None;
        self.state.created_at = persistence::now_ms();
        self.state.chat_title.clear();
        self.state.title_pending = false;
        self.cancel_stream();
        self.state.attached_files.clear();
        self.state.pending_regen = None;
        self.clear_session();
        cx.notify();
    }

    pub fn use_model(
        &mut self,
        model: Model,
        cx: &mut Context<Self>,
    ) {
        if let Some(idx) = self.state.pending_regen.take() {
            let valid = self.state.messages.get(idx).is_some_and(|m| m.role == Role::Assistant);
            if valid && !self.state.streaming {
                self.regenerate_at_with_model(idx, Some(model), cx);
                return;
            }
        }
        self.state.model = Some(model);
        self.start_new(cx);
    }

    pub fn load_stored(
        &mut self,
        stored: StoredChat,
        cx: &mut Context<Self>,
    ) {
        self.state.messages = stored
            .messages
            .into_iter()
            .map(|m| ChatTurn {
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
                    ttft: m.ttft,
                    total_time: m.total_time,
                    error: false,
                    ..Default::default()
                }],
                current: 0,

                reasoning_collapsed: true,
            })
            .collect();
        self.state.chat_id = Some(stored.id);
        self.state.created_at = stored.created_at;
        self.state.chat_title = stored.title;
        self.state.title_pending = false;

        self.state.model = stored
            .model_id
            .as_deref()
            .and_then(|id| self.find_model(cx, |r| r.model.identifier == id))
            .or_else(|| stored.model_name.as_deref().and_then(|name| self.find_model(cx, |r| r.name() == name)));
        self.cancel_stream();
        self.state.pending_regen = None;
        self.clear_session();

        self.pin_to_bottom();
        cx.notify();
    }

    pub(super) fn pin_to_bottom(&mut self) {
        self.scroll.scroll_to_bottom();
        self.pin_bottom_frames = 8;
    }

    pub(super) fn should_auto_scroll(&self) -> bool {
        let offset = self.scroll.offset();
        let max = self.scroll.max_offset();
        if max.y <= px(0.) {
            return true;
        }
        offset.y <= -(max.y - px(32.))
    }

    pub(super) fn save(&mut self) {
        if !self.state.messages.iter().any(|m| m.role == Role::User) {
            return;
        }
        let id = self.state.chat_id.clone().unwrap_or_else(|| format!("chat-{}", self.state.created_at));
        self.state.chat_id = Some(id.clone());
        let title = if title_gen::is_placeholder(&self.state.chat_title) {
            self.state
                .messages
                .iter()
                .find(|m| m.role == Role::User)
                .map(|m| truncate(&m.cur().text, 48))
                .unwrap_or_else(|| "New chat".to_string())
        } else {
            self.state.chat_title.clone()
        };
        let messages = self
            .state
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
                ttft: m.cur().ttft,
                total_time: m.cur().total_time,
            })
            .collect();
        persistence::save_chat(&StoredChat {
            id,
            title,
            model_name: self.state.model.as_ref().map(|m| m.name()),
            model_id: self.state.model.as_ref().map(|m| m.identifier.clone()),
            created_at: self.state.created_at,
            updated_at: persistence::now_ms(),
            messages,
        });
    }

    fn gen_settings_overlay(
        &self,
        cx: &mut Context<Self>,
    ) -> Option<gpui::AnyElement> {
        if !self.state.gen_settings_open {
            return None;
        }
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let border = theme.border;
        let fg = theme.text;

        let mode_row = div().flex().flex_col().gap_1().child(div().text_sm().text_color(fg).child("Sampling")).child(
            SegmentedControl::new("sampling-mode", self.state.sampling_mode as usize)
                .segment(
                    "Default",
                    cx.listener(|this, _, _, cx| {
                        this.state.sampling_mode = SamplingMode::Default;
                        cx.notify();
                    }),
                )
                .segment(
                    "Argmax",
                    cx.listener(|this, _, _, cx| {
                        this.state.sampling_mode = SamplingMode::Argmax;
                        cx.notify();
                    }),
                )
                .segment(
                    "Stochastic",
                    cx.listener(|this, _, _, cx| {
                        this.state.sampling_mode = SamplingMode::Stochastic;
                        cx.notify();
                    }),
                ),
        );

        let resolved = self.resolved_model(cx);
        let model_name = resolved.as_ref().map(|m| m.name()).unwrap_or_else(|| "No model".to_string());
        let vendor = resolved.as_ref().and_then(|m| m.family.as_ref().map(|f| f.vendor.name())).unwrap_or_default();
        let icon_url = resolved.as_ref().and_then(dark_icon_url);
        let reasoning_on = settings_state::current(cx).reasoning;

        let header = div()
            .flex()
            .items_center()
            .justify_between()
            .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).text_color(fg).child("Edit parameters"))
            .child(
                IconButton::new("gen-close", Icon::Close)
                    .color(theme.text_muted)
                    .icon_size(16.)
                    .hit_size(28.)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.state.gen_settings_open = false;
                        cx.notify();
                    })),
            );

        let model_row = div()
            .flex()
            .items_center()
            .gap_2()
            .child(VendorIcon::new(vendor).size(crate::tokens::icon::XL).icon_url(icon_url))
            .child(div().flex_1().min_w_0().truncate().text_sm().text_color(fg).child(model_name))
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

        if self.state.sampling_mode == SamplingMode::Stochastic {
            let view = cx.entity();
            let v = view.clone();
            card = card.child(slider_param(
                "Temperature",
                None,
                round2(self.state.temperature).to_string(),
                (self.state.temperature / 2.0).clamp(0., 1.),
                "temp-slider",
                &theme,
                move |frac, _, cx| {
                    v.update(cx, |this, cx| {
                        this.state.temperature = round2(frac * 2.0);
                        cx.notify();
                    });
                },
            ));
            let v = view.clone();
            card = card.child(slider_param(
                "Top K",
                None,
                self.state.top_k.to_string(),
                (self.state.top_k as f32 / 200.0).clamp(0., 1.),
                "topk-slider",
                &theme,
                move |frac, _, cx| {
                    v.update(cx, |this, cx| {
                        this.state.top_k = (frac * 200.0).round() as u32;
                        cx.notify();
                    });
                },
            ));
            let v = view.clone();
            let topp_box = param_checkbox("topp-cb", self.state.top_p > 0.0, &theme, {
                let v = view.clone();
                move |_, _, cx| {
                    v.update(cx, |this, cx| {
                        this.state.top_p = if this.state.top_p > 0.0 {
                            0.0
                        } else {
                            0.95
                        };
                        cx.notify();
                    });
                }
            });
            card = card.child(slider_param(
                "Top P",
                Some(topp_box.into_any_element()),
                round2(self.state.top_p).to_string(),
                self.state.top_p.clamp(0., 1.),
                "topp-slider",
                &theme,
                move |frac, _, cx| {
                    v.update(cx, |this, cx| {
                        this.state.top_p = round2(frac);
                        cx.notify();
                    });
                },
            ));
            let v = view.clone();
            let minp_box = param_checkbox("minp-cb", self.state.min_p > 0.0, &theme, {
                let v = view.clone();
                move |_, _, cx| {
                    v.update(cx, |this, cx| {
                        this.state.min_p = if this.state.min_p > 0.0 {
                            0.0
                        } else {
                            0.05
                        };
                        cx.notify();
                    });
                }
            });
            card = card.child(slider_param(
                "Min P",
                Some(minp_box.into_any_element()),
                round2(self.state.min_p).to_string(),
                self.state.min_p.clamp(0., 1.),
                "minp-slider",
                &theme,
                move |frac, _, cx| {
                    v.update(cx, |this, cx| {
                        this.state.min_p = round2(frac);
                        cx.notify();
                    });
                },
            ));
        }

        card = card.child(div().h_px().bg(border)).child(reasoning_row);

        let tokens_str = if self.state.max_tokens == 0 {
            "∞".to_string()
        } else {
            self.state.max_tokens.to_string()
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
                this.state.max_tokens = this.state.max_tokens.saturating_sub(128);
                cx.notify();
            }),
            cx.listener(|this, _, _, cx| {
                this.state.max_tokens = (this.state.max_tokens + 128).min(8192);
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
                    this.state.gen_settings_open = false;
                    cx.notify();
                }))
                .child(card)
                .into_any_element(),
        )
    }

    pub(super) fn resolved_model(
        &self,
        cx: &Context<Self>,
    ) -> Option<Model> {
        let store = self.store.read(cx);

        if let Some(model) = &self.state.model
            && !store.rows.iter().any(|r| r.model.identifier == model.identifier)
        {
            return Some(model.clone());
        }
        store.resolve_installed(self.state.model.as_ref())
    }

    fn find_model(
        &self,
        cx: &Context<Self>,
        matches: impl Fn(&crate::models_store::ModelRow) -> bool,
    ) -> Option<Model> {
        let lookup =
            |store: &Entity<ModelsStore>| store.read(cx).rows.iter().find(|r| matches(r)).map(|r| r.model.clone());
        lookup(&self.store).or_else(|| lookup(&self.cloud_store))
    }

    fn submit_from_button(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let text = self.input.read(cx).text();
        self.input.update(cx, |input, cx| input.clear(cx));
        self.send(text, cx);
    }
}

fn truncate(
    s: &str,
    max: usize,
) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max {
        trimmed.to_string()
    } else {
        let cut: String = trimmed.chars().take(max).collect();
        format!("{cut}…")
    }
}

impl Render for ChatView {
    fn render(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let streaming = self.state.streaming;

        if self.pin_bottom_frames > 0 {
            self.pin_bottom_frames -= 1;
            self.scroll.scroll_to_bottom();
            let entity = cx.entity();
            window.on_next_frame(move |_, cx| entity.update(cx, |_, cx| cx.notify()));
        }
        let show_reasoning = settings_state::current(cx).reasoning;
        let resolved = self.resolved_model(cx);
        let model_name = resolved.as_ref().map(|m| m.name()).unwrap_or_else(|| "Select model".to_string());
        let has_model = resolved.is_some();
        let trigger_icon_url = resolved.as_ref().and_then(dark_icon_url);
        let trigger_vendor =
            resolved.as_ref().and_then(|m| m.family.as_ref().map(|f| f.vendor.name())).unwrap_or_default();

        let mut column = div().flex().flex_col().gap_4().pt_4().w_full();
        if self.state.messages.is_empty() {
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
            let msg_count = self.state.messages.len();
            for (idx, msg) in self.state.messages.iter().enumerate() {
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
                                .child(crate::components::markdown::render(&cur.parsed_markdown(), &theme, idx)),
                        )
                        .into_any_element(),
                    Role::Assistant => {
                        let mut block = div().flex().flex_col().gap_2().w_full().min_w_0().pb_3();

                        if show_reasoning
                            && let Some(reasoning) = &cur.reasoning
                            && !reasoning.trim().is_empty()
                        {
                            let collapsed = msg.reasoning_collapsed;
                            let chevron = if collapsed {
                                Icon::ChevronDown
                            } else {
                                Icon::ChevronUp
                            };
                            let header = div()
                                .id(SharedString::from(format!("think-hdr-{idx}")))
                                .flex()
                                .items_center()
                                .justify_between()
                                .text_xs()
                                .text_color(theme.text_muted)
                                .cursor(CursorStyle::PointingHand)
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    if let Some(m) = this.state.messages.get_mut(idx) {
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
                                let text = div()
                                    .font_family(FONT_MONO)
                                    .text_size(crate::tokens::font::SMALL)
                                    .text_color(theme.text_muted)
                                    .child(reasoning.clone());
                                if streaming_here {
                                    panel = panel.child(
                                        div()
                                            .relative()
                                            .child(
                                                div()
                                                    .id(SharedString::from(format!("think-body-{idx}")))
                                                    .max_h(px(180.))
                                                    .overflow_y_scroll()
                                                    .track_scroll(&self.reasoning_scroll)
                                                    .child(text),
                                            )
                                            .child(div().absolute().bottom_0().left_0().right_0().h(px(28.)).bg(
                                                gpui::linear_gradient(
                                                    180.,
                                                    gpui::linear_color_stop(theme.bg_sub.opacity(0.), 0.),
                                                    gpui::linear_color_stop(theme.bg_sub, 1.),
                                                ),
                                            )),
                                    );
                                } else {
                                    panel = panel.child(text);
                                }
                            }
                            block = block.child(panel);
                        }

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
                            let label = if self.state.waiting_for_model {
                                "Waiting for model…"
                            } else {
                                "Generating…"
                            };

                            div()
                                .child(Loader::new().label(label))
                                .with_animation(
                                    "stream-pulse",
                                    Animation::new(Duration::from_millis(1400)).repeat(),
                                    |el, delta| {
                                        let t = 1.0 - (2.0 * delta - 1.0).abs();
                                        el.opacity(0.5 + 0.5 * t)
                                    },
                                )
                                .into_any_element()
                        } else {
                            div()
                                .w_full()
                                .min_w_0()
                                .text_color(theme.text)
                                .child(crate::components::markdown::render(&cur.parsed_markdown(), &theme, idx))
                                .into_any_element()
                        };
                        block = block.child(body);

                        let has_reasoning = cur.reasoning.as_ref().is_some_and(|r| !r.trim().is_empty());
                        let show_actions = !streaming
                            && (msg.versions.len() > 1 || !cur.text.is_empty() || cur.error || has_reasoning);
                        if show_actions {
                            let copy_text = cur.text.clone();
                            let show_perf =
                                cur.tps.is_some_and(|t| t.is_finite() && t > 0.0) || cur.tokens.is_some_and(|t| t > 0);

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
                                                if let Some(m) = this.state.messages.get_mut(idx) {
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
                                                if let Some(m) = this.state.messages.get_mut(idx) {
                                                    let max = m.versions.len().saturating_sub(1);
                                                    m.current = (m.current + 1).min(max);
                                                }
                                                cx.notify();
                                            })),
                                    );
                            }
                            if !cur.text.is_empty() || cur.error {
                                left = left.child(
                                    IconButton::new(gpui::SharedString::from(format!("copy-{idx}")), Icon::Copy)
                                        .color(theme.text_muted)
                                        .icon_size(14.)
                                        .hit_size(24.)
                                        .on_click(cx.listener(move |_this, _, _, cx| {
                                            cx.write_to_clipboard(gpui::ClipboardItem::new_string(copy_text.clone()));
                                        })),
                                );
                            }

                            let mut right = div().flex().items_center().gap_4();
                            right = right.child({
                                let is_open = self.state.msg_model_picker_open == Some(idx);
                                let msg_picker = self.model_picker_panel(cx, Some(idx));
                                let btn_bg = if is_open {
                                    theme.bg_hover
                                } else {
                                    gpui::transparent_black()
                                };
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
                                                let opening = this.state.msg_model_picker_open != Some(idx);
                                                this.close_popovers();
                                                if opening {
                                                    this.state.msg_model_picker_open = Some(idx);
                                                }
                                                cx.notify();
                                            }))
                                            .child(
                                                IconEl::new(Icon::ModelMenu, theme.text_muted)
                                                    .size(crate::tokens::icon::SM),
                                            )
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
                                            |this, _, _, cx| {
                                                this.state.msg_model_picker_open = None;
                                                cx.notify();
                                            },
                                            cx,
                                        ))
                                    })
                            });
                            if show_perf {
                                let perf_open = self.state.perf_open_msg == Some(idx);
                                let perf_btn_bg = if perf_open {
                                    theme.bg_hover
                                } else {
                                    gpui::transparent_black()
                                };
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
                                                    let opening = this.state.perf_open_msg != Some(idx);
                                                    this.close_popovers();
                                                    if opening {
                                                        this.state.perf_open_msg = Some(idx);
                                                    }
                                                    cx.notify();
                                                }))
                                                .child(
                                                    IconEl::new(Icon::Performance, theme.text_muted)
                                                        .size(crate::tokens::icon::SM),
                                                )
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
                                                    |this, _, _, cx| {
                                                        this.state.perf_open_msg = None;
                                                        cx.notify();
                                                    },
                                                    cx,
                                                )
                                            }))
                                        }),
                                );
                            }

                            block = block.child(
                                div().flex().items_center().justify_between().pt_6().pr_6().child(left).child(right),
                            );
                        }

                        block.into_any_element()
                    },
                });
            }
        }

        let (btn_id, btn_icon, btn_icon_size) = if streaming {
            ("chat-stop", Icon::Stop, crate::tokens::icon::SM)
        } else {
            ("chat-send", Icon::Send, crate::tokens::icon::MD)
        };
        let send_button = div()
            .id(btn_id)
            .flex()
            .items_center()
            .justify_center()
            .size(px(24.))
            .rounded_md()
            .bg(theme.text)
            .cursor(gpui::CursorStyle::PointingHand)
            .hover(|s| s.bg(theme.button_border_hover))
            .on_click(cx.listener(move |this, _, _, cx| {
                if streaming {
                    this.stop(cx);
                } else {
                    this.submit_from_button(cx);
                }
            }))
            .child(IconEl::new(btn_icon, theme.text_inverse).size(btn_icon_size));

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
                    .child(
                        div().w_full().max_w(px(CONTENT_MAX_WIDTH)).flex_1().min_h_0().flex().flex_col().child(
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
                    .child(
                        div().w_full().flex_shrink_0().flex().flex_col().items_center().child(
                            div().w_full().max_w(px(CONTENT_MAX_WIDTH)).flex().flex_col().items_end().gap_2().child(
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
                                    .when(!self.state.attached_files.is_empty(), |el| {
                                        let chips = self
                                            .state
                                            .attached_files
                                            .iter()
                                            .enumerate()
                                            .map(|(i, (name, ext, _))| {
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
                                                                if i < this.state.attached_files.len() {
                                                                    this.state.attached_files.remove(i);
                                                                    cx.notify();
                                                                }
                                                            })),
                                                    )
                                                    .into_any_element()
                                            })
                                            .collect::<Vec<_>>();
                                        el.child(div().flex().flex_wrap().gap_1().children(chips))
                                    })
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .justify_between()
                                            .child({
                                                let upload_panel = self.file_upload_panel(cx);
                                                div()
                                                    .relative()
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
                                                                this.state.file_upload_open =
                                                                    !this.state.file_upload_open;
                                                                if this.state.file_upload_open {
                                                                    this.state.model_picker_open = false;
                                                                    this.state.msg_model_picker_open = None;
                                                                    this.state.perf_open_msg = None;
                                                                }
                                                                cx.notify();
                                                            }))
                                                            .child(
                                                                IconEl::new(Icon::Plus, theme.text_muted)
                                                                    .size(crate::tokens::icon::MD),
                                                            ),
                                                    )
                                                    .when_some(upload_panel, |el, panel| {
                                                        el.child(Self::anchored_popover(
                                                            panel,
                                                            Anchor::BottomLeft,
                                                            |this, _, _, cx| {
                                                                this.state.file_upload_open = false;
                                                                cx.notify();
                                                            },
                                                            cx,
                                                        ))
                                                    })
                                            })
                                            .child({
                                                let mut controls = div().flex().items_center().gap(px(10.));
                                                if has_model {
                                                    controls = controls.child(
                                                        IconButton::new("gen-settings", Icon::Settings)
                                                            .color(theme.text_muted)
                                                            .icon_size(18.)
                                                            .hit_size(32.)
                                                            .on_click(cx.listener(|this, _, _, cx| {
                                                                this.state.gen_settings_open =
                                                                    !this.state.gen_settings_open;
                                                                cx.notify();
                                                            })),
                                                    );
                                                }
                                                let picker_panel = self.model_picker_panel(cx, None);
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
                                                                        let opening = !this.state.model_picker_open;
                                                                        this.close_popovers();
                                                                        if opening {
                                                                            this.state.model_picker_open = true;
                                                                        }
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
                                                                        IconEl::new(
                                                                            Icon::ChevronDown,
                                                                            theme.text_muted,
                                                                        )
                                                                        .size(crate::tokens::icon::XS),
                                                                    ),
                                                            )
                                                            .when(self.state.model_picker_open, |el| {
                                                                el.child(Self::anchored_popover(
                                                                    picker_panel,
                                                                    Anchor::BottomRight,
                                                                    |this, _, _, cx| {
                                                                        this.state.model_picker_open = false;
                                                                        cx.notify();
                                                                    },
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
