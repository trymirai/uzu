use futures::{StreamExt, channel::mpsc};
use gpui::{
    Context, CursorStyle, Entity, FontWeight, IntoElement, Render, SharedString, Window, div, prelude::*, px, relative,
};
use uzu::{
    player::Player,
    session::text_to_speech::TextToSpeechSessionStreamChunk,
    types::{
        basic::{CancelToken, PcmBatch},
        model::Model,
    },
};

use super::{
    pending_generation::PendingGeneration, right_pane_tab::RightPaneTab, synthesis_message::SynthesisMessage, vm::TtsVm,
};
use crate::{
    components::{Button, ButtonKind, Icon, IconButton, IconEl, InputEvent, Loader, TextInput, VendorIcon},
    engine,
    models_store::ModelsStore,
    text::truncate_with_ellipsis,
    theme::ActiveTheme,
    tts_history::{self, TtsHistoryEntry},
};

const CHAR_LIMIT: usize = 2000;

pub struct TtsView {
    store: Entity<ModelsStore>,
    input: Entity<TextInput>,
    selected: Option<Model>,
    player: Option<Player>,
    generating: bool,
    generation_id: u64,
    cancel: Option<CancelToken>,
    error: Option<String>,
    history: Vec<TtsHistoryEntry>,
    playing_id: Option<String>,
    pending_batches: Vec<PcmBatch>,
    pending_gen: Option<PendingGeneration>,
    right_tab: RightPaneTab,
}

impl TtsView {
    pub fn new(
        store: Entity<ModelsStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let input = cx.new(|cx| {
            TextInput::new(cx, "Start typing here or paste any text you want to turn into speech…")
                .multiline(false, 16, 40)
        });

        cx.subscribe(&input, |_, _input, event, cx| match event {
            InputEvent::Submit(_) | InputEvent::Changed(_) => cx.notify(),
        })
        .detach();
        cx.observe(&store, |_, _, cx| cx.notify()).detach();

        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(std::time::Duration::from_millis(250)).await;
                if this.update(cx, |view, cx| view.tick_playback(cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        Self {
            store,
            input,
            selected: None,
            player: None,
            generating: false,
            generation_id: 0,
            cancel: None,
            error: None,
            history: tts_history::list(),
            playing_id: None,
            pending_batches: Vec::new(),
            pending_gen: None,
            right_tab: RightPaneTab::Settings,
        }
    }

    fn pick_text_file(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let rx = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open text file".into()),
        });
        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else {
                return;
            };
            let Some(path) = paths.first() else {
                return;
            };
            if let Ok(content) = std::fs::read_to_string(path) {
                let _ = this.update(cx, |this, cx| {
                    this.input.update(cx, |input, cx| input.set_text(&content, cx));
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn clear_gen(&mut self) {
        self.pending_batches.clear();
        self.pending_gen = None;
    }

    fn append_pcm(
        &mut self,
        batch: PcmBatch,
    ) -> Result<(), String> {
        if self.player.is_none() {
            self.player = Some(Player::new().map_err(|e| format!("audio: {e}"))?);
        }
        self.player.as_ref().expect("player just opened").append_pcm_batch(batch).map_err(|e| format!("audio: {e}"))
    }

    fn reload_history(&mut self) {
        self.history = tts_history::list();
    }

    pub fn reload_after_clear(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        self.stop_playback(cx);
        self.generation_id = self.generation_id.wrapping_add(1);
        self.reload_history();
        cx.notify();
    }

    fn resolved_model(
        &self,
        cx: &Context<Self>,
    ) -> Option<Model> {
        self.store.read(cx).resolve_installed(self.selected.as_ref())
    }

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
        gpui_tokio::Tokio::spawn(cx, async move {
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

    fn play_history(
        &mut self,
        id: &str,
        cx: &mut Context<Self>,
    ) {
        if self.playing_id.as_deref() == Some(id) {
            self.stop_playback(cx);
            return;
        }
        let Some(batch) = tts_history::load_pcm(id) else {
            self.error = Some("audio file missing".into());
            cx.notify();
            return;
        };
        if let Some(player) = &self.player {
            player.stop();
        }
        if self.append_pcm(batch).is_err() {
            cx.notify();
            return;
        }
        self.playing_id = Some(id.to_string());
        self.error = None;
        cx.notify();
    }

    fn stop_playback(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        if let Some(player) = &self.player {
            player.stop();
        }
        self.playing_id = None;
        cx.notify();
    }

    fn tick_playback(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        if self.playing_id.is_some() && self.player.as_ref().is_some_and(|p| p.is_finished()) {
            self.playing_id = None;
            cx.notify();
        }
    }

    fn delete_history(
        &mut self,
        id: &str,
        cx: &mut Context<Self>,
    ) {
        if self.playing_id.as_deref() == Some(id) {
            self.stop_playback(cx);
        }
        tts_history::delete(id);
        self.reload_history();
        cx.notify();
    }

    fn restore_text(
        &mut self,
        text: &str,
        cx: &mut Context<Self>,
    ) {
        self.input.update(cx, |input, cx| input.set_text(text, cx));
        cx.notify();
    }

    fn stop(
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

    fn generate_from_button(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let text = self.input.read(cx).text();
        self.generate(text, cx);
    }

    fn model_card(
        &self,
        cx: &mut Context<Self>,
        vm: &TtsVm,
        selected: bool,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let id = vm.id.clone();

        let control = if vm.installed {
            let id = id.clone();
            IconButton::new(gpui::SharedString::from(format!("del-{}", vm.id)), Icon::Trash)
                .color(theme.text_muted)
                .icon_size(15.)
                .on_click(cx.listener(move |this, _, _, cx| {
                    let id = id.clone();
                    this.store.update(cx, |s, cx| s.delete(id, cx));
                }))
                .into_any_element()
        } else if vm.downloading {
            div()
                .text_xs()
                .text_color(theme.text_muted)
                .child(format!("{:.0}%", vm.progress * 100.0))
                .into_any_element()
        } else {
            let id = id.clone();
            IconButton::new(gpui::SharedString::from(format!("dl-{}", vm.id)), Icon::Download)
                .color(theme.text_muted)
                .icon_size(15.)
                .on_click(cx.listener(move |this, _, _, cx| {
                    let id = id.clone();
                    this.store.update(cx, |s, cx| s.toggle_download(id, cx));
                }))
                .into_any_element()
        };

        let badge = if vm.installed {
            div()
                .flex()
                .items_center()
                .gap_1()
                .px_2()
                .py_0p5()
                .rounded_md()
                .bg(theme.success.opacity(0.12))
                .text_size(crate::tokens::font::CAPTION)
                .text_color(theme.success)
                .child(IconEl::new(Icon::Check, theme.success).size(11.))
                .child("Installed")
                .into_any_element()
        } else if vm.downloading {
            div()
                .text_size(crate::tokens::font::CAPTION)
                .text_color(theme.text_muted)
                .child(format!("Downloading {:.0}%", vm.progress * 100.0))
                .into_any_element()
        } else {
            div()
                .text_size(crate::tokens::font::CAPTION)
                .text_color(theme.text_muted)
                .child("Not installed")
                .into_any_element()
        };

        let select_id = id.clone();
        let border = if selected {
            theme.success
        } else {
            theme.border
        };
        div()
            .id(gpui::SharedString::from(vm.id.clone()))
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .rounded_lg()
            .border_1()
            .border_color(border)
            .bg(theme.bg_sub)
            .when(vm.installed, |el| el.cursor(CursorStyle::PointingHand))
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(model) = this.store.read(cx).installed_model_by_id(&select_id) {
                    this.selected = Some(model);
                    cx.notify();
                }
            }))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        VendorIcon::new(vm.vendor.clone()).size(crate::tokens::icon::XXL).icon_url(vm.icon_url.clone()),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child(vm.name.clone()),
                    )
                    .child(div().text_xs().text_color(theme.text_muted).child(vm.size.clone()))
                    .child(control),
            )
            .child(div().flex().child(badge))
            .when(vm.downloading, |el| {
                el.child(
                    div()
                        .w_full()
                        .h(px(4.))
                        .rounded_full()
                        .bg(theme.bg_hover)
                        .child(div().h_full().w(relative(vm.progress.clamp(0., 1.))).rounded_full().bg(theme.text)),
                )
            })
    }

    fn history_row(
        &self,
        cx: &mut Context<Self>,
        entry: &TtsHistoryEntry,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let playing = self.playing_id.as_deref() == Some(entry.id.as_str());
        let preview = truncate_with_ellipsis(&entry.text, 72);
        let id = entry.id.clone();
        let play_id = id.clone();
        let del_id = id.clone();
        let restore = entry.text.clone();

        div()
            .id(SharedString::from(format!("hist-{}", entry.id)))
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .min_h(px(52.))
            .px_3()
            .rounded_lg()
            .hover(move |s| s.bg(hover))
            .child(
                div()
                    .id(SharedString::from(format!("restore-{}", entry.id)))
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .cursor(CursorStyle::PointingHand)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.restore_text(&restore, cx);
                    }))
                    .child(div().text_sm().text_color(theme.text).overflow_hidden().child(preview))
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.text_muted)
                            .child(format!("{} · {}", entry.model_name, entry.vendor)),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(
                        IconButton::new(
                            SharedString::from(format!("play-{}", entry.id)),
                            if playing {
                                Icon::Stop
                            } else {
                                Icon::Speech
                            },
                        )
                        .color(theme.text_muted)
                        .disabled(self.generating)
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if playing {
                                this.stop_playback(cx);
                            } else {
                                this.play_history(&play_id, cx);
                            }
                        })),
                    )
                    .child(
                        IconButton::new(SharedString::from(format!("del-hist-{}", entry.id)), Icon::Trash)
                            .color(theme.text_muted)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.delete_history(&del_id, cx);
                            })),
                    ),
            )
    }

    fn tab_button(
        &self,
        cx: &mut Context<Self>,
        label: &'static str,
        tab: RightPaneTab,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let active = self.right_tab == tab;
        div()
            .id(label)
            .flex_1()
            .flex()
            .items_center()
            .justify_center()
            .h(px(48.))
            .text_sm()
            .text_color(if active {
                theme.text
            } else {
                theme.text_muted
            })
            .border_b_2()
            .border_color(if active {
                theme.text
            } else {
                gpui::transparent_black()
            })
            .cursor(CursorStyle::PointingHand)
            .on_click(cx.listener(move |this, _, _, cx| {
                this.right_tab = tab;
                cx.notify();
            }))
            .child(label)
    }
}

impl Render for TtsView {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let selected_id = self.selected.as_ref().map(|m| m.identifier.clone());
        let resolved = self.resolved_model(cx);

        let (loading, models): (bool, Vec<TtsVm>) = {
            let store = self.store.read(cx);
            let rows = store.rows.iter().map(|r| TtsVm::from_row(r, theme.dark)).collect();
            (store.loading, rows)
        };
        let any_installed = models.iter().any(|r| r.installed);
        let char_count = self.input.read(cx).text().chars().count();
        let over = char_count > CHAR_LIMIT;

        let header_badge = resolved.as_ref().and_then(|m| models.iter().find(|v| v.id == m.identifier)).map(|vm| {
            div()
                .flex()
                .items_center()
                .gap_1()
                .px_2()
                .py_0p5()
                .rounded_md()
                .bg(theme.bg_sub)
                .child(VendorIcon::new(vm.vendor.clone()).size(crate::tokens::icon::MD).icon_url(vm.icon_url.clone()))
                .child(div().text_sm().text_color(theme.text_muted).child(vm.name.clone()))
        });

        let mut settings_content = div().flex().flex_col().gap_2().p_4().child(
            div().pb_1().text_sm().font_weight(FontWeight::MEDIUM).text_color(theme.text).child("Select a model"),
        );
        if models.is_empty() {
            settings_content = settings_content.child(if loading {
                div()
                    .py_6()
                    .flex()
                    .justify_center()
                    .child(Loader::new().label("Loading voice models…"))
                    .into_any_element()
            } else {
                div()
                    .py_6()
                    .text_sm()
                    .text_color(theme.text_muted)
                    .child("No voice models available.")
                    .into_any_element()
            });
        } else {
            for vm in &models {
                let selected = selected_id.as_deref() == Some(vm.id.as_str());
                settings_content = settings_content.child(self.model_card(cx, vm, selected));
            }
        }

        let history_content = if self.history.is_empty() {
            div().p_4().text_sm().text_color(theme.text_muted).child("No generations yet.").into_any_element()
        } else {
            let mut rows = div().flex().flex_col().gap_1().p_2();
            for entry in &self.history {
                rows = rows.child(self.history_row(cx, entry));
            }
            rows.into_any_element()
        };

        let right_body = match self.right_tab {
            RightPaneTab::Settings => settings_content.into_any_element(),
            RightPaneTab::History => history_content,
        };

        let status = if self.generating {
            Some(("Generating…".to_string(), theme.text_muted))
        } else {
            self.error.as_ref().map(|e| (e.clone(), theme.error))
        };

        div()
            .size_full()
            .flex()
            .flex_col()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .h(px(56.))
                    .px_5()
                    .border_b_1()
                    .border_color(theme.border)
                    .child(IconEl::new(Icon::Speech, theme.text).size(crate::tokens::icon::XL))
                    .child(
                        div().text_lg().font_weight(FontWeight::MEDIUM).text_color(theme.text).child("Text to Speech"),
                    )
                    .children(header_badge),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_row()
                    .child(
                        div()
                            .flex_1()
                            .min_h_0()
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .id("tts-editor")
                                    .flex_1()
                                    .min_h_0()
                                    .overflow_y_scroll()
                                    .p_5()
                                    .text_color(theme.text)
                                    .child(self.input.clone()),
                            )
                            .children(
                                status.map(|(text, color)| div().px_5().pb_1().text_sm().text_color(color).child(text)),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .gap_3()
                                    .h(px(60.))
                                    .px_5()
                                    .border_t_1()
                                    .border_color(theme.border)
                                    .child(
                                        div()
                                            .text_sm()
                                            .text_color(if over {
                                                theme.error
                                            } else {
                                                theme.text_muted
                                            })
                                            .child(format!("{char_count} / {CHAR_LIMIT} characters")),
                                    )
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .gap_2()
                                            .child(
                                                Button::new("tts-upload", "Upload text file")
                                                    .kind(ButtonKind::Secondary)
                                                    .icon(Icon::Download)
                                                    .on_click(cx.listener(|this, _, _, cx| this.pick_text_file(cx))),
                                            )
                                            .child(if self.generating {
                                                Button::new("tts-stop", "Stop")
                                                    .kind(ButtonKind::Danger)
                                                    .on_click(cx.listener(|this, _, _, cx| this.stop(cx)))
                                            } else {
                                                Button::new("tts-generate", "Generate speech")
                                                    .kind(ButtonKind::Primary)
                                                    .disabled(!any_installed || char_count == 0 || over)
                                                    .on_click(
                                                        cx.listener(|this, _, _, cx| this.generate_from_button(cx)),
                                                    )
                                            }),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .w(px(360.))
                            .flex_none()
                            .min_h_0()
                            .flex()
                            .flex_col()
                            .border_l_1()
                            .border_color(theme.border)
                            .child(
                                div()
                                    .flex()
                                    .border_b_1()
                                    .border_color(theme.border)
                                    .child(self.tab_button(cx, "Settings", RightPaneTab::Settings))
                                    .child(self.tab_button(cx, "History", RightPaneTab::History)),
                            )
                            .child(
                                div().id("tts-right-scroll").flex_1().min_h_0().overflow_y_scroll().child(right_body),
                            ),
                    ),
            )
    }
}
