//! Text-to-speech screen.

use futures::{StreamExt, channel::mpsc};
use gpui::{
    Context, CursorStyle, Entity, FontWeight, IntoElement, Render, SharedString, Window, div,
    prelude::*, px,
};
use uzu::{
    player::Player,
    session::text_to_speech::TextToSpeechSessionStreamChunk,
    storage::types::DownloadPhase,
    types::{
        basic::{CancelToken, PcmBatch},
        model::Model,
    },
};

use crate::{
    components::{
        Button, ButtonKind, ButtonSize, Icon, IconButton, IconEl, InputEvent, Loader, TextInput,
    },
    engine,
    models_store::ModelsStore,
    theme::{ActiveTheme, layout::CONTENT_MAX_WIDTH},
    tts_history::{self, TtsHistoryEntry},
};

enum TtsMsg {
    Started(CancelToken),
    Batch(PcmBatch),
    Done,
    Error(String),
}

struct TtsVm {
    id: String,
    name: String,
    vendor: String,
    installed: bool,
    downloading: bool,
    progress: f32,
}

struct PendingGen {
    text: String,
    model: Model,
    vendor: String,
}

pub struct TtsView {
    store: Entity<ModelsStore>,
    input: Entity<TextInput>,
    selected: Option<Model>,
    player: Option<Player>,
    generating: bool,
    cancel: Option<CancelToken>,
    error: Option<String>,
    history: Vec<TtsHistoryEntry>,
    playing_id: Option<String>,
    pending_batches: Vec<PcmBatch>,
    pending_gen: Option<PendingGen>,
}

impl TtsView {
    pub fn new(store: Entity<ModelsStore>, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| TextInput::new(cx, "Text to speak…"));
        cx.subscribe(&input, |this, _input, event, cx| match event {
            InputEvent::Submit(text) => this.generate(text.clone(), cx),
            InputEvent::Changed(_) => {}
        })
        .detach();
        cx.observe(&store, |_, _, cx| cx.notify()).detach();
        Self {
            store,
            input,
            selected: None,
            player: None,
            generating: false,
            cancel: None,
            error: None,
            history: tts_history::list(),
            playing_id: None,
            pending_batches: Vec::new(),
            pending_gen: None,
        }
    }

    fn clear_gen(&mut self) {
        self.pending_batches.clear();
        self.pending_gen = None;
    }

    fn append_pcm(&mut self, batch: PcmBatch) -> Result<(), String> {
        if self.player.is_none() {
            self.player = Some(Player::new().map_err(|e| format!("audio: {e}"))?);
        }
        self.player
            .as_ref()
            .expect("player just opened")
            .append_pcm_batch(batch)
            .map_err(|e| format!("audio: {e}"))
    }

    fn reload_history(&mut self) {
        self.history = tts_history::list();
    }

    fn resolved_model(&self, cx: &Context<Self>) -> Option<Model> {
        self.selected.clone().or_else(|| {
            self.store
                .read(cx)
                .rows
                .iter()
                .find(|r| r.is_installed())
                .map(|r| r.model.clone())
        })
    }

    fn generate(&mut self, text: String, cx: &mut Context<Self>) {
        if self.generating {
            return;
        }
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        let Some(model) = self.resolved_model(cx) else {
            self.error = Some("Download and select a voice model first.".to_string());
            cx.notify();
            return;
        };
        self.selected = Some(model.clone());
        self.generating = true;
        self.error = None;
        self.playing_id = None;
        self.pending_batches.clear();
        self.pending_gen = Some(PendingGen {
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

        let (tx, mut rx) = mpsc::unbounded::<TtsMsg>();
        gpui_tokio::Tokio::spawn(cx, async move {
            let session = match engine.text_to_speech(model).await {
                Ok(session) => session,
                Err(err) => {
                    let _ = tx.unbounded_send(TtsMsg::Error(err.to_string()));
                    return;
                }
            };
            let stream = session.synthesize_stream(text).await;
            let _ = tx.unbounded_send(TtsMsg::Started(stream.cancel_token()));
            while let Some(event) = stream.next().await {
                match event {
                    TextToSpeechSessionStreamChunk::Output { output } => {
                        let _ = tx.unbounded_send(TtsMsg::Batch(output.pcm_batch));
                    }
                    TextToSpeechSessionStreamChunk::Error { error } => {
                        let _ = tx.unbounded_send(TtsMsg::Error(format!("{error}")));
                    }
                }
            }
            let _ = tx.unbounded_send(TtsMsg::Done);
        })
        .detach();

        cx.spawn(async move |this, cx| {
            while let Some(msg) = rx.next().await {
                if this.update(cx, |view, cx| view.apply(msg, cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
    }

    fn apply(&mut self, msg: TtsMsg, cx: &mut Context<Self>) {
        match msg {
            TtsMsg::Started(token) => self.cancel = Some(token),
            TtsMsg::Batch(batch) => {
                self.pending_batches.push(batch.clone());
                if let Err(err) = self.append_pcm(batch) {
                    self.error = Some(err);
                    self.generating = false;
                }
            }
            TtsMsg::Done => {
                self.generating = false;
                self.cancel = None;
                if self.error.is_none() {
                    if let Some(pending) = self.pending_gen.as_ref() {
                        if tts_history::save_generation(
                            &pending.model,
                            &pending.vendor,
                            &pending.text,
                            &self.pending_batches,
                        )
                        .is_some()
                        {
                            self.reload_history();
                        }
                    }
                }
                self.clear_gen();
                cx.notify();
            }
            TtsMsg::Error(err) => {
                self.error = Some(err);
                self.generating = false;
                self.cancel = None;
                self.clear_gen();
                cx.notify();
            }
        }
    }

    fn play_history(&mut self, id: &str, cx: &mut Context<Self>) {
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

    fn stop_playback(&mut self, cx: &mut Context<Self>) {
        if let Some(player) = &self.player {
            player.stop();
        }
        self.playing_id = None;
        cx.notify();
    }

    fn delete_history(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.playing_id.as_deref() == Some(id) {
            self.stop_playback(cx);
        }
        tts_history::delete(id);
        self.reload_history();
        cx.notify();
    }

    fn restore_text(&mut self, text: &str, cx: &mut Context<Self>) {
        self.input.update(cx, |input, cx| input.set_text(text, cx));
        cx.notify();
    }

    fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        self.stop_playback(cx);
        self.generating = false;
        self.cancel = None;
        self.clear_gen();
        cx.notify();
    }

    fn generate_from_button(&mut self, cx: &mut Context<Self>) {
        let text = self.input.read(cx).text();
        self.generate(text, cx);
    }

    fn model_row(&self, cx: &mut Context<Self>, vm: &TtsVm, selected: bool) -> impl IntoElement {
        let theme = cx.theme().clone();
        let id = vm.id.clone();

        let control = if vm.installed {
            let id = id.clone();
            IconButton::new(gpui::SharedString::from(format!("del-{}", vm.id)), Icon::Trash)
                .color(theme.text_muted)
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
            Button::new(gpui::SharedString::from(format!("dl-{}", vm.id)), "Download")
                .kind(ButtonKind::Secondary)
                .size(ButtonSize::Small)
                .icon(Icon::Download)
                .on_click(cx.listener(move |this, _, _, cx| {
                    let id = id.clone();
                    this.store.update(cx, |s, cx| s.toggle_download(id, cx));
                }))
                .into_any_element()
        };

        let select_id = id.clone();
        let bg = if selected {
            theme.bg_hover
        } else {
            gpui::transparent_black()
        };
        div()
            .id(gpui::SharedString::from(vm.id.clone()))
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .h(px(52.))
            .px_3()
            .rounded_lg()
            .bg(bg)
            .when(vm.installed, |el| el.cursor(CursorStyle::PointingHand))
            .on_click(cx.listener(move |this, _, _, cx| {
                if let Some(model) = this
                    .store
                    .read(cx)
                    .rows
                    .iter()
                    .find(|r| r.id() == select_id && r.is_installed())
                    .map(|r| r.model.clone())
                {
                    this.selected = Some(model);
                    cx.notify();
                }
            }))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child(vm.name.clone()),
                    )
                    .child(div().text_xs().text_color(theme.text_muted).child(vm.vendor.clone())),
            )
            .child(control)
    }

    fn history_row(&self, cx: &mut Context<Self>, entry: &TtsHistoryEntry) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let playing = self.playing_id.as_deref() == Some(entry.id.as_str());
        let preview = truncate_line(&entry.text, 72);
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
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text)
                            .overflow_hidden()
                            .child(preview),
                    )
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
                            if playing { Icon::Stop } else { Icon::Speech },
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
}

fn truncate_line(text: &str, max: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    trimmed.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
}

impl Render for TtsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let selected_id = self.selected.as_ref().map(|m| m.identifier.clone());
        let resolved_name = self
            .resolved_model(cx)
            .map(|m| m.name())
            .unwrap_or_else(|| "No voice model".to_string());

        let (loading, models): (bool, Vec<TtsVm>) = {
            let store = self.store.read(cx);
            let rows = store
                .rows
                .iter()
                .map(|r| TtsVm {
                    id: r.id().to_string(),
                    name: r.name(),
                    vendor: r.vendor().unwrap_or_else(|| "Other".to_string()),
                    installed: r.is_installed(),
                    downloading: matches!(
                        r.phase(),
                        DownloadPhase::Downloading {} | DownloadPhase::Paused {}
                    ),
                    progress: r.progress(),
                })
                .collect();
            (store.loading, rows)
        };
        let any_installed = models.iter().any(|r| r.installed);

        let mut list = div().flex().flex_col().gap_1();
        if models.is_empty() {
            if loading {
                list = list.child(
                    div().py_6().flex().justify_center().child(Loader::new().label("Loading voice models…")),
                );
            } else {
                list = list.child(
                    div().py_6().text_color(theme.text_muted).child("No voice models available."),
                );
            }
        } else {
            for vm in &models {
                let selected = selected_id.as_deref() == Some(vm.id.as_str());
                list = list.child(self.model_row(cx, vm, selected));
            }
        }

        let status = if self.generating {
            Some(("Generating…".to_string(), theme.text_muted))
        } else {
            self.error.as_ref().map(|e| (e.clone(), theme.error))
        };

        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .child(
                div()
                    .id("tts-scroll")
                    .w_full()
                    .max_w(px(CONTENT_MAX_WIDTH))
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .overflow_y_scroll()
                    .px_6()
                    .child(
                        div()
                            .pt_10()
                            .pb_2()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(IconEl::new(Icon::Speech, theme.text).size(22.))
                            .child(
                                div()
                                    .text_xl()
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Text to Speech"),
                            ),
                    )
                    .child(list)
                    .child(
                        div()
                            .pt_6()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(theme.text_muted)
                                    .child(format!("Speak with {resolved_name}")),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .flex_1()
                                            .px_3()
                                            .py_2()
                                            .rounded_lg()
                                            .border_1()
                                            .border_color(theme.border)
                                            .bg(theme.card)
                                            .child(self.input.clone()),
                                    )
                                    .child(if self.generating {
                                        Button::new("tts-stop", "Stop")
                                            .kind(ButtonKind::Danger)
                                            .on_click(cx.listener(|this, _, _, cx| this.stop(cx)))
                                    } else {
                                        Button::new("tts-generate", "Generate")
                                            .kind(ButtonKind::Primary)
                                            .icon(Icon::Speech)
                                            .disabled(!any_installed)
                                            .on_click(cx.listener(|this, _, _, cx| {
                                                this.generate_from_button(cx)
                                            }))
                                    }),
                            )
                            .children(status.map(|(text, color)| {
                                div().text_sm().text_color(color).child(text)
                            })),
                    )
                    .child(
                        div()
                            .pt_8()
                            .pb_6()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(theme.text)
                                    .child("History"),
                            )
                            .child(if self.history.is_empty() {
                                div()
                                    .py_4()
                                    .text_sm()
                                    .text_color(theme.text_muted)
                                    .child("No generations yet.")
                                    .into_any_element()
                            } else {
                                let mut rows = div().flex().flex_col().gap_1();
                                for entry in &self.history {
                                    rows = rows.child(self.history_row(cx, entry));
                                }
                                rows.into_any_element()
                            }),
                    ),
            )
    }
}
