use std::time::Duration;

use gpui::{Context, Entity, FontWeight, IntoElement, Render, Window, div, prelude::*, px};
use uzu::{
    player::Player,
    types::{
        basic::{CancelToken, PcmBatch},
        model::Model,
    },
};

use super::{pending_generation::PendingGeneration, right_pane_tab::RightPaneTab, vm::TtsVm};
use crate::{
    components::{Button, ButtonKind, Icon, IconEl, InputEvent, Loader, TextInput, VendorIcon},
    models_store::ModelsStore,
    theme::ActiveTheme,
    tokens,
    tts_history::{self, TtsHistoryEntry},
};

pub(super) const CHAR_LIMIT: usize = 2000;

pub struct TtsView {
    pub(super) store: Entity<ModelsStore>,
    pub(super) input: Entity<TextInput>,
    pub(super) selected: Option<Model>,
    pub(super) player: Option<Player>,
    pub(super) generating: bool,
    pub(super) generation_id: u64,
    pub(super) cancel: Option<CancelToken>,
    pub(super) error: Option<String>,
    pub(super) history: Vec<TtsHistoryEntry>,
    pub(super) playing_id: Option<String>,
    pub(super) pending_batches: Vec<PcmBatch>,
    pub(super) pending_gen: Option<PendingGeneration>,
    pub(super) right_tab: RightPaneTab,
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
                cx.background_executor().timer(Duration::from_millis(250)).await;
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
                .child(VendorIcon::new(vm.vendor.clone()).size(tokens::icon::MD).icon_url(vm.icon_url.clone()))
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
                    .child(IconEl::new(Icon::Speech, theme.text).size(tokens::icon::XL))
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
