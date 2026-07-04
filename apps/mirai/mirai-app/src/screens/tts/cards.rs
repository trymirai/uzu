use gpui::{
    Context, CursorStyle, FontWeight, IntoElement, SharedString, div, prelude::*, px, relative, transparent_black,
};

use super::{right_pane_tab::RightPaneTab, view::TtsView, vm::TtsVm};
use crate::{
    components::{Icon, IconButton, IconEl, VendorIcon},
    text::truncate_with_ellipsis,
    theme::ActiveTheme,
    tokens,
    tts_history::TtsHistoryEntry,
};

impl TtsView {
    pub(super) fn model_card(
        &self,
        cx: &mut Context<Self>,
        vm: &TtsVm,
        selected: bool,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let id = vm.id.clone();

        let control = if vm.installed {
            let id = id.clone();
            IconButton::new(SharedString::from(format!("del-{}", vm.id)), Icon::Trash)
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
            IconButton::new(SharedString::from(format!("dl-{}", vm.id)), Icon::Download)
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
                .text_size(tokens::font::CAPTION)
                .text_color(theme.success)
                .child(IconEl::new(Icon::Check, theme.success).size(11.))
                .child("Installed")
                .into_any_element()
        } else if vm.downloading {
            div()
                .text_size(tokens::font::CAPTION)
                .text_color(theme.text_muted)
                .child(format!("Downloading {:.0}%", vm.progress * 100.0))
                .into_any_element()
        } else {
            div()
                .text_size(tokens::font::CAPTION)
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
            .id(SharedString::from(vm.id.clone()))
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
                    .child(VendorIcon::new(vm.vendor.clone()).size(tokens::icon::XXL).icon_url(vm.icon_url.clone()))
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

    pub(super) fn history_row(
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

    pub(super) fn tab_button(
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
                transparent_black()
            })
            .cursor(CursorStyle::PointingHand)
            .on_click(cx.listener(move |this, _, _, cx| {
                this.right_tab = tab;
                cx.notify();
            }))
            .child(label)
    }
}
