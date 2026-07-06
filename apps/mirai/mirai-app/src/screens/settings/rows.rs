use gpui::{AnyElement, Context, CursorStyle, FontWeight, IntoElement, div, prelude::*, px, transparent_black};

use super::{setting_kind::SettingKind, settings_tab::SettingsTab, view::SettingsView};
use crate::{
    components::{Button, ButtonKind, ButtonSize, Icon, IconEl, Toggle},
    theme::ActiveTheme,
    tokens,
};

const DISCORD_URL: &str = "https://discord.com/invite/gUhyn6Rb7x";

impl SettingsView {
    pub(super) fn action_button(
        &self,
        cx: &mut Context<Self>,
        id: &'static str,
        icon: Icon,
        label: &'static str,
        on_click: impl Fn(&mut Self, &mut Context<Self>) + 'static,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        div()
            .id(id)
            .flex()
            .items_center()
            .gap_1()
            .h(px(28.))
            .px_3()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.card)
            .text_xs()
            .text_color(theme.text)
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| on_click(this, cx)))
            .child(IconEl::new(icon, theme.text_muted).size(13.))
            .child(label)
            .into_any_element()
    }

    pub(super) fn toggle_row(
        &self,
        cx: &mut Context<Self>,
        title: &'static str,
        desc: &'static str,
        on: bool,
        kind: SettingKind,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let view = cx.entity();
        div()
            .flex()
            .items_center()
            .justify_between()
            .py_3()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .child(div().text_sm().font_weight(FontWeight::MEDIUM).text_color(theme.text).child(title))
                    .child(div().text_xs().text_color(theme.text_muted).child(desc)),
            )
            .child(Toggle::new(title, on).on_click(move |_, _, cx| {
                view.update(cx, |this, cx| this.flip(kind, cx));
            }))
    }

    pub(super) fn action_row(
        &self,
        cx: &mut Context<Self>,
        title: &'static str,
        desc: &'static str,
        right: AnyElement,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        div()
            .flex()
            .items_center()
            .justify_between()
            .py_3()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .child(div().text_sm().font_weight(FontWeight::MEDIUM).text_color(theme.text).child(title))
                    .child(div().text_xs().text_color(theme.text_muted).child(desc)),
            )
            .child(right)
            .into_any_element()
    }

    pub(super) fn idle_timeout_row(
        &self,
        cx: &mut Context<Self>,
        minutes: u32,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        let border = theme.border;
        let hover = theme.bg_hover;
        let stepper = |id: &'static str, label: &'static str| {
            div()
                .id(id)
                .flex()
                .items_center()
                .justify_center()
                .size(px(24.))
                .text_color(theme.text)
                .cursor(CursorStyle::PointingHand)
                .hover(move |s| s.bg(hover))
                .child(label)
        };
        div()
            .flex()
            .items_center()
            .justify_between()
            .pb_3()
            .child(div().text_sm().text_color(theme.text_muted).child("Idle timeout (minutes)"))
            .child(
                div()
                    .flex()
                    .items_center()
                    .rounded_md()
                    .border_1()
                    .border_color(border)
                    .bg(theme.bg)
                    .child(stepper("idle-dec", "−").on_click(cx.listener(|this, _, _, cx| this.bump_idle(-1, cx))))
                    .child(div().w(px(36.)).text_center().text_sm().text_color(theme.text).child(format!("{minutes}")))
                    .child(stepper("idle-inc", "+").on_click(cx.listener(|this, _, _, cx| this.bump_idle(1, cx)))),
            )
            .into_any_element()
    }

    pub(super) fn nav_item(
        &self,
        cx: &mut Context<Self>,
        label: &'static str,
        tab: SettingsTab,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        let active = self.tab == tab;
        let hover = theme.bg_hover;
        div()
            .id(label)
            .flex()
            .items_center()
            .h(px(32.))
            .px_2()
            .rounded_md()
            .text_sm()
            .text_color(if active {
                theme.text
            } else {
                theme.text_muted
            })
            .bg(if active {
                theme.bg_hover
            } else {
                transparent_black()
            })
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.tab = tab;
                cx.notify();
            }))
            .child(label)
            .into_any_element()
    }

    pub(super) fn divider(
        &self,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        div().h(px(1.)).w_full().bg(cx.theme().border).into_any_element()
    }

    pub(super) fn feedback_footer(
        &self,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        div()
            .w_full()
            .flex()
            .items_center()
            .justify_between()
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(IconEl::new(Icon::Heart, theme.info).size(tokens::icon::MD))
                    .child(
                        div()
                            .min_w_0()
                            .text_xs()
                            .text_color(theme.text_muted)
                            .child("Let us know your feedback or request a new feature"),
                    ),
            )
            .child(
                div().flex_shrink_0().child(
                    Button::new("give-feedback", "Give Feedback")
                        .kind(ButtonKind::Primary)
                        .size(ButtonSize::Small)
                        .on_click(|_, _, cx| cx.open_url(DISCORD_URL)),
                ),
            )
            .into_any_element()
    }
}
