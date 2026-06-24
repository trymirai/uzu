//! Welcome / onboarding screen shown on first launch.

use gpui::{
    Context, EventEmitter, FontWeight, Hsla, IntoElement, Render, Window, div, prelude::*, px,
};

use crate::{
    components::{Button, ButtonKind, Icon, IconEl},
    theme::ActiveTheme,
};

pub enum WelcomeEvent {
    Continue,
}

pub struct WelcomeView;

impl EventEmitter<WelcomeEvent> for WelcomeView {}

impl WelcomeView {
    pub fn new(_cx: &mut Context<Self>) -> Self {
        Self
    }
}

fn pill(icon: Icon, label: &'static str, fg: Hsla, bg: Hsla) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_2()
        .h(px(32.))
        .px_3()
        .rounded_full()
        .bg(bg)
        .text_color(fg)
        .text_sm()
        .child(IconEl::new(icon, fg).size(15.))
        .child(label)
}

impl Render for WelcomeView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_8()
            .bg(theme.bg)
            .text_color(theme.text)
            .child(IconEl::new(Icon::Logo, theme.text).size(64.))
            .child(
                div()
                    .text_size(px(32.))
                    .font_weight(FontWeight::MEDIUM)
                    .child("Welcome to Mirai"),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(pill(Icon::Lock, "100% Private", theme.text_muted, theme.bg_sub))
                    .child(pill(
                        Icon::Lightning,
                        "Blazing Fast Responses",
                        theme.text_muted,
                        theme.bg_sub,
                    ))
                    .child(pill(Icon::WifiOff, "Works Offline", theme.text_muted, theme.bg_sub)),
            )
            .child(
                Button::new("welcome-continue", "Continue without account")
                    .kind(ButtonKind::Primary)
                    .on_click(cx.listener(|_, _, _, cx| cx.emit(WelcomeEvent::Continue))),
            )
    }
}
