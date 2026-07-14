use std::time::Duration;

use gpui::{
    Animation, AnimationExt, Context, EventEmitter, FontWeight, Hsla, IntoElement, Render, Window, div, prelude::*, px,
};

use crate::{
    components::{Button, ButtonKind, Icon, IconEl},
    settings_state,
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

fn feature(
    icon: Icon,
    label: &'static str,
    fg: Hsla,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .gap_2()
        .text_color(fg)
        .text_sm()
        .child(IconEl::new(icon, fg).size(crate::tokens::icon::MD))
        .child(label)
}

impl Render for WelcomeView {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let share = settings_state::current(cx).share_usage_data;

        let mut checkbox = div()
            .size(px(18.))
            .flex_none()
            .rounded(crate::tokens::radius::SM)
            .border_1()
            .border_color(theme.border)
            .flex()
            .items_center()
            .justify_center();
        if share {
            checkbox = checkbox.bg(theme.text).child(IconEl::new(Icon::Check, theme.bg).size(crate::tokens::icon::XS));
        }

        let content = div()
            .flex()
            .flex_col()
            .items_center()
            .gap_8()
            .child(IconEl::new(Icon::Logo, theme.text).size(crate::tokens::icon::HERO))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .text_size(crate::tokens::font::DISPLAY)
                            .font_weight(FontWeight::MEDIUM)
                            .child("Welcome to Mirai"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.text_muted)
                            .child("Try the full private and local AI potential of your Mac"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_6()
                    .child(feature(Icon::Lock, "100% Private", theme.text_muted))
                    .child(feature(Icon::Lightning, "Blazing Fast", theme.text_muted))
                    .child(feature(Icon::WifiOff, "Works Offline", theme.text_muted)),
            )
            .child(
                Button::new("welcome-continue", "Continue without account")
                    .kind(ButtonKind::Primary)
                    .on_click(cx.listener(|_, _, _, cx| cx.emit(WelcomeEvent::Continue))),
            )
            .child(
                div()
                    .id("welcome-analytics")
                    .flex()
                    .items_center()
                    .gap_2()
                    .cursor(gpui::CursorStyle::PointingHand)
                    .on_click(cx.listener(|_, _, _, cx| {
                        let mut s = settings_state::current(cx);
                        s.share_usage_data = !s.share_usage_data;
                        settings_state::set(cx, s);
                        cx.notify();
                    }))
                    .child(checkbox)
                    .child(div().text_xs().text_color(theme.text_muted).child("Share anonymous usage data")),
            );

        let content = if cfg!(test) {
            content.into_any_element()
        } else {
            content
                .with_animation("welcome-in", Animation::new(Duration::from_millis(500)), |el, delta| {
                    el.opacity(1.0 - (1.0 - delta).powi(3))
                })
                .into_any_element()
        };

        div().size_full().flex().items_center().justify_center().bg(theme.bg).text_color(theme.text).child(content)
    }
}
