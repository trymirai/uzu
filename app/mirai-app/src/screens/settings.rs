//! Settings screen: appearance + reasoning toggles, global instructions, about.
//! (Profile / Privacy / Connectors tabs and auto-eject are later refinements.)

use gpui::{Context, Entity, FontWeight, IntoElement, Render, Window, div, prelude::*, px};

use crate::{
    components::{InputEvent, TextInput, Toggle},
    persistence, settings_state,
    theme::{self, ActiveTheme, Theme, layout::CONTENT_MAX_WIDTH},
};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy)]
enum SettingKind {
    DarkMode,
    Reasoning,
}

pub struct SettingsView {
    instructions: Entity<TextInput>,
}

impl SettingsView {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let instructions =
            cx.new(|cx| TextInput::new(cx, "Instructions applied to every chat…"));
        let current = persistence::global_instructions();
        if !current.is_empty() {
            instructions.update(cx, |input, cx| input.set_text(current, cx));
        }
        cx.subscribe(&instructions, |_this, _input, event, _cx| match event {
            InputEvent::Submit(text) => persistence::set_global_instructions(text),
        })
        .detach();
        settings_state::observe(cx, |_, cx| cx.notify()).detach();
        Self { instructions }
    }

    fn flip(&mut self, kind: SettingKind, cx: &mut Context<Self>) {
        let mut settings = settings_state::current(cx);
        match kind {
            SettingKind::DarkMode => {
                settings.dark_mode = !settings.dark_mode;
                let next = if settings.dark_mode {
                    Theme::dark()
                } else {
                    Theme::light()
                };
                theme::set_theme(cx, next);
            }
            SettingKind::Reasoning => settings.reasoning = !settings.reasoning,
        }
        settings_state::set(cx, settings);
    }

    fn toggle_row(
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
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child(title),
                    )
                    .child(div().text_xs().text_color(theme.text_muted).child(desc)),
            )
            .child(Toggle::new(title, on).on_click(move |_, _, cx| {
                view.update(cx, |this, cx| this.flip(kind, cx));
            }))
    }

    fn section(
        &self,
        cx: &mut Context<Self>,
        title: &'static str,
        body: impl IntoElement,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        div()
            .flex()
            .flex_col()
            .pt_6()
            .child(
                div()
                    .pb_1()
                    .text_xs()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text_muted)
                    .child(title),
            )
            .child(body)
    }
}

impl Render for SettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let settings = settings_state::current(cx);

        let general = div()
            .flex()
            .flex_col()
            .child(self.toggle_row(
                cx,
                "Dark mode",
                "Use the dark color scheme",
                settings.dark_mode,
                SettingKind::DarkMode,
            ))
            .child(self.toggle_row(
                cx,
                "Show reasoning",
                "Display the model's chain-of-thought",
                settings.reasoning,
                SettingKind::Reasoning,
            ));

        let instructions_box = div()
            .flex()
            .items_center()
            .w_full()
            .px_3()
            .py_2()
            .rounded_lg()
            .border_1()
            .border_color(theme.border)
            .bg(theme.card)
            .child(self.instructions.clone());

        let about = div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_sm()
                    .text_color(theme.text)
                    .child("Mirai"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.text_muted)
                    .child(format!("Version {APP_VERSION}")),
            );

        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .child(
                div()
                    .id("settings-scroll")
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
                            .text_xl()
                            .font_weight(FontWeight::MEDIUM)
                            .child("Settings"),
                    )
                    .child(self.section(cx, "General", general))
                    .child(self.section(cx, "Global instructions", instructions_box))
                    .child(self.section(cx, "About", about)),
            )
    }
}
