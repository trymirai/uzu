//! Settings screen: tabbed (General / Privacy / About). Connectors and
//! auto-eject are later refinements.

use gpui::{
    AnyElement, Context, CursorStyle, Entity, FontWeight, IntoElement, Render, Window, div,
    prelude::*, px,
};

use crate::{
    components::{Icon, IconEl, InputEvent, SegmentedControl, TextInput, Toggle},
    persistence, settings_state,
    theme::{ActiveTheme, Theme, layout::CONTENT_MAX_WIDTH},
};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy)]
enum SettingKind {
    Reasoning,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    General,
    Privacy,
    About,
}

pub struct SettingsView {
    instructions: Entity<TextInput>,
    instructions_open: bool,
    tab: SettingsTab,
}

impl SettingsView {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let instructions = cx
            .new(|cx| TextInput::new(cx, "Instructions applied to every chat…").multiline(false, 3, 6));
        let current = persistence::global_instructions();
        if !current.is_empty() {
            instructions.update(cx, |input, cx| input.set_text(current, cx));
        }
        cx.subscribe(&instructions, |_this, _input, event, _cx| match event {
            InputEvent::Submit(text) | InputEvent::Changed(text) => {
                persistence::set_global_instructions(text)
            }
        })
        .detach();
        settings_state::observe(cx, |_, cx| cx.notify()).detach();
        Self {
            instructions,
            instructions_open: false,
            tab: SettingsTab::General,
        }
    }

    /// Expandable "Add instructions to all chats" card (mirai-chat parity),
    /// shared in spirit with the Chats screen.
    fn instructions_card(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;
        let open = self.instructions_open;

        let header = div()
            .id("settings-instr-card")
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .py_3()
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(|this, _, _, cx| {
                this.instructions_open = !this.instructions_open;
                cx.notify();
            }))
            .child(IconEl::new(if open { Icon::Close } else { Icon::Plus }, theme.text).size(16.))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child("Add instructions to all chats"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.text_muted)
                            .child("Tailor the way the model responds"),
                    ),
            );

        let mut card = div()
            .w_full()
            .rounded_lg()
            .border_1()
            .border_color(theme.border)
            .bg(theme.card)
            .child(header);

        if open {
            card = card.child(
                div().px_4().pb_3().child(
                    div()
                        .w_full()
                        .px_3()
                        .py_2()
                        .rounded_md()
                        .border_1()
                        .border_color(theme.border)
                        .bg(theme.bg)
                        .child(self.instructions.clone()),
                ),
            );
        }
        card.into_any_element()
    }

    fn flip(&mut self, kind: SettingKind, cx: &mut Context<Self>) {
        let mut settings = settings_state::current(cx);
        match kind {
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

        let general = div().flex().flex_col().child(self.toggle_row(
            cx,
            "Default reasoning mode",
            "Let reasoning models think by default. You can override this for each model.",
            settings.reasoning,
            SettingKind::Reasoning,
        ));

        let content = match self.tab {
            SettingsTab::General => div()
                .flex()
                .flex_col()
                .gap_2()
                .pt_6()
                .child(self.instructions_card(cx))
                .child(general)
                .into_any_element(),
            SettingsTab::Privacy => {
                let body = div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(div().text_sm().text_color(theme.text).child(
                        "All data is processed and stored locally on your device.",
                    ))
                    .child(div().text_xs().text_color(theme.text_muted).child(
                        "Models run on-device — your conversations never leave your machine.",
                    ));
                self.section(cx, "Privacy", body).into_any_element()
            }
            SettingsTab::About => {
                let body = div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(div().text_sm().text_color(theme.text).child("Mirai"))
                    .child(div().text_xs().text_color(theme.text_muted).child(format!(
                        "Version {APP_VERSION}"
                    )))
                    .child(link_row("about-uzu", "uzu engine on GitHub", "https://github.com/trymirai/uzu", &theme));
                self.section(cx, "About", body).into_any_element()
            }
        };

        let tabs = SegmentedControl::new("settings-tabs", self.tab as usize)
            .segment(
                "General",
                cx.listener(|this, _, _, cx| {
                    this.tab = SettingsTab::General;
                    cx.notify();
                }),
            )
            .segment(
                "Privacy",
                cx.listener(|this, _, _, cx| {
                    this.tab = SettingsTab::Privacy;
                    cx.notify();
                }),
            )
            .segment(
                "About Mirai",
                cx.listener(|this, _, _, cx| {
                    this.tab = SettingsTab::About;
                    cx.notify();
                }),
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
                            .w_full()
                            .text_center()
                            .pt_10()
                            .pb_3()
                            .text_xl()
                            .font_weight(FontWeight::MEDIUM)
                            .child("Settings"),
                    )
                    .child(div().max_w(px(360.)).child(tabs))
                    .child(content),
            )
    }
}

/// A clickable About-tab row that opens `url` in the browser.
fn link_row(id: &'static str, label: &'static str, url: &'static str, theme: &Theme) -> impl IntoElement {
    let hover = theme.bg_hover;
    div()
        .id(id)
        .flex()
        .items_center()
        .py_1()
        .text_sm()
        .text_color(theme.info)
        .cursor(gpui::CursorStyle::PointingHand)
        .hover(move |s| s.bg(hover))
        .on_click(move |_, _, cx| cx.open_url(url))
        .child(label)
}
