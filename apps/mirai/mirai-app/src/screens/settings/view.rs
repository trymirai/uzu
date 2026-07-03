use std::collections::HashSet;

use gpui::{Context, Entity, EventEmitter, FontWeight, IntoElement, Render, Window, div, prelude::*, px};

use super::{clear_data::ClearDataStep, event::SettingsEvent, settings_tab::SettingsTab};
use crate::{
    data_ops::{CleanupCategory, CleanupPreview},
    instructions_card::InstructionsCard,
    settings_state,
    theme::ActiveTheme,
    tokens,
};

pub struct SettingsView {
    pub(super) instructions: Entity<InstructionsCard>,
    pub(super) tab: SettingsTab,
    pub(super) clear_data_open: bool,
    pub(super) clear_data_step: ClearDataStep,
    pub(super) clear_data_selected: HashSet<CleanupCategory>,
    pub(super) clear_data_preview: CleanupPreview,
    pub(super) clear_data_results: Vec<(CleanupCategory, bool)>,
    pub(super) clear_data_busy: bool,
}

impl EventEmitter<SettingsEvent> for SettingsView {}

impl SettingsView {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let instructions =
            cx.new(|cx| InstructionsCard::new(cx, "settings-instr-card", "Instructions applied to every chat…"));
        settings_state::observe(cx, |_, cx| cx.notify()).detach();
        Self {
            instructions,
            tab: SettingsTab::General,
            clear_data_open: false,
            clear_data_step: ClearDataStep::Select,
            clear_data_selected: CleanupCategory::ALL.into_iter().collect(),
            clear_data_preview: CleanupPreview::default(),
            clear_data_results: Vec::new(),
            clear_data_busy: false,
        }
    }
}

impl Render for SettingsView {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();

        let content = match self.tab {
            SettingsTab::General => self.general_content(cx),
            SettingsTab::Privacy => self.privacy_content(cx),
            SettingsTab::About => self.about_content(cx),
        };

        let nav = div()
            .w(px(160.))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .gap_1()
            .p_2()
            .border_r_1()
            .border_color(theme.border)
            .child(self.nav_item(cx, "General", SettingsTab::General))
            .child(self.nav_item(cx, "Privacy", SettingsTab::Privacy))
            .child(self.nav_item(cx, "About Mirai", SettingsTab::About));

        let modal = self.clear_data_modal(cx);

        let mut root = div()
            .size_full()
            .relative()
            .flex()
            .flex_col()
            .child(
                div().w_full().flex_none().pt_10().pb_3().px_6().border_b_1().border_color(theme.border).child(
                    div()
                        .text_size(tokens::font::LABEL)
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(theme.text)
                        .child("Settings"),
                ),
            )
            .child(
                div().flex().flex_1().min_h_0().child(nav).child(
                    div()
                        .flex_1()
                        .min_h_0()
                        .flex()
                        .flex_col()
                        .child(
                            div()
                                .id("settings-content")
                                .flex_1()
                                .min_h_0()
                                .overflow_y_scroll()
                                .px_6()
                                .py_4()
                                .child(content),
                        )
                        .when(matches!(self.tab, SettingsTab::General | SettingsTab::Privacy), |el| {
                            el.child(
                                div()
                                    .flex_none()
                                    .w_full()
                                    .border_t_1()
                                    .border_color(theme.border)
                                    .px_6()
                                    .py_3()
                                    .child(self.feedback_footer(cx)),
                            )
                        }),
                ),
            );
        if let Some(modal) = modal {
            root = root.child(modal);
        }
        root
    }
}
