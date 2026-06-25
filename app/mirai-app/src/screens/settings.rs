//! Settings screen: an inner sidebar (General / Privacy / About Mirai) with a
//! scrollable content panel, mirroring mirai-chat. The OS-integration rows
//! (run-on-startup, menu bar, global shortcut) persist their preference; the
//! native hooks behind them are tracked separately.

use std::collections::HashSet;

use gpui::{
    AnyElement, Context, CursorStyle, Entity, FontWeight, IntoElement, Render, SharedString, Window,
    div, prelude::*, px,
};

use crate::{
    components::{Button, ButtonKind, ButtonSize, Icon, IconEl, InputEvent, TextInput, Toggle},
    data_ops::{self, CleanupCategory, CleanupPreview},
    native_dialog,
    persistence, settings_state,
    theme::{ActiveTheme, Theme},
};

const DISCORD_URL: &str = "https://discord.com/invite/gUhyn6Rb7x";

#[derive(Clone, Copy)]
enum SettingKind {
    Reasoning,
    RunOnStartup,
    ShowInMenuBar,
    AutoEject,
    ShareUsage,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    General,
    Privacy,
    About,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClearDataStep {
    Select,
    Confirm,
    Result,
}

pub struct SettingsView {
    instructions: Entity<TextInput>,
    instructions_open: bool,
    tab: SettingsTab,
    clear_data_open: bool,
    clear_data_step: ClearDataStep,
    clear_data_selected: HashSet<CleanupCategory>,
    clear_data_preview: CleanupPreview,
    clear_data_results: Vec<(CleanupCategory, bool)>,
    clear_data_busy: bool,
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
            clear_data_open: false,
            clear_data_step: ClearDataStep::Select,
            clear_data_selected: CleanupCategory::ALL.into_iter().collect(),
            clear_data_preview: CleanupPreview::default(),
            clear_data_results: Vec::new(),
            clear_data_busy: false,
        }
    }

    fn open_clear_data(&mut self, cx: &mut Context<Self>) {
        self.clear_data_open = true;
        self.clear_data_step = ClearDataStep::Select;
        self.clear_data_results.clear();
        self.clear_data_busy = false;
        self.clear_data_preview = data_ops::cleanup_preview_disk();
        self.clear_data_selected = CleanupCategory::ALL.into_iter().collect();
        if self.clear_data_preview.dialogs.count == 0 {
            self.clear_data_selected.remove(&CleanupCategory::Dialogs);
        }
        if self.clear_data_preview.files.count == 0 {
            self.clear_data_selected.remove(&CleanupCategory::Files);
        }
        if self.clear_data_preview.logs_size_bytes == 0 {
            self.clear_data_selected.remove(&CleanupCategory::Logs);
        }
        cx.notify();

        if let Some(engine) = data_ops::try_engine_from_app(cx) {
            let view = cx.entity();
            cx.spawn(async move |_, cx| {
                let models = data_ops::model_cleanup_stats(&engine).await;
                view.update(cx, |this, cx| {
                    if !this.clear_data_open {
                        return;
                    }
                    this.clear_data_preview.models = models;
                    if this.clear_data_preview.models.count == 0 {
                        this.clear_data_selected.remove(&CleanupCategory::Models);
                    }
                    cx.notify();
                });
            })
            .detach();
        }
    }

    fn close_clear_data(&mut self, cx: &mut Context<Self>) {
        self.clear_data_open = false;
        self.clear_data_busy = false;
        cx.notify();
    }

    fn toggle_clear_category(&mut self, cat: CleanupCategory, cx: &mut Context<Self>) {
        if self.clear_data_selected.contains(&cat) {
            self.clear_data_selected.remove(&cat);
        } else {
            self.clear_data_selected.insert(cat);
        }
        cx.notify();
    }

    fn run_clear_data(&mut self, cx: &mut Context<Self>) {
        let selected: Vec<CleanupCategory> = CleanupCategory::ALL
            .into_iter()
            .filter(|c| self.clear_data_selected.contains(c))
            .collect();
        if selected.is_empty() {
            return;
        }
        self.clear_data_busy = true;
        cx.notify();

        let engine = data_ops::try_engine_from_app(cx);
        let view = cx.entity();
        cx.spawn(async move |_, cx| {
            let results = data_ops::execute_cleanup(engine.as_ref(), &selected).await;
            view.update(cx, |this, cx| {
                this.clear_data_busy = false;
                this.clear_data_results = results;
                this.clear_data_step = ClearDataStep::Result;
                cx.notify();
            });
        })
        .detach();
    }

    fn export_chats(&mut self, cx: &mut Context<Self>) {
        let Some(data) = data_ops::export_chats_zip() else {
            crate::toast::push(cx, "No chats to export", crate::toast::ToastKind::Info);
            return;
        };
        let default_name = data_ops::export_zip_default_name();
        let Some(path) = native_dialog::save_file("Export chats", &default_name, "zip", "ZIP archive")
        else {
            return;
        };
        if native_dialog::write_bytes(&path, &data) {
            crate::toast::push(cx, "Chats exported", crate::toast::ToastKind::Info);
        } else {
            crate::toast::push(cx, "Export failed", crate::toast::ToastKind::Error);
        }
    }

    fn export_logs(&mut self, cx: &mut Context<Self>) {
        let Some(data) = data_ops::read_log_bytes() else {
            crate::toast::push(cx, "No log file found", crate::toast::ToastKind::Info);
            return;
        };
        let Some(path) =
            native_dialog::save_file("Export logs", "mirai.log", "log", "Log file")
        else {
            return;
        };
        if native_dialog::write_bytes(&path, &data) {
            crate::toast::push(cx, "Logs exported", crate::toast::ToastKind::Info);
        } else {
            crate::toast::push(cx, "Export failed", crate::toast::ToastKind::Error);
        }
    }

    fn action_button(
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
            SettingKind::RunOnStartup => settings.run_on_startup = !settings.run_on_startup,
            SettingKind::ShowInMenuBar => settings.show_in_menu_bar = !settings.show_in_menu_bar,
            SettingKind::AutoEject => settings.auto_eject = !settings.auto_eject,
            SettingKind::ShareUsage => settings.share_usage_data = !settings.share_usage_data,
        }
        settings_state::set(cx, settings);
    }

    fn bump_idle(&mut self, delta: i32, cx: &mut Context<Self>) {
        let mut settings = settings_state::current(cx);
        let next = settings.idle_timeout_minutes as i32 + delta;
        settings.idle_timeout_minutes = next.clamp(1, 120) as u32;
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

    /// A non-toggle row: title + description on the left, a control on the right.
    fn action_row(
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
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child(title),
                    )
                    .child(div().text_xs().text_color(theme.text_muted).child(desc)),
            )
            .child(right)
            .into_any_element()
    }

    /// Indented "Idle timeout (minutes)" stepper, shown under Auto-eject.
    fn idle_timeout_row(&self, cx: &mut Context<Self>, minutes: u32) -> AnyElement {
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
            .child(
                div()
                    .text_sm()
                    .text_color(theme.text_muted)
                    .child("Idle timeout (minutes)"),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .rounded_md()
                    .border_1()
                    .border_color(border)
                    .bg(theme.bg)
                    .child(
                        stepper("idle-dec", "−")
                            .on_click(cx.listener(|this, _, _, cx| this.bump_idle(-1, cx))),
                    )
                    .child(
                        div()
                            .w(px(36.))
                            .text_center()
                            .text_sm()
                            .text_color(theme.text)
                            .child(format!("{minutes}")),
                    )
                    .child(
                        stepper("idle-inc", "+")
                            .on_click(cx.listener(|this, _, _, cx| this.bump_idle(1, cx))),
                    ),
            )
            .into_any_element()
    }

    /// Select a tab by index (0 General, 1 Privacy, 2 About) — used by visual tests.
    pub fn select_tab(&mut self, index: usize, cx: &mut Context<Self>) {
        self.tab = match index {
            1 => SettingsTab::Privacy,
            2 => SettingsTab::About,
            _ => SettingsTab::General,
        };
        cx.notify();
    }

    /// One inner-sidebar nav item.
    fn nav_item(&self, cx: &mut Context<Self>, label: &'static str, tab: SettingsTab) -> AnyElement {
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
            .text_color(if active { theme.text } else { theme.text_muted })
            .bg(if active { theme.bg_hover } else { gpui::transparent_black() })
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.tab = tab;
                cx.notify();
            }))
            .child(label)
            .into_any_element()
    }

    fn divider(&self, cx: &mut Context<Self>) -> AnyElement {
        div().h(px(1.)).w_full().bg(cx.theme().border).into_any_element()
    }

    /// Feedback row shown at the foot of General / Privacy.
    fn feedback_footer(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme().clone();
        div()
            .flex()
            .items_center()
            .justify_between()
            .pt_4()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(IconEl::new(Icon::Heart, theme.info).size(16.))
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.text_muted)
                            .child("Let us know your feedback or request a new feature"),
                    ),
            )
            .child(
                Button::new("give-feedback", "Give Feedback")
                    .kind(ButtonKind::Secondary)
                    .size(ButtonSize::Small)
                    .on_click(|_, _, cx| cx.open_url(DISCORD_URL)),
            )
            .into_any_element()
    }

    fn general_content(&self, cx: &mut Context<Self>) -> AnyElement {
        let settings = settings_state::current(cx);
        let theme = cx.theme().clone();
        let hover = theme.bg_hover;

        // "Set shortcut" control. Global-shortcut capture isn't wired yet, so it
        // explains itself when clicked rather than silently doing nothing.
        let set_shortcut = div()
            .id("set-shortcut")
            .flex()
            .items_center()
            .justify_center()
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
            .on_click(cx.listener(|_, _, _, cx| {
                crate::toast::push(
                    cx,
                    "Global shortcut capture is coming soon",
                    crate::toast::ToastKind::Info,
                );
            }))
            .child("Set shortcut")
            .into_any_element();

        let mut col = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(self.instructions_card(cx))
            .child(self.divider(cx))
            .child(self.toggle_row(
                cx,
                "Run on startup",
                "Automatically start Mirai when you log in to your computer",
                settings.run_on_startup,
                SettingKind::RunOnStartup,
            ))
            .child(self.toggle_row(
                cx,
                "Show in the menu bar",
                "Mirai will be always available in your menu bar",
                settings.show_in_menu_bar,
                SettingKind::ShowInMenuBar,
            ))
            .child(self.action_row(
                cx,
                "Quick entry keyboard shortcut",
                "Open Mirai from anywhere with a shortcut",
                set_shortcut,
            ))
            .child(self.toggle_row(
                cx,
                "Default reasoning mode",
                "Let reasoning models think by default. You can override this for each model.",
                settings.reasoning,
                SettingKind::Reasoning,
            ))
            .child(self.toggle_row(
                cx,
                "Auto-eject models when idle",
                "Automatically unload local models after inactivity",
                settings.auto_eject,
                SettingKind::AutoEject,
            ));
        if settings.auto_eject {
            col = col.child(self.idle_timeout_row(cx, settings.idle_timeout_minutes));
        }
        col.child(self.feedback_footer(cx)).into_any_element()
    }

    fn privacy_content(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme().clone();
        let settings = settings_state::current(cx);
        let export_chats = self.action_button(cx, "export-chats", Icon::Download, "Export", |this, cx| {
            this.export_chats(cx);
        });
        let export_logs = self.action_button(cx, "export-logs", Icon::Download, "Export", |this, cx| {
            this.export_logs(cx);
        });
        let clear_data = self.action_button(cx, "clear-data", Icon::Trash, "Clear data", |this, cx| {
            this.open_clear_data(cx);
        });
        div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .pb_2()
                    .child(IconEl::new(Icon::Shield, theme.info).size(22.))
                    .child(
                        div()
                            .text_lg()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.text)
                            .child("All data is processed and stored locally on your device."),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.text_muted)
                            .child("Your privacy is built into the core of how Mirai runs."),
                    ),
            )
            .child(self.divider(cx))
            .child(legal_row(
                "terms",
                "Terms of Service",
                "https://artifacts.trymirai.com/legal/Mirai_Tech_Terms_of_Use.pdf",
                &theme,
            ))
            .child(legal_row(
                "privacy-policy",
                "Privacy Policy",
                "https://artifacts.trymirai.com/legal/Mirai_Tech_Privacy_Policy.pdf",
                &theme,
            ))
            .child(self.action_row(
                cx,
                "Export all your chats",
                "As Markdown files in .zip archive",
                export_chats,
            ))
            .child(self.action_row(
                cx,
                "Export logs",
                "Save the current Mirai log file for debugging",
                export_logs,
            ))
            .child(self.action_row(
                cx,
                "Clear data",
                "Delete dialogs, generated audio, models, or logs",
                clear_data,
            ))
            .child(self.toggle_row(
                cx,
                "Share anonymous usage data",
                "Help us improve Mirai by sharing anonymous usage data.",
                settings.share_usage_data,
                SettingKind::ShareUsage,
            ))
            .child(self.feedback_footer(cx))
            .into_any_element()
    }

    fn about_content(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme().clone();
        let text = theme.text;
        let muted = theme.text_muted;
        let hover = theme.bg_hover;
        let border = theme.border;

        // One cell of the horizontal Website | GitHub | Vision | Docs band.
        let cell = |id: &'static str, label: &'static str, url: &'static str| {
            div()
                .id(id)
                .flex_1()
                .flex()
                .items_center()
                .justify_between()
                .h(px(52.))
                .px_4()
                .cursor(CursorStyle::PointingHand)
                .hover(move |s| s.bg(hover))
                .on_click(move |_, _, cx| cx.open_url(url))
                .child(div().text_sm().text_color(text).child(label))
                .child(IconEl::new(Icon::ChevronRight, muted).size(14.))
        };
        let vsep = || div().w(px(1.)).h(px(24.)).bg(border);

        // Header + links band, grouped in one rounded card.
        let header_card = div()
            .rounded_lg()
            .border_1()
            .border_color(border)
            .overflow_hidden()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .p_4()
                    .child(IconEl::new(Icon::Heart, theme.info).size(22.))
                    .child(
                        div()
                            .text_lg()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(text)
                            .child(
                                "Done by a team who share a vision for accessible and powerful \
                                 local AI.",
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .border_t_1()
                    .border_color(border)
                    .child(cell("about-website", "Website", "https://trymirai.com/"))
                    .child(vsep())
                    .child(cell("about-github", "GitHub", "https://github.com/trymirai"))
                    .child(vsep())
                    .child(cell("about-vision", "Vision", "https://trymirai.com/about-us"))
                    .child(vsep())
                    .child(cell("about-docs", "Docs", "https://docs.trymirai.com/")),
            );

        div()
            .flex()
            .flex_col()
            .gap_3()
            .child(header_card)
            .child(
                div()
                    .pt_6()
                    .text_xs()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(muted)
                    .child("Our products"),
            )
            .child(product_row(
                "prod-platform",
                "Mirai platform",
                "a web console where you can set up the SDK for your product",
                "https://platform.trymirai.com",
                &theme,
            ))
            .child(self.divider(cx))
            .child(product_row(
                "prod-cli",
                "Command-line interface",
                "that allows you to interactively send messages to a model or start a local server",
                "https://docs.trymirai.com/overview/cli",
                &theme,
            ))
            .child(self.divider(cx))
            .child(product_row(
                "prod-engine",
                "Rust inference engine",
                "designed to run models on specific hardware",
                "https://github.com/trymirai/uzu",
                &theme,
            ))
            .into_any_element()
    }

    fn clear_data_modal(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        if !self.clear_data_open {
            return None;
        }
        let theme = cx.theme().clone();
        let preview = self.clear_data_preview.clone();
        let selected = self.clear_data_selected.clone();
        let busy = self.clear_data_busy;
        let step = self.clear_data_step;

        let checkbox_row = |cat: CleanupCategory| {
            let checked = selected.contains(&cat);
            let desc = data_ops::category_description(cat, &preview);
            let box_color = if checked { theme.info } else { theme.border };
            div()
                .id(SharedString::from(format!("clear-cat-{:?}", cat)))
                .flex()
                .items_start()
                .gap_3()
                .pb_2()
                .cursor(CursorStyle::PointingHand)
                .on_click(cx.listener(move |this, _, _, cx| this.toggle_clear_category(cat, cx)))
                .child(
                    div()
                        .mt(px(2.))
                        .size(px(14.))
                        .rounded_sm()
                        .border_1()
                        .border_color(box_color)
                        .bg(if checked { theme.info.opacity(0.15) } else { gpui::transparent_black() })
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_xs()
                        .text_color(theme.info)
                        .child(if checked { "✓" } else { "" }),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_0p5()
                        .child(
                            div()
                                .text_sm()
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(theme.text)
                                .child(cat.label()),
                        )
                        .child(div().text_xs().text_color(theme.text_muted).child(desc)),
                )
        };

        let (title, body, footer): (SharedString, AnyElement, AnyElement) = match step {
            ClearDataStep::Select => {
                let selected_count = CleanupCategory::ALL
                    .iter()
                    .filter(|c| selected.contains(c))
                    .count();
                let review = Button::new("clear-review", "Review")
                    .kind(ButtonKind::Danger)
                    .size(ButtonSize::Small)
                    .disabled(selected_count == 0 || busy)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.clear_data_step = ClearDataStep::Confirm;
                        cx.notify();
                    }));
                let cancel = Button::new("clear-cancel", "Cancel")
                    .kind(ButtonKind::Secondary)
                    .size(ButtonSize::Small)
                    .on_click(cx.listener(|this, _, _, cx| this.close_clear_data(cx)));
                (
                    "Clear data".into(),
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.text_muted)
                                .child("Select the data categories you want to permanently delete."),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .children(CleanupCategory::ALL.map(checkbox_row)),
                        )
                        .into_any_element(),
                    div().flex().justify_end().gap_2().child(cancel).child(review).into_any_element(),
                )
            }
            ClearDataStep::Confirm => {
                let cats: Vec<CleanupCategory> = CleanupCategory::ALL
                    .into_iter()
                    .filter(|c| selected.contains(c))
                    .collect();
                let summary = confirm_summary(&cats);
                let delete = Button::new("clear-delete", "Delete")
                    .kind(ButtonKind::Danger)
                    .size(ButtonSize::Small)
                    .disabled(busy)
                    .on_click(cx.listener(|this, _, _, cx| this.run_clear_data(cx)));
                let back = Button::new("clear-back", "Back")
                    .kind(ButtonKind::Secondary)
                    .size(ButtonSize::Small)
                    .disabled(busy)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.clear_data_step = ClearDataStep::Select;
                        cx.notify();
                    }));
                let cancel = Button::new("clear-cancel", "Cancel")
                    .kind(ButtonKind::Secondary)
                    .size(ButtonSize::Small)
                    .disabled(busy)
                    .on_click(cx.listener(|this, _, _, cx| this.close_clear_data(cx)));
                (
                    "Are you sure?".into(),
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.text_muted)
                                .child(format!(
                                    "This will permanently delete: {summary}. This action cannot be undone."
                                )),
                        )
                        .into_any_element(),
                    div()
                        .flex()
                        .justify_end()
                        .gap_2()
                        .child(cancel)
                        .child(back)
                        .child(delete)
                        .into_any_element(),
                )
            }
            ClearDataStep::Result => {
                let close = Button::new("clear-close", "Close")
                    .kind(ButtonKind::Primary)
                    .size(ButtonSize::Small)
                    .on_click(cx.listener(|this, _, _, cx| this.close_clear_data(cx)));
                let mut lines = div().flex().flex_col().gap_2();
                for cat in CleanupCategory::ALL.iter().filter(|c| selected.contains(c)) {
                    let done = self
                        .clear_data_results
                        .iter()
                        .find(|(c, _)| c == cat)
                        .is_some_and(|(_, ok)| *ok);
                    let mark = if done { "✓" } else { "✗" };
                    let status = if done { "cleared" } else { "failed" };
                    lines = lines.child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(div().text_sm().child(mark))
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(if done { theme.text } else { theme.text_muted })
                                    .child(format!("{} {status}", cat.label())),
                            ),
                    );
                }
                (
                    "Done".into(),
                    lines.into_any_element(),
                    div().flex().justify_end().gap_2().child(close).into_any_element(),
                )
            }
        };

        Some(
            div()
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .bg(gpui::black().opacity(0.5))
                .occlude()
                .child(
                    div()
                        .w(px(400.))
                        .flex()
                        .flex_col()
                        .gap_3()
                        .p_4()
                        .rounded_xl()
                        .bg(theme.card)
                        .border_1()
                        .border_color(theme.border)
                        .child(
                            div()
                                .text_color(theme.text)
                                .font_weight(FontWeight::MEDIUM)
                                .child(title),
                        )
                        .child(body)
                        .child(footer),
                )
                .into_any_element(),
        )
    }
}

impl Render for SettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
            // Title bar spanning the full width, above the sidebar + content.
            .child(
                div()
                    .w_full()
                    .flex_none()
                    .pt_10()
                    .pb_3()
                    .px_6()
                    .border_b_1()
                    .border_color(theme.border)
                    .child(
                        div()
                            .text_size(px(16.))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text)
                            .child("Settings"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h_0()
                    .child(nav)
                    .child(
                        div()
                            .id("settings-content")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .px_6()
                            .py_4()
                            .child(content),
                    ),
            );
        if let Some(modal) = modal {
            root = root.child(modal);
        }
        root
    }
}

fn confirm_summary(categories: &[CleanupCategory]) -> String {
    let labels: Vec<&str> = categories.iter().map(|c| c.label()).collect();
    match labels.len() {
        0 => String::new(),
        1 => labels[0].to_string(),
        2 => format!("{} and {}", labels[0], labels[1]),
        _ => format!(
            "{}, and {}",
            labels[..labels.len() - 1].join(", "),
            labels[labels.len() - 1]
        ),
    }
}

/// A clickable text link that opens `url` in the browser.
/// A full-width legal/document row: label on the left, chevron on the right,
/// opening `url` on click (Privacy tab's Terms / Privacy entries).
fn legal_row(id: &'static str, label: &'static str, url: &'static str, theme: &Theme) -> impl IntoElement {
    let hover = theme.bg_hover;
    let text = theme.text;
    let muted = theme.text_muted;
    div()
        .id(id)
        .flex()
        .items_center()
        .justify_between()
        .py_3()
        .px_2()
        .rounded_md()
        .cursor(gpui::CursorStyle::PointingHand)
        .hover(move |s| s.bg(hover))
        .on_click(move |_, _, cx| cx.open_url(url))
        .child(div().text_sm().text_color(text).child(label))
        .child(IconEl::new(Icon::ChevronRight, muted).size(16.))
}

/// A product row (title + description + chevron) linking to `url`.
fn product_row(
    id: &'static str,
    title: &'static str,
    desc: &'static str,
    url: &'static str,
    theme: &Theme,
) -> impl IntoElement {
    let hover = theme.bg_hover;
    let text = theme.text;
    let muted = theme.text_muted;
    div()
        .id(id)
        .flex()
        .items_center()
        .justify_between()
        .py_2()
        .px_2()
        .rounded_md()
        .cursor(gpui::CursorStyle::PointingHand)
        .hover(move |s| s.bg(hover))
        .on_click(move |_, _, cx| cx.open_url(url))
        .child(
            div()
                .flex()
                .flex_col()
                .child(div().text_sm().text_color(text).child(title))
                .child(div().text_xs().text_color(muted).child(desc)),
        )
        .child(IconEl::new(Icon::ChevronRight, muted).size(14.))
}
