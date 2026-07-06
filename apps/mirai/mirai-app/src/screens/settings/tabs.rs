use gpui::{AnyElement, Context, CursorStyle, FontWeight, IntoElement, div, prelude::*, px};

use super::{setting_kind::SettingKind, view::SettingsView};
use crate::{
    components::{Icon, IconEl},
    settings_state,
    theme::{ActiveTheme, Theme},
    toast::{self, ToastKind},
    tokens,
};

impl SettingsView {
    pub(super) fn general_content(
        &self,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let settings = settings_state::current(cx);
        let theme = cx.theme().clone();

        let shortcut_hover = theme.text.opacity(0.85);
        let set_shortcut = div()
            .id("set-shortcut")
            .flex()
            .items_center()
            .justify_center()
            .h(px(28.))
            .px_3()
            .rounded_md()
            .bg(theme.text)
            .text_xs()
            .font_weight(FontWeight::MEDIUM)
            .text_color(theme.bg)
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(shortcut_hover))
            .on_click(cx.listener(|_, _, _, cx| {
                toast::push(cx, "Global shortcut capture is coming soon", ToastKind::Info);
            }))
            .child("Set shortcut")
            .into_any_element();

        let mut col = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(self.instructions.clone())
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
        col.into_any_element()
    }

    pub(super) fn privacy_content(
        &self,
        cx: &mut Context<Self>,
    ) -> AnyElement {
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
                    .child(IconEl::new(Icon::Shield, theme.info).size(tokens::icon::XXL))
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
            .child(self.action_row(cx, "Export all your chats", "As Markdown files in .zip archive", export_chats))
            .child(self.action_row(cx, "Export logs", "Save the current Mirai log file for debugging", export_logs))
            .child(self.action_row(cx, "Clear data", "Delete dialogs, generated audio, models, or logs", clear_data))
            .child(self.toggle_row(
                cx,
                "Share anonymous usage data",
                "Help us improve Mirai by sharing anonymous usage data.",
                settings.share_usage_data,
                SettingKind::ShareUsage,
            ))
            .into_any_element()
    }

    pub(super) fn about_content(
        &self,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme().clone();
        let text = theme.text;
        let muted = theme.text_muted;
        let hover = theme.bg_hover;
        let border = theme.border;

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
                .child(IconEl::new(Icon::ChevronRight, muted).size(tokens::icon::SM))
        };
        let vsep = || div().w(px(1.)).h(px(24.)).bg(border);

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
                    .child(IconEl::new(Icon::Heart, theme.info).size(tokens::icon::XXL))
                    .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).text_color(text).child(
                        "Done by a team who share a vision for accessible and powerful \
                                 local AI.",
                    )),
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
            .child(div().pt_6().text_xs().font_weight(FontWeight::MEDIUM).text_color(muted).child("Our products"))
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
}

fn legal_row(
    id: &'static str,
    label: &'static str,
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
        .py_3()
        .rounded_md()
        .cursor(CursorStyle::PointingHand)
        .hover(move |s| s.bg(hover))
        .on_click(move |_, _, cx| cx.open_url(url))
        .child(div().text_sm().text_color(text).child(label))
        .child(IconEl::new(Icon::ChevronRight, muted).size(tokens::icon::MD))
}

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
        .cursor(CursorStyle::PointingHand)
        .hover(move |s| s.bg(hover))
        .on_click(move |_, _, cx| cx.open_url(url))
        .child(
            div()
                .flex()
                .flex_col()
                .child(div().text_sm().text_color(text).child(title))
                .child(div().text_xs().text_color(muted).child(desc)),
        )
        .child(IconEl::new(Icon::ChevronRight, muted).size(tokens::icon::SM))
}
