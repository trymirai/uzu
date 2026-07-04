use gpui::{Context, CursorStyle, IntoElement, SharedString, div, prelude::*, px, transparent_black};

use super::{route::Route, section::Section, shell::MiraiApp};
use crate::{
    components::{Icon, IconEl, Toggle},
    engine_capabilities::{CLASSIFICATION, TEXT_TO_SPEECH},
    settings_state,
    text::truncate_with_ellipsis,
    theme::{ActiveTheme, layout::SIDEBAR_WIDTH},
    tokens,
};

impl MiraiApp {
    fn nav_item(
        &self,
        cx: &mut Context<Self>,
        id: &'static str,
        icon: Icon,
        label: &'static str,
        target: Route,
        active: bool,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let fg = theme.text;
        let bg = if active {
            theme.bg_hover
        } else {
            transparent_black()
        };
        let hover_bg = theme.bg_hover;

        div()
            .id(id)
            .flex()
            .items_center()
            .gap_2()
            .h(px(34.))
            .px_2()
            .mx_2()
            .rounded_md()
            .bg(bg)
            .text_color(fg)
            .text_sm()
            .cursor(CursorStyle::PointingHand)
            .hover(move |s| s.bg(hover_bg))
            .child(IconEl::new(icon, fg).size(tokens::icon::LG))
            .child(label)
            .on_click(cx.listener(move |this, _event, _window, cx| {
                this.navigate(target.clone(), cx);
            }))
    }

    fn apps_soon_item(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        div()
            .flex()
            .items_center()
            .justify_between()
            .h(px(34.))
            .px_2()
            .mx_2()
            .rounded_md()
            .text_color(theme.text)
            .text_sm()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(IconEl::new(Icon::Apps, theme.text).size(tokens::icon::LG))
                    .child("Apps"),
            )
            .child(div().text_xs().text_color(theme.text_muted).child("Soon"))
    }

    pub(super) fn render_sidebar(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let section = self.route.section();

        div()
            .w(px(SIDEBAR_WIDTH))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .bg(theme.bg_sidebar)
            .border_r_1()
            .border_color(theme.border)
            .child(div().h(px(52.)))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .py_1()
                    .child(self.nav_item(
                        cx,
                        "nav-new-chat",
                        Icon::Plus,
                        "New Chat",
                        Route::Chat(None),
                        section == Section::Chat,
                    ))
                    .child(self.nav_item(
                        cx,
                        "nav-chats",
                        Icon::Chats,
                        "Chats",
                        Route::Chats,
                        section == Section::Chats,
                    ))
                    .child(self.nav_item(
                        cx,
                        "nav-models",
                        Icon::Models,
                        "Local Models",
                        Route::LocalModels,
                        section == Section::LocalModels,
                    ))
                    .when(TEXT_TO_SPEECH, |column| {
                        column.child(self.nav_item(
                            cx,
                            "nav-tts",
                            Icon::Speech,
                            "Text to Speech",
                            Route::Tts,
                            section == Section::Tts,
                        ))
                    })
                    .when(CLASSIFICATION, |column| {
                        column.child(self.nav_item(
                            cx,
                            "nav-routers",
                            Icon::Routers,
                            "Routers",
                            Route::Routers,
                            section == Section::Routers,
                        ))
                    })
                    .child(self.apps_soon_item(cx)),
            )
            .child(self.render_recent_chats(cx))
            .child(self.render_settings_menu(cx))
    }

    fn render_settings_menu(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let open = self.settings_menu_open;
        let dark = settings_state::current(cx).dark_mode;
        let hover = theme.bg_hover;

        let mut wrap = div().flex().flex_col().border_t_1().border_color(theme.border);

        wrap = wrap.child(
            div()
                .id("settings-trigger")
                .flex()
                .items_center()
                .justify_between()
                .h(px(40.))
                .px_4()
                .text_sm()
                .text_color(theme.text)
                .cursor(CursorStyle::PointingHand)
                .hover(move |s| s.bg(hover))
                .on_click(cx.listener(|this, _, _, cx| {
                    this.settings_menu_open = !this.settings_menu_open;
                    cx.notify();
                }))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(IconEl::new(Icon::Settings, theme.text).size(tokens::icon::LG))
                        .child("Settings"),
                )
                .child(
                    IconEl::new(
                        if open {
                            Icon::ChevronUp
                        } else {
                            Icon::ChevronDown
                        },
                        theme.text_muted,
                    )
                    .size(tokens::icon::SM),
                ),
        );

        if open {
            let muted = theme.text_muted;
            let link_hover = theme.text;
            let mut links = div().flex().items_center().gap_4().px_4().pt_2().pb_2();
            for (id, icon, url) in [
                ("lnk-github", Icon::Github, "https://github.com/trymirai"),
                ("lnk-x", Icon::XSocial, "https://x.com/trymirai"),
                ("lnk-discord", Icon::Discord, "https://discord.gg/trymirai"),
            ] {
                let url = url.to_string();
                links = links.child(
                    div()
                        .id(id)
                        .flex()
                        .items_center()
                        .justify_center()
                        .size(px(28.))
                        .rounded_md()
                        .text_color(muted)
                        .cursor(CursorStyle::PointingHand)
                        .hover(move |s| s.text_color(link_hover).bg(hover))
                        .on_click(move |_, _, cx| cx.open_url(&url))
                        .child(IconEl::new(icon, muted).size(tokens::icon::LG)),
                );
            }
            wrap = wrap
                .child(links)
                .child(
                    div()
                        .id("settings-open")
                        .flex()
                        .items_center()
                        .h(px(34.))
                        .px_4()
                        .text_sm()
                        .text_color(theme.text)
                        .cursor(CursorStyle::PointingHand)
                        .hover(move |s| s.bg(hover))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.settings_menu_open = false;
                            this.navigate(Route::Settings, cx);
                        }))
                        .child("Settings"),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .h(px(38.))
                        .px_4()
                        .text_sm()
                        .text_color(theme.text)
                        .child("Dark mode")
                        .child(
                            Toggle::new("dark-mode", dark)
                                .on_click(cx.listener(|this, _, _, cx| this.toggle_dark_mode(cx))),
                        ),
                );
        }

        wrap
    }

    fn render_recent_chats(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover_bg = theme.bg_hover;
        let active = match &self.route {
            Route::Chat(Some(id)) => Some(id.clone()),
            _ => None,
        };

        let mut col = div().id("recent-chats").flex_1().min_h_0().flex().flex_col().overflow_y_scroll().px_2().pt_2();

        if self.recent_chats.is_empty() {
            return col
                .child(div().px_2().text_xs().text_color(theme.text_muted).child("No chats yet"))
                .into_any_element();
        }

        col = col.child(div().px_2().pb_1().text_xs().text_color(theme.text_muted).child("Recent"));
        for chat in &self.recent_chats {
            let id = chat.id.clone();
            let is_active = active.as_deref() == Some(chat.id.as_str());
            let fg = if is_active {
                theme.text
            } else {
                theme.text_muted
            };
            let bg = if is_active {
                theme.bg_hover
            } else {
                transparent_black()
            };
            col = col.child(
                div()
                    .id(SharedString::from(format!("recent-{}", chat.id)))
                    .h(px(30.))
                    .px_2()
                    .flex()
                    .items_center()
                    .rounded_md()
                    .bg(bg)
                    .text_sm()
                    .text_color(fg)
                    .cursor(CursorStyle::PointingHand)
                    .hover(move |s| s.bg(hover_bg))
                    .on_click(cx.listener(move |this, _, _, cx| this.open_chat(id.clone(), cx)))
                    .child(truncate_with_ellipsis(&chat.title, 26)),
            );
        }
        col.into_any_element()
    }
}
