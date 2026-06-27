//! The application shell: a left navigation sidebar, a content outlet that
//! swaps on the active `Route`, and a bottom footer bar. Navigation is just a
//! `Route` field on the root view, switched via `cx.listener` + `cx.notify()`.

use gpui::{
    AnyElement, Context, CursorStyle, Entity, IntoElement, Render, SharedString, Window, div,
    prelude::*, px,
};

use crate::{
    components::{Icon, IconEl, Toggle},
    models_store::{ModelKind, ModelsStore},
    persistence, settings_state, toast,
    screens::{
        ChatView, ChatEvent, ChatsEvent, ChatsView, CloudEvent, CloudModelsView, LocalModelsEvent,
        LocalModelsView, RoutersView, SettingsView, TtsView, WelcomeEvent, WelcomeView,
    },
    theme::{
        self, ActiveTheme, FONT_SANS, TEXT_SIZE,
        layout::{FOOTER_HEIGHT, SIDEBAR_WIDTH},
    },
};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Which top-level screen is showing. `Chat(None)` is a fresh chat.
#[derive(Clone)]
pub enum Route {
    Welcome,
    Chat(Option<String>),
    Chats,
    LocalModels,
    CloudModels,
    Routers,
    Tts,
    Settings,
}

/// Coarse grouping used for sidebar active-state highlighting.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Welcome,
    Chat,
    Chats,
    LocalModels,
    CloudModels,
    Routers,
    Tts,
    Settings,
}

impl Route {
    fn section(&self) -> Section {
        match self {
            Route::Welcome => Section::Welcome,
            Route::Chat(_) => Section::Chat,
            Route::Chats => Section::Chats,
            Route::LocalModels => Section::LocalModels,
            Route::CloudModels => Section::CloudModels,
            Route::Routers => Section::Routers,
            Route::Tts => Section::Tts,
            Route::Settings => Section::Settings,
        }
    }
}

/// Root view.
pub struct MiraiApp {
    route: Route,
    #[allow(dead_code)] // used by the footer / chat model selector in later steps
    models: Entity<ModelsStore>,
    local_models: Entity<LocalModelsView>,
    chat: Entity<ChatView>,
    chats: Entity<ChatsView>,
    welcome: Entity<WelcomeView>,
    settings: Entity<SettingsView>,
    routers: Entity<RoutersView>,
    tts: Entity<TtsView>,
    cloud: Entity<CloudModelsView>,
    /// Cached recent chats for the sidebar list; refreshed on navigation.
    recent_chats: Vec<persistence::StoredChat>,
    /// Whether the bottom "Settings" menu is expanded (mirai-chat parity).
    settings_menu_open: bool,
}

impl MiraiApp {
    pub fn new(cx: &mut Context<Self>) -> Self {
        // Apply the persisted color scheme (theme::init defaults to dark).
        if !persistence::load_settings().dark_mode {
            theme::set_theme(cx, theme::Theme::light());
        }
        // Repaint the whole tree when the theme is swapped (Settings toggle).
        theme::observe_theme(cx, |_, cx| cx.notify()).detach();
        // Repaint when toasts change.
        toast::observe(cx, |_, cx| cx.notify()).detach();

        let models = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud_store = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let local_models = cx.new(|cx| LocalModelsView::new(models.clone(), cx));
        let chat = cx.new(|cx| ChatView::new(models.clone(), cloud_store.clone(), cx));

        // Tapping an installed local model starts a chat with it.
        cx.subscribe(&local_models, |this, _view, event, cx| match event {
            LocalModelsEvent::UseModel(model) => {
                this.chat
                    .update(cx, |chat, cx| chat.use_model(model.clone(), cx));
                this.route = Route::Chat(None);
                cx.notify();
            }
        })
        .detach();
        let chats = cx.new(ChatsView::new);
        let welcome = cx.new(WelcomeView::new);
        let settings = cx.new(SettingsView::new);
        let routers_store = cx.new(|cx| ModelsStore::new(ModelKind::Classification, cx));
        let routers = cx.new(|cx| RoutersView::new(routers_store, cx));
        let tts_store = cx.new(|cx| ModelsStore::new(ModelKind::TextToSpeech, cx));
        let tts = cx.new(|cx| TtsView::new(tts_store, cx));
        let cloud = cx.new(|cx| CloudModelsView::new(cloud_store, cx));

        // Chatting with a cloud model picked on the Cloud Models screen.
        cx.subscribe(&cloud, |this, _cloud, event, cx| match event {
            CloudEvent::UseModel(model) => {
                this.chat
                    .update(cx, |chat, cx| chat.use_model(model.clone(), cx));
                this.route = Route::Chat(None);
                cx.notify();
            }
        })
        .detach();

        // Opening a saved chat from the history screen.
        cx.subscribe(&chats, |this, _chats, event, cx| match event {
            ChatsEvent::Open(id) => this.open_chat(id.clone(), cx),
        })
        .detach();

        cx.subscribe(&chat, |this, _chat, event, cx| match event {
            ChatEvent::Updated => {
                this.recent_chats = persistence::list_chats();
                cx.notify();
            }
            ChatEvent::OpenLocalModels => this.navigate(Route::LocalModels, cx),
        })
        .detach();

        // Dismissing the welcome screen.
        cx.subscribe(&welcome, |this, _welcome, event, cx| match event {
            WelcomeEvent::Continue => {
                persistence::set_seen_welcome();
                this.navigate(Route::LocalModels, cx);
            }
        })
        .detach();

        // First run shows onboarding; afterwards land on Local Models (matching
        // Electron mirai-chat's launch redirect). `MIRAI_SCREEN` overrides the
        // landing route for visual-QA/screenshot automation.
        let route = match std::env::var("MIRAI_SCREEN").ok().as_deref() {
            Some("chat") => Route::Chat(None),
            Some("chats") => Route::Chats,
            Some("local") => Route::LocalModels,
            Some("cloud") => Route::CloudModels,
            Some("routers") => Route::Routers,
            Some("tts") => Route::Tts,
            Some("settings") => Route::Settings,
            Some("welcome") => Route::Welcome,
            _ if persistence::has_seen_welcome() => Route::LocalModels,
            _ => Route::Welcome,
        };

        Self {
            route,
            models,
            local_models,
            chat,
            chats,
            welcome,
            settings,
            routers,
            tts,
            cloud,
            recent_chats: persistence::list_chats(),
            settings_menu_open: false,
        }
    }

    #[cfg(test)]
    pub fn open_settings_menu(&mut self, cx: &mut Context<Self>) {
        self.settings_menu_open = true;
        cx.notify();
    }

    /// Flip the color scheme and persist it (used by the settings menu).
    fn toggle_dark_mode(&mut self, cx: &mut Context<Self>) {
        let mut settings = settings_state::current(cx);
        settings.dark_mode = !settings.dark_mode;
        let next =
            if settings.dark_mode { theme::Theme::dark() } else { theme::Theme::light() };
        theme::set_theme(cx, next);
        settings_state::set(cx, settings);
        cx.notify();
    }

    /// Navigate to a route, running any side effects (reset/reload).
    fn navigate(&mut self, route: Route, cx: &mut Context<Self>) {
        match &route {
            Route::Chat(None) => self.chat.update(cx, |chat, cx| chat.start_new(cx)),
            Route::Chats => self.chats.update(cx, |chats, cx| chats.reload(cx)),
            _ => {}
        }
        self.recent_chats = persistence::list_chats();
        self.route = route;
        cx.notify();
    }

    fn open_chat(&mut self, id: String, cx: &mut Context<Self>) {
        if let Some(stored) = persistence::load_chat(&id) {
            self.chat.update(cx, |chat, cx| chat.load_stored(stored, cx));
            self.recent_chats = persistence::list_chats();
            self.route = Route::Chat(Some(id));
            cx.notify();
        }
    }

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
        // Labels/icons are always white; only the active row gets a highlight bg.
        let fg = theme.text;
        let bg = if active {
            theme.bg_hover
        } else {
            gpui::transparent_black()
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
            .child(IconEl::new(icon, fg).size(crate::tokens::icon::LG))
            .child(label)
            .on_click(cx.listener(move |this, _event, _window, cx| {
                this.navigate(target.clone(), cx);
            }))
    }

    /// Disabled "Apps" nav row with a trailing "Soon" tag (mirai-chat parity).
    fn apps_soon_item(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
                    .child(IconEl::new(Icon::Apps, theme.text).size(crate::tokens::icon::LG))
                    .child("Apps"),
            )
            .child(div().text_xs().text_color(theme.text_muted).child("Soon"))
    }

    fn render_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
            // Empty header spacer to clear the macOS traffic lights at y≈18
            // (the Electron app shows no logo/name in the sidebar).
            .child(div().h(px(52.)))
            // Primary navigation.
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
                    .child(self.nav_item(
                        cx,
                        "nav-tts",
                        Icon::Speech,
                        "Text to Speech",
                        Route::Tts,
                        section == Section::Tts,
                    ))
                    .child(self.apps_soon_item(cx)),
            )
            // Recent chats fill the space between nav and the pinned Settings.
            .child(self.render_recent_chats(cx))
            // Settings menu pinned to the bottom; expands upward (mirai-chat parity).
            .child(self.render_settings_menu(cx))
    }

    /// Bottom "Settings" row that expands an inline menu upward: external links,
    /// a Settings entry, and a dark-mode toggle, mirroring mirai-chat.
    fn render_settings_menu(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let open = self.settings_menu_open;
        let dark = settings_state::current(cx).dark_mode;
        let hover = theme.bg_hover;

        let mut wrap = div().flex().flex_col().border_t_1().border_color(theme.border);

        // Trigger at the top of the group; content expands below it.
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
                        .child(IconEl::new(Icon::Settings, theme.text).size(crate::tokens::icon::LG))
                        .child("Settings"),
                )
                .child(
                    IconEl::new(
                        if open { Icon::ChevronUp } else { Icon::ChevronDown },
                        theme.text_muted,
                    )
                    .size(crate::tokens::icon::SM),
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
                        .child(IconEl::new(icon, muted).size(crate::tokens::icon::LG)),
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
                        .child(Toggle::new("dark-mode", dark).on_click(cx.listener(
                            |this, _, _, cx| this.toggle_dark_mode(cx),
                        ))),
                );
        }

        wrap
    }

    /// Scrollable list of recent chats in the sidebar (mirai-chat parity).
    fn render_recent_chats(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover_bg = theme.bg_hover;
        let active = match &self.route {
            Route::Chat(Some(id)) => Some(id.clone()),
            _ => None,
        };

        let mut col = div()
            .id("recent-chats")
            .flex_1()
            .min_h_0()
            .flex()
            .flex_col()
            .overflow_y_scroll()
            .px_2()
            .pt_2();

        if self.recent_chats.is_empty() {
            return col
                .child(div().px_2().text_xs().text_color(theme.text_muted).child("No chats yet"))
                .into_any_element();
        }

        col = col.child(div().px_2().pb_1().text_xs().text_color(theme.text_muted).child("Recent"));
        for chat in &self.recent_chats {
            let id = chat.id.clone();
            let is_active = active.as_deref() == Some(chat.id.as_str());
            let fg = if is_active { theme.text } else { theme.text_muted };
            let bg = if is_active { theme.bg_hover } else { gpui::transparent_black() };
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
                    .child(truncate_title(&chat.title)),
            );
        }
        col.into_any_element()
    }

    fn render_content(&self, _cx: &mut Context<Self>) -> AnyElement {
        match &self.route {
            Route::LocalModels => self.local_models.clone().into_any_element(),
            Route::CloudModels => self.cloud.clone().into_any_element(),
            Route::Chat(_) => self.chat.clone().into_any_element(),
            Route::Chats => self.chats.clone().into_any_element(),
            Route::Routers => self.routers.clone().into_any_element(),
            Route::Tts => self.tts.clone().into_any_element(),
            Route::Settings => self.settings.clone().into_any_element(),
            // Welcome is rendered full-screen in `render`, never here.
            Route::Welcome => div().into_any_element(),
        }
    }

    fn render_footer(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let hover_bg = theme.bg_hover;
        let loaded = self.chat.read(cx).loaded_model_name();

        let left = match loaded {
            Some(name) => div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().text_color(theme.text).child(name))
                .child(
                    div()
                        .id("footer-eject")
                        .flex()
                        .items_center()
                        .gap_1()
                        .px_1()
                        .rounded_md()
                        .cursor(CursorStyle::PointingHand)
                        .hover(move |s| s.bg(hover_bg))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.chat.update(cx, |chat, cx| chat.eject(cx));
                        }))
                        .child(IconEl::new(Icon::Eject, theme.text_muted).size(11.))
                        .child("Eject"),
                )
                .into_any_element(),
            None => div().child("No model loaded").into_any_element(),
        };

        div()
            .h(px(FOOTER_HEIGHT))
            .flex_none()
            .w_full()
            .flex()
            .items_center()
            .justify_between()
            .px_3()
            .bg(theme.bg_sidebar)
            .border_t_1()
            .border_color(theme.border)
            .text_color(theme.text_muted)
            .text_size(crate::tokens::font::CAPTION)
            .child(left)
            .child(div().child(format!("v{APP_VERSION}")))
    }
}

fn truncate_title(s: &str) -> String {
    let s = s.trim();
    if s.chars().count() <= 26 {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(26).collect::<String>())
    }
}

impl Render for MiraiApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let root = div()
            .size_full()
            .relative()
            .bg(theme.bg)
            .text_color(theme.text)
            .font_family(FONT_SANS)
            .text_size(px(TEXT_SIZE));

        // Welcome is full-screen (no sidebar/footer).
        if matches!(self.route, Route::Welcome) {
            return root
                .child(self.welcome.clone())
                .children(toast::render_overlay(cx));
        }

        root.flex()
            .flex_col()
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_1()
                    .min_h_0()
                    .overflow_hidden()
                    .child(self.render_sidebar(cx))
                    .child(
                        div()
                            .flex_1()
                            .min_h_0()
                            .overflow_hidden()
                            .child(self.render_content(cx)),
                    ),
            )
            .child(self.render_footer(cx))
            .children(toast::render_overlay(cx))
    }
}
