//! The application shell: a left navigation sidebar, a content outlet that
//! swaps on the active `Route`, and a bottom footer bar. Navigation is just a
//! `Route` field on the root view, switched via `cx.listener` + `cx.notify()`.

use gpui::{
    AnyElement, Context, CursorStyle, Entity, IntoElement, Render, Window, div, prelude::*, px,
};

use super::route::Route;
use crate::{
    components::{Icon, IconEl},
    models_store::{ModelKind, ModelsStore},
    persistence, settings_state, toast,
    screens::{
        ChatView, ChatEvent, ChatsEvent, ChatsView, CloudEvent, CloudModelsView, LocalModelsEvent,
        LocalModelsView, RoutersView, SettingsEvent, SettingsView, TtsView, WelcomeEvent,
        WelcomeView,
    },
    theme::{self, ActiveTheme, FONT_SANS, TEXT_SIZE, layout::FOOTER_HEIGHT},
};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Root view.
pub struct MiraiApp {
    pub(super) route: Route,
    local_models: Entity<LocalModelsView>,
    chat: Entity<ChatView>,
    chats: Entity<ChatsView>,
    welcome: Entity<WelcomeView>,
    settings: Entity<SettingsView>,
    routers: Entity<RoutersView>,
    tts: Entity<TtsView>,
    cloud: Entity<CloudModelsView>,
    /// Cached recent chats for the sidebar list; refreshed on navigation.
    pub(super) recent_chats: Vec<persistence::StoredChat>,
    /// Whether the bottom "Settings" menu is expanded (mirai-chat parity).
    pub(super) settings_menu_open: bool,
    /// The macOS menu-bar status item, present while "Show in menu bar" is on.
    /// `!Send`; lives on the main-thread entity. Dropping it removes the item.
    tray: Option<tray_icon::TrayIcon>,
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

        // Reconcile the menu-bar status item and drain its menu clicks.
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(250))
                    .await;
                if this.update(cx, |app, cx| app.tick_menu_bar(cx)).is_err() {
                    break;
                }
            }
        })
        .detach();

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

        // Clearing data deletes chats on disk; refresh the sidebar's cache.
        cx.subscribe(&settings, |this, _settings, event, cx| match event {
            SettingsEvent::DataCleared => {
                this.recent_chats = persistence::list_chats();
                cx.notify();
            }
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
            tray: None,
        }
    }

    /// Keep the menu-bar status item in sync with the "Show in menu bar" toggle
    /// and route any pending status-item menu clicks. Driven by a short timer
    /// (`tray-icon`'s `MenuEvent` channel isn't wired into GPUI's run loop).
    fn tick_menu_bar(&mut self, cx: &mut Context<Self>) {
        let want = settings_state::current(cx).show_in_menu_bar;
        if want && self.tray.is_none() {
            self.tray = crate::menu_bar::build();
        } else if !want && self.tray.is_some() {
            self.tray = None;
        }
        while let Ok(event) = tray_icon::menu::MenuEvent::receiver().try_recv() {
            match event.id.0.as_str() {
                crate::menu_bar::OPEN => cx.activate(true),
                crate::menu_bar::NEW_CHAT => {
                    self.navigate(Route::Chat(None), cx);
                    cx.activate(true);
                }
                crate::menu_bar::OPEN_CHATS => {
                    self.navigate(Route::Chats, cx);
                    cx.activate(true);
                }
                crate::menu_bar::SETTINGS => {
                    self.navigate(Route::Settings, cx);
                    cx.activate(true);
                }
                crate::menu_bar::QUIT => cx.quit(),
                _ => {}
            }
        }
    }

    #[cfg(test)]
    pub fn open_settings_menu(&mut self, cx: &mut Context<Self>) {
        self.settings_menu_open = true;
        cx.notify();
    }

    /// Flip the color scheme and persist it (used by the settings menu).
    pub(super) fn toggle_dark_mode(&mut self, cx: &mut Context<Self>) {
        let mut settings = settings_state::current(cx);
        settings.dark_mode = !settings.dark_mode;
        let next =
            if settings.dark_mode { theme::Theme::dark() } else { theme::Theme::light() };
        theme::set_theme(cx, next);
        settings_state::set(cx, settings);
        cx.notify();
    }

    /// Navigate to a route, running any side effects (reset/reload).
    pub(super) fn navigate(&mut self, route: Route, cx: &mut Context<Self>) {
        match &route {
            Route::Chat(None) => self.chat.update(cx, |chat, cx| chat.start_new(cx)),
            Route::Chats => self.chats.update(cx, |chats, cx| chats.reload(cx)),
            _ => {}
        }
        self.recent_chats = persistence::list_chats();
        self.route = route;
        cx.notify();
    }

    pub(super) fn open_chat(&mut self, id: String, cx: &mut Context<Self>) {
        if let Some(stored) = persistence::load_chat(&id) {
            self.chat.update(cx, |chat, cx| chat.load_stored(stored, cx));
            self.recent_chats = persistence::list_chats();
            self.route = Route::Chat(Some(id));
            cx.notify();
        }
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
