use gpui::{
    AnyElement, App, Bounds, Context, CursorStyle, Entity, IntoElement, Render, TitlebarOptions, Window, WindowBounds,
    WindowOptions, div, point, prelude::*, px, size,
};

use super::route::Route;
use crate::{
    components::{Icon, IconEl},
    menu_bar::TrayAction,
    models_store::{ModelKind, ModelsStore},
    persistence,
    screens::{
        ChatEvent, ChatView, ChatsEvent, ChatsView, CloudEvent, CloudModelsView, LocalModelsEvent, LocalModelsView,
        RoutersView, SettingsEvent, SettingsView, TtsView, WelcomeEvent, WelcomeView,
    },
    settings_state,
    theme::{self, ActiveTheme, FONT_SANS, TEXT_SIZE, layout::FOOTER_HEIGHT},
    toast,
};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

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

    pub(super) recent_chats: Vec<persistence::StoredChat>,

    pub(super) settings_menu_open: bool,
}

impl MiraiApp {
    pub fn new(cx: &mut Context<Self>) -> Self {
        if !settings_state::current(cx).dark_mode {
            theme::set_theme(cx, theme::Theme::light());
        }

        theme::observe_theme(cx, |_, cx| cx.notify()).detach();
        toast::observe(cx, |_, cx| cx.notify()).detach();

        let weak = cx.entity().downgrade();
        crate::menu_bar::register_app(cx, weak);

        let models = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud_store = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let local_models = cx.new(|cx| LocalModelsView::new(models.clone(), cx));
        let chat = cx.new(|cx| ChatView::new(models.clone(), cloud_store.clone(), cx));

        cx.subscribe(&local_models, |this, _view, event, cx| match event {
            LocalModelsEvent::UseModel(model) => {
                this.chat.update(cx, |chat, cx| chat.use_model(model.clone(), cx));
                this.route = Route::Chat(None);
                cx.notify();
            },
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

        cx.subscribe(&cloud, |this, _cloud, event, cx| match event {
            CloudEvent::UseModel(model) => {
                this.chat.update(cx, |chat, cx| chat.use_model(model.clone(), cx));
                this.route = Route::Chat(None);
                cx.notify();
            },
        })
        .detach();

        cx.subscribe(&chats, |this, _chats, event, cx| match event {
            ChatsEvent::Open(id) => this.open_chat(id.clone(), cx),
            ChatsEvent::Changed => {
                this.recent_chats = persistence::list_chats();
                cx.notify();
            },
        })
        .detach();

        cx.subscribe(&chat, |this, _chat, event, cx| match event {
            ChatEvent::Updated => {
                this.recent_chats = persistence::list_chats();
                cx.notify();
            },
            ChatEvent::OpenLocalModels => this.navigate(Route::LocalModels, cx),
        })
        .detach();

        cx.subscribe(&settings, |this, _settings, event, cx| match event {
            SettingsEvent::DataCleared {
                dialogs,
                audio,
            } => {
                this.recent_chats = persistence::list_chats();

                if *dialogs {
                    this.chat.update(cx, |chat, cx| chat.start_new(cx));
                }

                if *audio {
                    this.tts.update(cx, |tts, cx| tts.reload_after_clear(cx));
                }
                cx.notify();
            },
        })
        .detach();

        cx.subscribe(&welcome, |this, _welcome, event, cx| match event {
            WelcomeEvent::Continue => {
                persistence::set_seen_welcome();
                this.navigate(Route::LocalModels, cx);
            },
        })
        .detach();

        let route = match std::env::var("MIRAI_SCREEN").ok().as_deref() {
            Some("chat") => Route::Chat(None),
            Some("chats") => Route::Chats,
            Some("local") => Route::LocalModels,
            Some("cloud") => Route::CloudModels,
            Some("routers") if crate::engine_capabilities::CLASSIFICATION => Route::Routers,
            Some("tts") if crate::engine_capabilities::TEXT_TO_SPEECH => Route::Tts,
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
        }
    }

    pub fn handle_tray_action(
        &mut self,
        action: TrayAction,
        cx: &mut Context<Self>,
    ) {
        let route = match action {
            TrayAction::NewChat => Route::Chat(None),
            TrayAction::OpenChats => Route::Chats,
            TrayAction::Settings => Route::Settings,
        };
        self.navigate(route, cx);
    }

    pub(super) fn toggle_dark_mode(
        &mut self,
        cx: &mut Context<Self>,
    ) {
        let mut settings = settings_state::current(cx);
        settings.dark_mode = !settings.dark_mode;
        let next = if settings.dark_mode {
            theme::Theme::dark()
        } else {
            theme::Theme::light()
        };
        theme::set_theme(cx, next);
        settings_state::set(cx, settings);
        cx.notify();
    }

    pub(super) fn navigate(
        &mut self,
        route: Route,
        cx: &mut Context<Self>,
    ) {
        match &route {
            Route::Chat(None) => self.chat.update(cx, |chat, cx| chat.start_new(cx)),
            Route::Chats => self.chats.update(cx, |chats, cx| chats.reload(cx)),
            _ => {},
        }
        self.recent_chats = persistence::list_chats();
        self.route = route;
        cx.notify();
    }

    pub(super) fn open_chat(
        &mut self,
        id: String,
        cx: &mut Context<Self>,
    ) {
        if let Some(stored) = persistence::load_chat(&id) {
            self.chat.update(cx, |chat, cx| chat.load_stored(stored, cx));
            self.recent_chats = persistence::list_chats();
            self.route = Route::Chat(Some(id));
            cx.notify();
        }
    }

    fn render_content(
        &self,
        _cx: &mut Context<Self>,
    ) -> AnyElement {
        match &self.route {
            Route::LocalModels => self.local_models.clone().into_any_element(),
            Route::CloudModels => self.cloud.clone().into_any_element(),
            Route::Chat(_) => self.chat.clone().into_any_element(),
            Route::Chats => self.chats.clone().into_any_element(),
            Route::Routers => self.routers.clone().into_any_element(),
            Route::Tts => self.tts.clone().into_any_element(),
            Route::Settings => self.settings.clone().into_any_element(),

            Route::Welcome => div().into_any_element(),
        }
    }

    fn render_footer(
        &self,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
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
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme().clone();
        let root = div()
            .size_full()
            .relative()
            .bg(theme.bg)
            .text_color(theme.text)
            .font_family(FONT_SANS)
            .text_size(px(TEXT_SIZE));

        if matches!(self.route, Route::Welcome) {
            return root.child(self.welcome.clone()).children(toast::render_overlay(cx));
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
                    .child(div().flex_1().min_h_0().overflow_hidden().child(self.render_content(cx))),
            )
            .child(self.render_footer(cx))
            .children(toast::render_overlay(cx))
    }
}

pub fn open_window(cx: &mut App) {
    let bounds = Bounds::centered(None, size(px(1200.), px(800.)), cx);
    let window = cx.open_window(
        WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(TitlebarOptions {
                title: None,
                appears_transparent: true,
                traffic_light_position: Some(point(px(20.), px(18.))),
            }),
            window_min_size: Some(size(px(720.), px(560.))),
            ..Default::default()
        },
        |_, cx| cx.new(|cx| MiraiApp::new(cx)),
    );
    if window.is_ok() {
        cx.activate(true);
    }
}
