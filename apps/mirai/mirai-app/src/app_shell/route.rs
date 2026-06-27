//! Top-level navigation route and the coarse grouping used for sidebar
//! active-state highlighting.

/// Which top-level screen is showing. `Chat(None)` is a fresh chat.
#[derive(Clone)]
pub(super) enum Route {
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
pub(super) enum Section {
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
    pub(super) fn section(&self) -> Section {
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
