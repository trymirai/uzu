use super::section::Section;

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
