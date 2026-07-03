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
