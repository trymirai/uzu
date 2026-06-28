/// Emitted to the shell so it can react to settings-screen side effects.
pub enum SettingsEvent {
    /// The "Clear data" wizard finished deleting chats/models/etc. — the shell
    /// should refresh cached views (e.g. the sidebar's recent-chats list).
    /// `dialogs`: deleted chats were included → reset the live chat so it can't
    /// re-save a now-deleted conversation. `audio`: generated TTS audio was
    /// cleared → reload the TTS history and stop playback.
    DataCleared {
        dialogs: bool,
        audio: bool,
    },
}
