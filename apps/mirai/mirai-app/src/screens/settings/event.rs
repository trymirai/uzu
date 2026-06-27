/// Emitted to the shell so it can react to settings-screen side effects.
pub enum SettingsEvent {
    /// The "Clear data" wizard finished deleting chats/models/etc. — the shell
    /// should refresh cached views (e.g. the sidebar's recent-chats list). When
    /// `dialogs` is true, deleted chats were included, so the shell must also
    /// reset the live chat to stop it re-saving a now-deleted conversation.
    DataCleared {
        dialogs: bool,
    },
}
