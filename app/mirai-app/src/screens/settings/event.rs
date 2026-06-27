/// Emitted to the shell so it can react to settings-screen side effects.
pub enum SettingsEvent {
    /// The "Clear data" wizard finished deleting chats/models/etc. — the shell
    /// should refresh any cached views (e.g. the sidebar's recent-chats list).
    DataCleared,
}
