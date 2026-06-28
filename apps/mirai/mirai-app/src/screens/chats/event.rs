/// Emitted to the shell.
pub enum ChatsEvent {
    /// Open a saved chat.
    Open(String),
    /// Chats were deleted/renamed here — refresh the sidebar's cached list.
    Changed,
}
