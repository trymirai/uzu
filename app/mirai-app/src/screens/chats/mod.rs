//! Chats history screen: saved-chat list with search, rename, multi-select
//! delete, and the global-instructions card. [`view`] holds `ChatsView`.

mod event;
mod util;
mod view;

pub use event::ChatsEvent;
pub use view::ChatsView;
