//! App shell: window chrome, sidebar navigation, content outlet, and footer.
//! [`route`] holds the navigation enums; [`shell`] holds `MiraiApp`.

mod route;
mod shell;
mod sidebar;

pub use shell::MiraiApp;
