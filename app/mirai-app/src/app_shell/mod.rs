//! App shell: window chrome, sidebar navigation, content outlet, and footer.
//! [`route`] holds the navigation enums; [`shell`] holds `MiraiApp`.

mod route;
mod shell;

pub use shell::MiraiApp;
