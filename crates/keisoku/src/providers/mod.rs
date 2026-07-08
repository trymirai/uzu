mod instant;
#[cfg(target_os = "macos")]
mod interval;
#[cfg(target_os = "macos")]
mod session;

pub mod data;
pub mod marker;

pub use instant::Instant;
#[cfg(target_os = "macos")]
pub use interval::Interval;
#[cfg(target_os = "macos")]
pub use session::Session;
