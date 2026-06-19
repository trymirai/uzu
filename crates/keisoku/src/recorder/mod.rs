//! Background recorder: samples the system at a fixed interval into a
//! time-series [`Session`] that can be marked with inference phases and
//! exported as JSON for offline correlation.

mod config;
mod device;
mod handle;
mod marker;
mod session;

pub use config::Config;
pub use device::Device;
pub use handle::{RecorderHandle, start};
pub use marker::Marker;
pub use session::Session;
