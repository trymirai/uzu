mod error;
mod power_mode_control;
mod power_mode_functions;

pub use error::Error as PowerModeError;
pub use power_mode_control::PowerModeControl;
pub use power_mode_functions::{read_modes, write_mode};
