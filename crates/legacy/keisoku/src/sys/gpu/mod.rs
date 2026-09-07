mod error;
mod gpu_functions;
mod gpu_power_control;

pub use error::Error as GpuControlError;
pub use gpu_functions::set_power_limit;
pub use gpu_power_control::GpuPowerControl;
