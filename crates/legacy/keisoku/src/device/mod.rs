mod device;
#[cfg(target_os = "macos")]
mod interval_handle;

pub use device::Device;
#[cfg(target_os = "macos")]
pub use interval_handle::IntervalHandle;
