mod error;
#[cfg(feature = "hardware-control")]
mod fan_control;
#[cfg(feature = "hardware-control")]
mod fan_state;
mod fourcc;
mod smc;
mod smc_key_data;
mod smc_key_info;
mod smc_limit_data;
mod smc_version;

pub use error::Error as SmcError;
#[cfg(feature = "hardware-control")]
pub use fan_control::FanControl;
#[cfg(feature = "hardware-control")]
pub use fan_state::FanState;
pub use fourcc::fourcc;
pub use smc::Smc;
pub use smc_key_data::SmcKeyData;
pub use smc_key_info::SmcKeyInfo;
pub use smc_limit_data::SmcLimitData;
pub use smc_version::SmcVersion;
