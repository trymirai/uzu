#[cfg(target_vendor = "apple")]
mod apple;
mod universal;

#[cfg(target_vendor = "apple")]
pub use crate::prelude::apple::*;
pub use crate::prelude::universal::*;
