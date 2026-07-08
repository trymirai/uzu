#[cfg(target_os = "macos")]
use crate::sys::smc::{FansSnapshot, Smc};

#[cfg(target_os = "macos")]
pub(crate) fn new_smc() -> Option<Smc> {
    Smc::new()
}

#[cfg(target_os = "macos")]
pub(crate) fn fans(smc: &Smc) -> FansSnapshot {
    smc.fans()
}

#[cfg(not(target_os = "macos"))]
pub(crate) struct Smc;

#[cfg(not(target_os = "macos"))]
pub(crate) fn new_smc() -> Option<Smc> {
    None
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn fans(_smc: &Smc) -> crate::sys::smc::FansSnapshot {
    crate::sys::smc::FansSnapshot::default()
}
