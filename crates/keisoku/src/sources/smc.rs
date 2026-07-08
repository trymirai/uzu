use crate::sys::smc::{FansSnapshot, Smc};

pub(crate) fn new_smc() -> Option<Smc> {
    Smc::new()
}

pub(crate) fn fans(smc: &Smc) -> FansSnapshot {
    smc.fans()
}
