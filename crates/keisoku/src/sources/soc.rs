#[cfg(target_os = "macos")]
use crate::sys::{ioreport::decode::FrequencyTables, soc::SocInfo};

#[cfg(target_os = "macos")]
pub(crate) fn new_soc() -> Option<SocInfo> {
    SocInfo::new()
}

#[cfg(target_os = "macos")]
pub(crate) fn frequencies(soc: &SocInfo) -> FrequencyTables<'_> {
    FrequencyTables {
        ecpu: &soc.ecpu_frequencies,
        pcpu: &soc.pcpu_frequencies,
        gpu: &soc.gpu_frequencies,
        ecpu_cores: soc.ecpu_cores,
        pcpu_cores: soc.pcpu_cores,
    }
}
