use super::{instant_set::InstantMetric, typelist::Metric};
use crate::sources::Sources;
#[cfg(not(target_os = "macos"))]
use crate::sys::perflevel_cores;

pub struct PerformanceCores;

impl Metric for PerformanceCores {
    type Value = u8;
    const TYPE_BIT: u128 = 1 << 4;
}

impl InstantMetric for PerformanceCores {
    fn read(sources: &mut Sources) -> u8 {
        #[cfg(target_os = "macos")]
        {
            sources.soc().map(|soc| soc.pcpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            perflevel_cores().0
        }
    }
}
