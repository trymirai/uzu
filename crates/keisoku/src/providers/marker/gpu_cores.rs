use super::{instant_set::InstantMetric, typelist::Metric};
use crate::sources::Sources;

pub struct GpuCores;

impl Metric for GpuCores {
    type Value = u8;
    const TYPE_BIT: u128 = 1 << 5;
}

impl InstantMetric for GpuCores {
    fn read(sources: &mut Sources) -> u8 {
        #[cfg(target_os = "macos")]
        {
            sources.soc().map(|soc| soc.gpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            0
        }
    }
}
