use super::{constant::ConstantMetric, typelist::Metric};
use crate::sources::Sources;
#[cfg(not(target_os = "macos"))]
use crate::sys::perflevel_cores;

pub struct EfficiencyCores;

impl Metric for EfficiencyCores {
    type Value = u8;
    const TYPE_BIT: u128 = 1 << 3;
}

impl ConstantMetric for EfficiencyCores {
    fn read(sources: &mut Sources) -> u8 {
        #[cfg(target_os = "macos")]
        {
            sources.soc().map(|soc| soc.ecpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            perflevel_cores().1
        }
    }
}
