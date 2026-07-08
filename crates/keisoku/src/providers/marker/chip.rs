use super::{instant_set::InstantMetric, typelist::Metric};
use crate::sources::Sources;

pub struct Chip;

impl Metric for Chip {
    type Value = String;
    const TYPE_BIT: u128 = 1 << 1;
}

impl InstantMetric for Chip {
    fn read(sources: &mut Sources) -> String {
        #[cfg(target_os = "macos")]
        {
            sources.soc().map(|soc| soc.chip_name.clone()).unwrap_or_default()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            crate::sys::sysctl_string("hw.machine").unwrap_or_default()
        }
    }
}
