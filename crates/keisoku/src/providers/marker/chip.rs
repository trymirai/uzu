use super::{constant::ConstantMetric, typelist::Metric};
use crate::sources::Sources;

pub struct Chip;

impl Metric for Chip {
    type Value = String;
    const TYPE_BIT: u128 = 1 << 1;
}

impl ConstantMetric for Chip {
    fn read(sources: &mut Sources) -> String {
        #[cfg(target_os = "macos")]
        if let Some(soc) = sources.soc()
            && !soc.chip_name.is_empty()
        {
            return soc.chip_name.clone();
        }
        #[cfg(not(target_os = "macos"))]
        if let Some(model) = crate::sys::sysctl_string("hw.machine").filter(|model| !model.is_empty()) {
            return model;
        }
        sources.system().cpus().first().map(|cpu| cpu.brand().trim().to_string()).unwrap_or_default()
    }
}
