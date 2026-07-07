use super::reading::Reading;
#[cfg(not(target_os = "macos"))]
use super::sysctl::sysctl_string;
use crate::sources::Sources;

pub struct Chip;

impl Reading for Chip {
    type Value = String;

    fn read(sources: &mut Sources) -> String {
        #[cfg(target_os = "macos")]
        if let Some(soc) = sources.soc()
            && !soc.chip_name.is_empty()
        {
            return soc.chip_name.clone();
        }
        #[cfg(not(target_os = "macos"))]
        if let Some(model) = sysctl_string("hw.machine").filter(|model| !model.is_empty()) {
            return model;
        }
        sources.system().cpus().first().map(|cpu| cpu.brand().trim().to_string()).unwrap_or_default()
    }
}
