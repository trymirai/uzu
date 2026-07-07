use super::reading::Reading;
#[cfg(not(target_os = "macos"))]
use super::sysctl::perflevel_cores;
use crate::sources::Sources;

pub struct PerformanceCores;

impl Reading for PerformanceCores {
    type Value = u8;

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
