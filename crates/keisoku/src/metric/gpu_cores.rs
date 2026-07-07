use super::reading::Reading;
use crate::sources::Sources;

pub struct GpuCores;

impl Reading for GpuCores {
    type Value = u8;

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
