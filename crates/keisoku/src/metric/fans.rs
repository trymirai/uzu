use super::reading::Reading;
use crate::{metrics::FanMetrics, sources::Sources};

pub struct Fans;

impl Reading for Fans {
    type Value = Option<FanMetrics>;

    fn read(sources: &mut Sources) -> Option<FanMetrics> {
        #[cfg(target_os = "macos")]
        {
            sources.smc().map(|smc| smc.fans())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            None
        }
    }
}
