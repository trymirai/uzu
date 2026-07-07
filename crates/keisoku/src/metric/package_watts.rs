use super::reading::Reading;
use crate::{sources::Sources, units::Watts};

pub struct PackageWatts;

impl Reading for PackageWatts {
    type Value = Option<Watts>;

    fn read(sources: &mut Sources) -> Option<Watts> {
        #[cfg(target_os = "macos")]
        {
            sources.smc().and_then(|smc| smc.package_watts())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            None
        }
    }
}
