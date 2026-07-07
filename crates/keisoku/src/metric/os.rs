use super::reading::Reading;
use crate::sources::Sources;

pub struct Os;

impl Reading for Os {
    type Value = String;

    fn read(_sources: &mut Sources) -> String {
        sysinfo::System::long_os_version().unwrap_or_default()
    }
}
