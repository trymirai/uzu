use super::reading::Reading;
use crate::{sources::Sources, units::Bytes};

pub struct RamTotal;

impl Reading for RamTotal {
    type Value = Bytes;

    fn read(sources: &mut Sources) -> Bytes {
        Bytes(sources.system().total_memory())
    }
}
