use super::reading::Reading;
use crate::{metrics::MemoryMetrics, sources::Sources};

pub struct Memory;

impl Reading for Memory {
    type Value = Option<MemoryMetrics>;

    fn read(_sources: &mut Sources) -> Option<MemoryMetrics> {
        MemoryMetrics::read()
    }
}
