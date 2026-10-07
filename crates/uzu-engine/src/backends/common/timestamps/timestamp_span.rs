use std::time::Instant;

use super::BlockName;

#[derive(Debug, Clone)]
pub struct TimestampSpan {
    pub name: BlockName,
    pub start: Instant,
    pub end: Instant,
}
