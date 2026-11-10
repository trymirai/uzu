use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TimestampSpan {
    pub name: String,
    pub start: Instant,
    pub end: Instant,
}

#[cfg(test)]
#[path = "../../../unit/backends/common/timestamp_test.rs"]
mod tests;
