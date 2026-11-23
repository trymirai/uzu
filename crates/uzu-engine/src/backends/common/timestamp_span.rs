use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TimestampSpan {
    pub name: String,
    pub source: TimestampSource,
    pub start: Instant,
    pub end: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampSource {
    Cpu,
    Gpu,
}

#[cfg(test)]
#[path = "../../../unit/backends/common/timestamp_test.rs"]
mod tests;
