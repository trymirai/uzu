use std::time::Instant;

#[derive(Debug, Clone)]
pub struct TimestampSpan {
    pub name: String,
    pub start: Instant,
    pub end: Instant,
}
