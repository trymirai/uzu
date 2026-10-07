#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimestampSampleEntry {
    Start(String),
    End(String),
}
