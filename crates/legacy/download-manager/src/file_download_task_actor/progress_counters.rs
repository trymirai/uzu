#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgressCounters {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}
