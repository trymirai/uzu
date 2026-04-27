use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestRecord {
    pub order: u64,
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub range: Option<(u64, Option<u64>)>,
    pub status: u16,
    pub bytes_sent: u64,
}
