use std::time::Duration;

use indexmap::IndexMap;

pub struct Config {
    pub base_url: String,
    pub timeout: Duration,
    pub headers: IndexMap<String, String>,
}

impl Config {
    pub fn new(
        base_url: String,
        timeout: Duration,
        headers: IndexMap<String, String>,
    ) -> Self {
        Self {
            base_url,
            timeout,
            headers,
        }
    }
}
