#[derive(Debug, Clone)]
pub struct AsyncFetcherConfig {
    pub connections_per_file: u16,
    pub retries: u16,
    pub progress_interval_ms: u64,
}

impl Default for AsyncFetcherConfig {
    fn default() -> Self {
        Self {
            connections_per_file: 4,
            retries: 3,
            progress_interval_ms: 500,
        }
    }
}

impl AsyncFetcherConfig {
    pub fn with_connections_per_file(
        mut self,
        connections_per_file: u16,
    ) -> Self {
        self.connections_per_file = connections_per_file;
        self
    }

    pub fn with_retries(
        mut self,
        retries: u16,
    ) -> Self {
        self.retries = retries;
        self
    }

    pub fn with_progress_interval_ms(
        mut self,
        progress_interval_ms: u64,
    ) -> Self {
        self.progress_interval_ms = progress_interval_ms;
        self
    }
}
