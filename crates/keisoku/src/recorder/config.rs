use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Config {
    pub interval: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(1000),
        }
    }
}
