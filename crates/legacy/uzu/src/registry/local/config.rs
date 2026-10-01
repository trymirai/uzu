#[derive(Clone)]
pub struct Config {
    pub backend_identifier: String,
    pub backend_version: String,
    pub path: String,
}

impl Config {
    pub fn new(
        backend_identifier: String,
        backend_version: String,
        path: String,
    ) -> Self {
        Self {
            backend_identifier,
            backend_version,
            path,
        }
    }
}
