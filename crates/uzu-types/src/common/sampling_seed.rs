#[derive(Clone)]
pub enum SamplingSeed {
    Default {},
    Custom {
        seed: i64,
    },
}

impl Default for SamplingSeed {
    fn default() -> Self {
        SamplingSeed::Default {}
    }
}
