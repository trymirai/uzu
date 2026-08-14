#[derive(Clone)]
pub enum ContextLength {
    Default {},
    Maximal {},
    Custom {
        length: i64,
    },
}

impl Default for ContextLength {
    fn default() -> Self {
        ContextLength::Default {}
    }
}
