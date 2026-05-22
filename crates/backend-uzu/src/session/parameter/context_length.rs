#[derive(Debug, Clone, Copy)]
pub enum ContextLength {
    Default,
    Maximal,
    Custom(usize),
}

impl Default for ContextLength {
    fn default() -> Self {
        ContextLength::Default
    }
}
