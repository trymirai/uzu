#[derive(Debug, Clone, Copy, Default)]
pub enum ContextLength {
    #[default]
    Default,
    Maximal,
    Custom(usize),
}
