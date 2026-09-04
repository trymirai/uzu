/// The Mirai API surface. Each variant names one endpoint, so call sites refer
/// to a path by name instead of repeating a string literal.
pub enum Endpoint {
    FetchModels,
}

impl Endpoint {
    pub fn path(&self) -> &'static str {
        match self {
            Self::FetchModels => "fetch/models",
        }
    }
}
