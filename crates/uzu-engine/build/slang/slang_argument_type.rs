#[derive(Debug, Clone)]
pub enum SlangArgumentType {
    Ptr,
    Constant,
    Axis(Box<str>),
    Groups,
    Threads(Box<str>),
}
