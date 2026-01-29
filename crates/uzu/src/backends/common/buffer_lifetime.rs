#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferLifetime {
    Permanent,
    Scratch,
}
