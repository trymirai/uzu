#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedFileState {
    Valid,
    Invalid,
    Missing,
}
