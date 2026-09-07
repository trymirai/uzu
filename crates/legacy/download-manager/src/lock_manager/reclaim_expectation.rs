use crate::LockFileInfo;

#[derive(Clone, Debug)]
pub enum ReclaimExpectation {
    Matching(LockFileInfo),
    UnparseableSnapshot(Vec<u8>),
}
