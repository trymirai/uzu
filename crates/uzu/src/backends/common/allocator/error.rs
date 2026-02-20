use std::fmt;

#[derive(Debug, Clone)]
pub enum AllocError {
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    AllocationFailed {
        size: usize,
        reason: String,
    },
}

impl fmt::Display for AllocError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            AllocError::OutOfMemory {
                requested,
                available,
            } => {
                write!(f, "Out of memory: requested {} bytes, {} available", requested, available)
            },
            AllocError::AllocationFailed {
                size,
                reason,
            } => {
                write!(f, "Failed to allocate {} bytes: {}", size, reason)
            },
        }
    }
}

impl std::error::Error for AllocError {}
