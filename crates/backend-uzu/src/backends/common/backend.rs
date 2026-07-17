use std::{error::Error, fmt::Debug};

use super::{CommandBuffer, Context, DenseBuffer, Kernels, SparseBuffer};

pub trait Backend: Debug + Clone + Send + Sync + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type DenseBuffer: DenseBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error + Debug;

    const MIN_ALLOCATION_ALIGNMENT: usize;
    const MAX_ALLOCATION_ALIGNMENT: usize;
    const ALLOCATION_GRANULARITY: usize;
    const MAX_INLINE_BYTES: usize;
}
