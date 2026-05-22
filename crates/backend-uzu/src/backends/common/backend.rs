use std::{error::Error, fmt::Debug};

use super::{
    CommandBuffer, Context, DenseBuffer, Kernels, SparseBuffer,
    kernel::{ManualKernels, matmul::MatmulError},
};

pub trait Backend: Debug + Clone + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type DenseBuffer: DenseBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type Kernels: Kernels<Backend = Self> + ManualKernels;
    type Error: Error + Debug + From<MatmulError<Self>>;

    const MIN_ALLOCATION_ALIGNMENT: usize;
    const MAX_INLINE_BYTES: usize;
}
