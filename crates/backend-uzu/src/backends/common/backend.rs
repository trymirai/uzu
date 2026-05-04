use std::{error::Error, fmt::Debug};

use super::{CommandBuffer, Context, DenseBuffer, Event, Kernels, kernel::ManualKernels};

pub trait Backend: Debug + Clone + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type DenseBuffer: DenseBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type Event: Event<Backend = Self>;
    type Kernels: Kernels<Backend = Self> + ManualKernels;
    type Error: Error + Debug;

    const MIN_ALLOCATION_ALIGNMENT: usize;
    const MAX_INLINE_BYTES: usize;
}
