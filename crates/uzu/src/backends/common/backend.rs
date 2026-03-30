use std::{error::Error, fmt::Debug};

use super::{Buffer, CommandBuffer, Context, Event, Kernels, kernel::ManualKernels};

pub trait Backend: Debug + Clone + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type Buffer: Buffer<Backend = Self>;
    type Event: Event<Backend = Self>;
    type Kernels: Kernels<Backend = Self> + ManualKernels;
    type Error: Error + Debug;

    const MIN_ALLOCATION_ALIGNMENT: usize;
    const MAX_INLINE_BYTES: usize;
}
