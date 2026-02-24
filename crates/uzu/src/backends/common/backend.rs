use std::{error::Error, fmt::Debug};

use super::{CommandBuffer, Context, CopyEncoder, Event, Kernels, NativeBuffer, kernel::matmul::MatmulKernels};

pub trait Backend: Debug + Clone + 'static {
    type Context: Context<Backend = Self>;
    type NativeBuffer: NativeBuffer<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type ComputeEncoder;
    type CopyEncoder: CopyEncoder<Backend = Self>;
    type Event: Event<Backend = Self>;
    type Kernels: Kernels<Backend = Self> + MatmulKernels;
    type Error: Error + Debug;

    const MAX_INLINE_BYTES: usize;
}
