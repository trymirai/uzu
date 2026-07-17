use std::{error::Error, fmt::Debug};

use crate::backends::common::{CommandBuffer, Context, DenseBuffer, Kernels, SparseBuffer};

pub trait SharedEvent: Clone + Send + Sync + Unpin {
    fn signaled_value(&self) -> u64;

    fn wait_until_signaled_value_timeout_ms(
        &self,
        value: u64,
        timeout_ms: u64,
    ) -> bool;

    fn signal(
        &self,
        value: u64,
    );
}

pub trait Backend: Debug + Clone + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type DenseBuffer: DenseBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type Kernels: Kernels<Backend = Self>;
    type SharedEvent: SharedEvent;
    type Error: Error + Debug;

    const MIN_ALLOCATION_ALIGNMENT: usize;
    const MAX_ALLOCATION_ALIGNMENT: usize;
    const ALLOCATION_GRANULARITY: usize;
    const MAX_INLINE_BYTES: usize;
}
