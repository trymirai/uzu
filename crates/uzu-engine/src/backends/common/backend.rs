use std::{error::Error, fmt::Debug};

use super::{CommandBuffer, ConstantBuffer, Context, GlobalBuffer, Kernels, ScratchBuffer, SparseBuffer};

pub trait Backend: Debug + Clone + Send + Sync + 'static {
    type Context: Context<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type GlobalBuffer: GlobalBuffer<Backend = Self>;
    type ConstantBuffer: ConstantBuffer<Backend = Self>;
    type ScratchBuffer: ScratchBuffer<Backend = Self>;
    type SparseBuffer: SparseBuffer<Backend = Self>;
    type AllocationPool: Send + Sync;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error + Debug;

    const NAME: &'static str;
}
