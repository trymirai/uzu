use crate::backends::common::{Allocation, Backend, CommandBuffer, Kernels};

pub const MAX_K: u32 = 512;

pub trait RadixTopKSmall: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<RadixTopKSmall = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        columns: u32,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode(
        &self,
        input: &Allocation<Self::Backend>,
        output_ids: &mut Allocation<Self::Backend>,
        output_scores: &mut Allocation<Self::Backend>,
        rows: u32,
        k: u32,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
