use crate::backends::common::{Allocation, Backend, Encoder, Kernels};

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
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
