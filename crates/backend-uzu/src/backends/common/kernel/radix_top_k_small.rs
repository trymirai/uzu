use crate::backends::common::{Allocation, Backend, Encoder};

pub const MAX_K: u32 = 512;

pub trait RadixTopKSmall<B: Backend>: Sized {
    fn new(
        context: &B::Context,
        columns: u32,
    ) -> Result<Self, B::Error>;

    fn encode(
        &self,
        input: &Allocation<B>,
        output_ids: &mut Allocation<B>,
        output_scores: &mut Allocation<B>,
        rows: u32,
        k: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error>;
}
