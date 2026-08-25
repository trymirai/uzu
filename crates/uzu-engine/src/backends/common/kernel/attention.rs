use crate::{
    backends::common::{Allocation, Backend, BufferArg, Encoder},
    data_type::DataType,
    encodable_block::mixer::attention::AttentionStateType,
};

pub trait AttentionKernel<B: Backend>: Sized + Send + Sync {
    fn new(
        context: &B::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, B::Error>;

    fn encode<'a, KT, VT>(
        &self,
        arguments: AttentionArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>
    where
        KT: BufferArg<'a, B>,
        VT: BufferArg<'a, B>;
}

#[derive(Clone, Copy)]
pub struct AttentionKernelConfig {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub sliding_window_size: Option<u32>,
    pub scale: Option<f32>,
    pub data_type: DataType,
}

pub struct AttentionArguments<'a, B: Backend, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>> {
    pub queries: &'a Allocation<B>,
    pub keys: KT,
    pub values: VT,
    pub suffix_length: u32,
    pub trie: Option<&'a Allocation<B>>,
    pub sinks: Option<&'a Allocation<B>>,
    pub state_type: &'a AttentionStateType,
}
