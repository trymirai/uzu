use crate::{
    backends::common::{Allocation, Backend, BufferArg, Encoder, Kernels},
    data_type::DataType,
    encodable_block::mixer::attention::AttentionStateType,
};

pub trait AttentionKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode<'a, KT, VT>(
        &self,
        arguments: AttentionArguments<'a, Self::Backend, KT, VT>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<Allocation<Self::Backend>, <Self::Backend as Backend>::Error>
    where
        KT: BufferArg<'a, Self::Backend>,
        VT: BufferArg<'a, Self::Backend>;
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
