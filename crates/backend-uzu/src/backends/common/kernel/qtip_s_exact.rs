use crate::{
    backends::common::{Allocation, Backend, Encoder},
    data_type::DataType,
};

pub struct QtipGaussianArguments<'a, B: Backend> {
    pub input: &'a Allocation<B>,
    pub codes: &'a Allocation<B>,
    pub codebook: &'a Allocation<B>,
    /// V4 only: the same Q8 table re-laid as [components 0,1 of all states][components 2,3 of all states]
    pub codebook_split: Option<&'a Allocation<B>>,
    /// trellis state width: 16 (65536-entry table) or 15 (32768-entry table, V4 only)
    pub state_bits: u32,
    /// 0 = plain table, 1 = antipodal L=16 table stored as its first half (kernels negate on state bit 15)
    pub table_mode: u32,
    pub codebook_scale: f32,
    pub scales: &'a Allocation<B>,
    pub gains: &'a Allocation<B>,
    pub signs: &'a Allocation<B>,
    pub small_q: &'a Allocation<B>,
    pub batch: u32,
    pub rows: u32,
    pub columns: u32,
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: u32,
}

pub struct D4S4EmbeddingArguments<'a, B: Backend> {
    pub token_ids: &'a Allocation<B>,
    pub codes: &'a Allocation<B>,
    pub row_scales: &'a Allocation<B>,
    pub ladder_indices: &'a Allocation<B>,
    pub table: &'a Allocation<B>,
    pub ladder: &'a Allocation<B>,
    pub output_hadamard_factors: &'a Allocation<B>,
    pub output: &'a mut Allocation<B>,
    pub batch: u32,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub input_scale: f32,
}

pub struct I3S4ReadoutArguments<'a, B: Backend> {
    pub input: &'a Allocation<B>,
    pub codes: &'a Allocation<B>,
    pub row_scales: &'a Allocation<B>,
    pub ladder_indices: &'a Allocation<B>,
    pub ladder: &'a Allocation<B>,
    pub input_hadamard_factors: &'a Allocation<B>,
    pub batch: u32,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub output_data_type: DataType,
}

pub struct I3S4SparseReadoutArguments<'a, B: Backend> {
    pub input: &'a Allocation<B>,
    pub token_ids: &'a Allocation<B>,
    pub codes: &'a Allocation<B>,
    pub row_scales: &'a Allocation<B>,
    pub ladder_indices: &'a Allocation<B>,
    pub ladder: &'a Allocation<B>,
    pub input_hadamard_factors: &'a Allocation<B>,
    pub rows: u32,
    pub ids_per_row: u32,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub output_data_type: DataType,
    /// 0 disables the fused soft cap
    pub soft_cap: f32,
}

pub trait QtipSExactKernel<B: Backend>: Send + Sync + Sized {
    fn new(context: &B::Context) -> Result<Self, B::Error>;

    fn encode_qtip_gaussian(
        &self,
        arguments: QtipGaussianArguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;

    fn encode_d4_s4_embedding(
        &self,
        arguments: D4S4EmbeddingArguments<'_, B>,
        encoder: &mut Encoder<B>,
    );

    fn encode_i3_s4_readout(
        &self,
        arguments: I3S4ReadoutArguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;

    /// Weaver candidate scoring against an i3/S4 head: `out[r][j] = <x_r, W[token_ids[r][j]]>`
    fn encode_i3_s4_readout_sparse(
        &self,
        arguments: I3S4SparseReadoutArguments<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;

    /// `out[i] = token_ids[i] < hot_rows ? hot[i] : cold[i]` over `count` bf16 residuals
    fn encode_residual_merge_hot(
        &self,
        hot: &Allocation<B>,
        cold: &Allocation<B>,
        token_ids: &Allocation<B>,
        output: &mut Allocation<B>,
        hot_rows: u32,
        count: u32,
        encoder: &mut Encoder<B>,
    );
}
