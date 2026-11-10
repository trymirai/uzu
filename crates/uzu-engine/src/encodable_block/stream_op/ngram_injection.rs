use crate::{
    backends::common::{Backend, CommandBuffer},
    config::{residual::gated::GatedResidualConfig, stream_op::ngram_injection::NGramInjectionConfig},
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, linear::Linear, stream_op::StreamOpNewError},
    safetensors::ParameterTree,
};

// TODO: Recommended shape, besides the projections:
// - `ngram_embedding`: a table of `sum(hash_config.head_vocab_sizes)` rows, each
//   `embedding_dim / num_heads` wide.
// - `key_norm`, `query_norm`, `conv_norm`: per-stream norms (scales `[stream_count, model_dim]`),
//   the same kind the gated residual reader needs.
// - `conv`: a separable causal conv over `stream_count * model_dim` channels with
//   `conv_kernel_size` taps.
pub struct NGramInjection<B: Backend> {
    /// `embedding_dim -> stream_count * model_dim`
    #[expect(dead_code)]
    key_projection: Box<dyn Linear<B>>,
    /// `embedding_dim -> model_dim`
    #[expect(dead_code)]
    value_projection: Box<dyn Linear<B>>,
    // TODO: More fields
}

impl<B: Backend> NGramInjection<B> {
    #[expect(unused_variables)]
    pub async fn new(
        model_dim: u32,
        data_type: DataType,
        config: &NGramInjectionConfig,
        residual_config: &GatedResidualConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, StreamOpNewError<B>> {
        todo!()
    }

    // Takes and returns a gated stream, `[tokens, stream_count, model_dim]`.
    #[expect(unused_variables)]
    pub fn encode(
        &self,
        stream: B::ScratchBuffer,
        batch_dim: &BatchTopology,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        todo!("n-gram injection: hash token n-grams, gate embeddings per stream, causal conv, add to stream")
    }
}
