use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        gpu_types::trellis::LOOKUP_MAX_COLUMNS,
        kernel::{TrellisEmbeddingLookupKernel, matmul::MatmulB},
    },
    config::weight_matrix::AnyWeightMatrixSpec,
    data_type::DataType,
    encodable_block::linear::{Gather, LinearMatmul, TransformedLinearError, input_transform::TrellisRotation},
    parameters::ParameterTree,
};

/// A trellis embedding table: its rows are vocabulary entries stored in the Kronecker basis, so the readout is an
/// ordinary trellis linear (rotate, then trellis GEMM) and a lookup decodes one row and undoes the rotation.
pub struct TrellisEmbedding<B: Backend> {
    linear: LinearMatmul<B>,
    rotation: TrellisRotation<B>,
    lookup: <B::Kernels as Kernels>::TrellisEmbeddingLookupKernel,
    vocab_size: u32,
    model_dim: u32,
}

impl<B: Backend> TrellisEmbedding<B> {
    pub fn load(
        context: &B::Context,
        tree: &ParameterTree<B>,
        quantization_spec: AnyWeightMatrixSpec,
        vocab_size: u32,
        model_dim: u32,
        data_type: DataType,
    ) -> Result<Self, TransformedLinearError<B>> {
        if data_type != DataType::BF16 || model_dim > LOOKUP_MAX_COLUMNS {
            return Err(TransformedLinearError::UnsupportedConfiguration(format!(
                "trellis embedding with {data_type:?} and model dim {model_dim}"
            )));
        }
        let linear = LinearMatmul::load(
            context,
            quantization_spec,
            model_dim,
            vocab_size,
            data_type,
            data_type,
            data_type,
            &tree.subtree("quantized"),
            None,
            None,
        )?;
        let MatmulB::Trellis {
            format,
            ..
        } = linear.matrix.matmul_b()
        else {
            return Err(TransformedLinearError::UnsupportedConfiguration("Kronecker table without a trellis".into()));
        };
        let lookup = <B::Kernels as Kernels>::TrellisEmbeddingLookupKernel::new(
            context,
            format.vector_width,
            format.transition_bits,
            format.restart_columns,
        )
        .map_err(TransformedLinearError::BackendError)?;
        Ok(Self {
            rotation: TrellisRotation::load(context, &tree.subtree("incoherence_signs"), model_dim)?,
            linear,
            lookup,
            vocab_size,
            model_dim,
        })
    }

    pub fn encode_lookup(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        output: impl BufferMut<Backend = B>,
        batch_dim: u32,
        scale: f32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let MatmulB::Trellis {
            codes,
            row_scales,
            codebook,
            ..
        } = self.linear.matrix.matmul_b()
        else {
            unreachable!("trellis embeddings load trellis matrices")
        };
        self.lookup.encode(
            token_ids,
            codes,
            row_scales,
            &codebook,
            &self.rotation.signs,
            &self.rotation.mixing,
            output,
            batch_dim,
            self.vocab_size,
            self.model_dim,
            scale,
            command_buffer,
        );
    }

    pub fn encode_readout(
        &self,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        gather: Option<Gather<impl BufferRef<Backend = B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let input = self.rotation.prepare(input, batch_dim, command_buffer)?;
        self.linear.encode_with_a(input.as_matmul_a(), batch_dim, gather, command_buffer)
    }
}
