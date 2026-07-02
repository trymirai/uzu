use crate::{
    backends::{
        common::{Backend, Kernels, kernel::FullPrecisionEmbeddingLookupKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type FullPrecisionEmbeddingMetalKernel = <<Metal as Backend>::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel;

block_bench! {
    name: full_precision_embedding,
    block: FullPrecisionEmbeddingMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        token_ids: [parameters.tokens] : DataType::U64,
        weights: [model_shapes::PROFILING_VOCABULARY_SIZE * parameters.model_dim] : parameters.data_type,
        output: [parameters.tokens * parameters.model_dim] : parameters.data_type,
    },
    build(context, parameters) {
        FullPrecisionEmbeddingMetalKernel::new(context, parameters.data_type).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &buffers.token_ids,
            &buffers.weights,
            &mut buffers.output,
            parameters.tokens as u32,
            model_shapes::PROFILING_VOCABULARY_SIZE as u32,
            parameters.model_dim as u32,
            1.0,
            encoder,
        );
    },
}
