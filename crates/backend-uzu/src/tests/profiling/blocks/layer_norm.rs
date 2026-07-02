use crate::{
    backends::{
        common::{Backend, Kernels, kernel::LayerNormKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type LayerNormMetalKernel = <<Metal as Backend>::Kernels as Kernels>::LayerNormKernel;

block_bench! {
    name: layer_norm,
    block: LayerNormMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        input: [parameters.tokens * parameters.model_dim] : parameters.data_type,
        scales: [parameters.model_dim] : DataType::F32,
        output: [parameters.tokens * parameters.model_dim] : parameters.data_type,
    },
    build(context, parameters) {
        LayerNormMetalKernel::new(
            context,
            parameters.data_type,
            DataType::F32,
            parameters.data_type,
            DataType::F32,
            false,
        )
        .unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            Some(&buffers.input),
            &buffers.scales,
            &mut buffers.output,
            parameters.tokens as u32,
            parameters.model_dim as u32,
            1e-6,
            0.0,
            0,
            encoder,
        );
    },
}
