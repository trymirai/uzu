use crate::{
    backends::{
        common::{Backend, Kernels, kernel::TensorAddBiasKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type TensorAddBiasMetalKernel = <<Metal as Backend>::Kernels as Kernels>::TensorAddBiasKernel;

block_bench! {
    name: tensor_add_bias,
    block: TensorAddBiasMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        input: [parameters.tokens * parameters.model_dim] : parameters.data_type,
        bias: [parameters.model_dim] : parameters.data_type,
        output: [parameters.tokens * parameters.model_dim] : parameters.data_type,
    },
    build(context, parameters) {
        TensorAddBiasMetalKernel::new(context, parameters.data_type, parameters.data_type, false).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            Some(&buffers.input),
            &buffers.bias,
            &mut buffers.output,
            parameters.model_dim as u32,
            (parameters.tokens * parameters.model_dim) as u32,
            encoder,
        );
    },
}
