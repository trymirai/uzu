use crate::{
    backends::{
        common::{Backend, Kernels, gpu_types::activation_type::ActivationType, kernel::ActivationKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type ActivationMetalKernel = <<Metal as Backend>::Kernels as Kernels>::ActivationKernel;

block_bench! {
    name: activation,
    block: ActivationMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        input: [parameters.tokens * parameters.model_dim] : parameters.data_type,
        output: [parameters.tokens * parameters.model_dim] : parameters.data_type,
    },
    build(context, parameters) {
        ActivationMetalKernel::new(context, parameters.data_type, false).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            Some(&buffers.input),
            &mut buffers.output,
            (parameters.tokens * parameters.model_dim) as u32,
            ActivationType::SILU,
            encoder,
        );
    },
}
