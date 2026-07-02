use crate::{
    backends::{
        common::{Backend, Kernels, kernel::SigmoidGateKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type SigmoidGateMetalKernel = <<Metal as Backend>::Kernels as Kernels>::SigmoidGateKernel;

block_bench! {
    name: sigmoid_gate,
    block: SigmoidGateMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        gate: [parameters.tokens * parameters.model_dim] : parameters.data_type,
        output: [parameters.tokens * parameters.model_dim] : parameters.data_type,
    },
    build(context, parameters) {
        SigmoidGateMetalKernel::new(context, parameters.data_type).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &buffers.gate,
            &mut buffers.output,
            (parameters.tokens * parameters.model_dim) as u32,
            encoder,
        );
    },
}
