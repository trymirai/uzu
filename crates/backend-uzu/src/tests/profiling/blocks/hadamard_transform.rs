use crate::{
    backends::{
        common::{
            Backend, Kernels, gpu_types::hadamard_order::HadamardTransformOrder, kernel::HadamardTransformKernel,
        },
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type HadamardTransformMetalKernel = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel;

block_bench! {
    name: hadamard_transform,
    block: HadamardTransformMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        data: [parameters.tokens * parameters.model_dim] : parameters.data_type,
        factors: [parameters.model_dim] : DataType::I32,
    },
    build(context, parameters) {
        HadamardTransformMetalKernel::new(context, parameters.data_type, HadamardTransformOrder::Input).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &mut buffers.data,
            &buffers.factors,
            parameters.model_dim as u32,
            parameters.tokens as u32,
            encoder,
        );
    },
}
