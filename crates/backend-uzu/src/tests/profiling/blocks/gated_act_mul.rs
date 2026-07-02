use crate::{
    backends::{
        common::{Allocation, Backend, Kernels, gpu_types::activation_type::ActivationType, kernel::GatedActMulKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type GatedActMulMetalKernel = <<Metal as Backend>::Kernels as Kernels>::GatedActMulKernel;

block_bench! {
    name: gated_act_mul,
    block: GatedActMulMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        gated_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        act_operand: [parameters.tokens * 2 * parameters.gated_dim] : parameters.data_type,
        output: [parameters.tokens * parameters.gated_dim] : parameters.data_type,
    },
    build(context, parameters) {
        GatedActMulMetalKernel::new(context, parameters.data_type, true, false).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &buffers.act_operand,
            None::<&Allocation<Metal>>,
            &mut buffers.output,
            None::<&Allocation<Metal>>,
            parameters.gated_dim as u32,
            parameters.tokens as u32,
            0,
            0,
            ActivationType::SILU,
            encoder,
        );
    },
}
