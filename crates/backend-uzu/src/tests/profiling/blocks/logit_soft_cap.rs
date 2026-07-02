use crate::{
    backends::{
        common::{Backend, Kernels, kernel::LogitSoftCapKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type LogitSoftCapMetalKernel = <<Metal as Backend>::Kernels as Kernels>::LogitSoftCapKernel;

block_bench! {
    name: logit_soft_cap,
    block: LogitSoftCapMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        logits: [parameters.tokens * parameters.model_dim] : parameters.data_type,
    },
    build(context, parameters) {
        LogitSoftCapMetalKernel::new(context, parameters.data_type).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &mut buffers.logits,
            (parameters.tokens * parameters.model_dim) as u32,
            30.0,
            encoder,
        );
    },
}
