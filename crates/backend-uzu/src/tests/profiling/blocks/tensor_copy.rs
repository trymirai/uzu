use crate::{
    backends::{
        common::{Backend, Kernels, kernel::TensorCopyKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type TensorCopyMetalKernel = <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel;

block_bench! {
    name: tensor_copy,
    block: TensorCopyMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        source: [parameters.tokens * parameters.model_dim] : parameters.data_type,
        destination: [parameters.tokens * parameters.model_dim] : parameters.data_type,
    },
    build(context, parameters) {
        TensorCopyMetalKernel::new(context, parameters.data_type).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &buffers.source,
            &mut buffers.destination,
            (parameters.tokens * parameters.model_dim) as u32,
            encoder,
        );
    },
}
