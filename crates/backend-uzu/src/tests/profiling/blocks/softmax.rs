use crate::{
    backends::{
        common::{Allocation, Backend, Kernels, kernel::SoftmaxKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type SoftmaxMetalKernel = <<Metal as Backend>::Kernels as Kernels>::SoftmaxKernel;

block_bench! {
    name: softmax,
    block: SoftmaxMetalKernel,
    params {
        rows: usize = model_shapes::REAL_TOKEN_COUNTS,
        row_length: usize = model_shapes::profiling_model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        values: [parameters.rows * parameters.row_length] : parameters.data_type,
    },
    build(context, parameters) {
        SoftmaxMetalKernel::new(context, parameters.data_type, false).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &mut buffers.values,
            None::<&Allocation<Metal>>,
            parameters.row_length as u32,
            parameters.rows as u32,
            1,
            encoder,
        );
    },
}
