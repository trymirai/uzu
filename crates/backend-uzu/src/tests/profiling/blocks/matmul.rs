use crate::{
    backends::{
        common::{
            Backend, Kernels,
            kernel::matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::block_bench::block_bench,
};

type MatmulMetalKernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

block_bench! {
    name: matmul,
    block: MatmulMetalKernel,
    params {
        tokens: usize = [1usize],
        input_dimension: usize = [1usize],
        output_dimension: usize = [1usize],
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        a: [parameters.tokens * parameters.input_dimension] : parameters.data_type,
        b: [parameters.input_dimension * parameters.output_dimension] : parameters.data_type,
        d: [parameters.tokens * parameters.output_dimension] : parameters.data_type,
    },
    build(context, parameters) {
        MatmulMetalKernel::new(
            context,
            parameters.data_type,
            parameters.data_type,
            parameters.data_type,
        )
        .unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel
            .encode(
                MatmulArguments {
                    a: &buffers.a,
                    a_offset: 0,
                    b: MatmulB::FullPrecision { b: &buffers.b },
                    b_offset: 0,
                    b_leading_dimension: None,
                    b_transpose: false,
                    d: &mut buffers.d,
                    d_transform: MatmulDOps::none(),
                    m: parameters.tokens as u32,
                    n: parameters.output_dimension as u32,
                    k: parameters.input_dimension as u32,
                },
                encoder,
            )
            .unwrap();
    },
}
