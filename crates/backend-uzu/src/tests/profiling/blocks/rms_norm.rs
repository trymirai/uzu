use crate::{
    backends::{
        common::{Allocation, Backend, Kernels, kernel::RMSNormKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type RmsNormKernel = <<Metal as Backend>::Kernels as Kernels>::RMSNormKernel;

block_bench! {
    name: rms_norm,
    block: RmsNormKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        model_dim: usize = model_shapes::model_dimensions(),
        data_type: DataType = [DataType::F32, DataType::BF16],
        full_layer_upcast: bool = [false, true],
    },
    buffers(parameters) {
        input: [parameters.tokens * parameters.model_dim] : parameters.data_type,
        scales: [parameters.model_dim] : DataType::F32,
        output: [parameters.tokens * parameters.model_dim] : parameters.data_type,
    },
    build(context, parameters) {
        RmsNormKernel::new(
            context,
            parameters.data_type,
            DataType::F32,
            parameters.data_type,
            DataType::F32,
            false,
            parameters.full_layer_upcast,
            false,
            false,
            false,
            false,
            false,
        )
        .unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            Some(&buffers.input),
            &buffers.scales,
            &mut buffers.output,
            None::<&mut Allocation<Metal>>,
            None::<&Allocation<Metal>>,
            parameters.tokens as u32,
            parameters.model_dim as u32,
            1e-6,
            0.0,
            1.0,
            encoder,
        );
    },
}
