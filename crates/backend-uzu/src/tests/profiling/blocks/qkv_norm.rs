use crate::{
    backends::{
        common::{Backend, Kernels, kernel::QKVNormKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type QkvNormMetalKernel = <<Metal as Backend>::Kernels as Kernels>::QKVNormKernel;

block_bench! {
    name: qkv_norm,
    block: QkvNormMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        qkv_input: [parameters.tokens * (model_shapes::ATTENTION_NUM_HEADS + 2 * model_shapes::ATTENTION_NUM_GROUPS) * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        scales: [model_shapes::ATTENTION_HEAD_DIM] : DataType::F32,
        qkv_output: [parameters.tokens * (model_shapes::ATTENTION_NUM_HEADS + 2 * model_shapes::ATTENTION_NUM_GROUPS) * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
    },
    build(context, parameters) {
        QkvNormMetalKernel::new(
            context,
            parameters.data_type,
            DataType::F32,
            parameters.data_type,
            DataType::F32,
            false,
            false,
        )
        .unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            Some(&buffers.qkv_input),
            Some(&buffers.scales),
            &mut buffers.qkv_output,
            parameters.tokens as u32,
            model_shapes::ATTENTION_NUM_HEADS as u32,
            model_shapes::ATTENTION_NUM_GROUPS as u32,
            model_shapes::ATTENTION_HEAD_DIM as u32,
            1e-6,
            0.0,
            0,
            model_shapes::ATTENTION_NUM_HEADS as u32,
            false,
            encoder,
        );
    },
}
