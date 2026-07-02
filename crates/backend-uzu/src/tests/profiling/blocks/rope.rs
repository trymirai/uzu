use crate::{
    backends::{
        common::{Backend, Kernels, kernel::RopeKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type RopeMetalKernel = <<Metal as Backend>::Kernels as Kernels>::RopeKernel;

block_bench! {
    name: rope,
    block: RopeMetalKernel,
    params {
        tokens: usize = model_shapes::REAL_TOKEN_COUNTS,
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        qkv: [parameters.tokens * (model_shapes::ATTENTION_NUM_HEADS + 2 * model_shapes::ATTENTION_NUM_GROUPS) * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        cosines: [parameters.tokens * model_shapes::ATTENTION_HEAD_DIM] : DataType::F32,
        sines: [parameters.tokens * model_shapes::ATTENTION_HEAD_DIM] : DataType::F32,
        rotated_queries: [model_shapes::ATTENTION_NUM_HEADS * parameters.tokens * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        rotated_keys: [model_shapes::ATTENTION_NUM_GROUPS * parameters.tokens * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
    },
    build(context, parameters) {
        RopeMetalKernel::new(context, parameters.data_type, DataType::F32, false).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &buffers.qkv,
            &buffers.cosines,
            &buffers.sines,
            &mut buffers.rotated_queries,
            Some(&mut buffers.rotated_keys),
            model_shapes::ATTENTION_HEAD_DIM as u32,
            model_shapes::ATTENTION_HEAD_DIM as u32,
            model_shapes::ATTENTION_NUM_HEADS as u32,
            Some(model_shapes::ATTENTION_NUM_GROUPS as u32),
            parameters.tokens as u32,
            encoder,
        );
    },
}
