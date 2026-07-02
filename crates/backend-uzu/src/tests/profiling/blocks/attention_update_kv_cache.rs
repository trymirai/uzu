use crate::{
    backends::{
        common::{Backend, Kernels, kernel::AttentionUpdateKVCacheKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type AttentionUpdateKVCacheMetalKernel = <<Metal as Backend>::Kernels as Kernels>::AttentionUpdateKVCacheKernel;

block_bench! {
    name: attention_update_kv_cache,
    block: AttentionUpdateKVCacheMetalKernel,
    params {
        context_length: usize = model_shapes::attention_context_lengths(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        rotated_keys: [model_shapes::ATTENTION_NUM_GROUPS * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        qkv: [(model_shapes::ATTENTION_NUM_HEADS + 2 * model_shapes::ATTENTION_NUM_GROUPS) * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        key_cache: [parameters.context_length * model_shapes::ATTENTION_NUM_GROUPS * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        value_cache: [parameters.context_length * model_shapes::ATTENTION_NUM_GROUPS * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
    },
    build(context, parameters) {
        AttentionUpdateKVCacheMetalKernel::new(context, parameters.data_type, false).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            Some(&buffers.rotated_keys),
            &buffers.qkv,
            &mut buffers.key_cache,
            &mut buffers.value_cache,
            model_shapes::ATTENTION_NUM_GROUPS as u32,
            model_shapes::ATTENTION_NUM_HEADS as u32,
            model_shapes::ATTENTION_HEAD_DIM as u32,
            1,
            0,
            parameters.context_length as u32,
            encoder,
        );
    },
}
