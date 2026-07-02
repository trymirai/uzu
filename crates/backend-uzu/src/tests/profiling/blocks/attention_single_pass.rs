use crate::{
    backends::{
        common::{Allocation, Backend, Kernels, gpu_types::ring::RingParams, kernel::AttentionSinglePassKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type AttentionSinglePassMetalKernel = <<Metal as Backend>::Kernels as Kernels>::AttentionSinglePassKernel;

block_bench! {
    name: attention_single_pass,
    block: AttentionSinglePassMetalKernel,
    params {
        context_length: usize = model_shapes::attention_context_lengths(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        queries: [model_shapes::ATTENTION_NUM_HEADS * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        keys: [parameters.context_length * model_shapes::ATTENTION_NUM_GROUPS * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        values: [parameters.context_length * model_shapes::ATTENTION_NUM_GROUPS * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
        output: [model_shapes::ATTENTION_NUM_HEADS * model_shapes::ATTENTION_HEAD_DIM] : parameters.data_type,
    },
    build(context, parameters) {
        AttentionSinglePassMetalKernel::new(
            context,
            parameters.data_type,
            model_shapes::ATTENTION_HEAD_DIM as u32,
            false,
            false,
            false,
            false,
            false,
        )
        .unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        let head_dim = model_shapes::ATTENTION_HEAD_DIM as u32;
        let num_groups = model_shapes::ATTENTION_NUM_GROUPS as u32;
        kernel.encode(
            &buffers.queries,
            &buffers.keys,
            &buffers.values,
            &mut buffers.output,
            (model_shapes::ATTENTION_NUM_HEADS / model_shapes::ATTENTION_NUM_GROUPS) as u32,
            parameters.context_length as u32,
            head_dim,
            num_groups * head_dim,
            head_dim,
            num_groups * head_dim,
            None::<RingParams>,
            1.0 / (model_shapes::ATTENTION_HEAD_DIM as f32).sqrt(),
            None::<&Allocation<Metal>>,
            None::<u32>,
            None::<&Allocation<Metal>>,
            model_shapes::ATTENTION_NUM_HEADS as u32,
            1,
            encoder,
        );
    },
}
