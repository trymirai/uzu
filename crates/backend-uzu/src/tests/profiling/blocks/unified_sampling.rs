use crate::{
    backends::{
        common::{Allocation, Backend, Kernels, kernel::UnifiedSamplingKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type UnifiedSamplingMetalKernel = <<Metal as Backend>::Kernels as Kernels>::UnifiedSamplingKernel;

block_bench! {
    name: unified_sampling,
    block: UnifiedSamplingMetalKernel,
    params {
        vocab_size: usize = model_shapes::profiling_vocabulary_sizes(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        logits: [parameters.vocab_size] : parameters.data_type,
        output: [1] : DataType::U32,
    },
    build(context, parameters) {
        UnifiedSamplingMetalKernel::new(
            context,
            parameters.data_type,
            false,
            false,
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
            &buffers.logits,
            &mut buffers.output,
            None::<&Allocation<Metal>>,
            None::<&Allocation<Metal>>,
            None::<f32>,
            None::<u32>,
            None::<f32>,
            None::<f32>,
            parameters.vocab_size as u32,
            1,
            encoder,
        );
    },
}
