use crate::{
    backends::{
        common::{Backend, Kernels, kernel::RepetitionPenaltyKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::profiling::{block_bench::block_bench, model_shapes},
};

type RepetitionPenaltyMetalKernel = <<Metal as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel;

block_bench! {
    name: repetition_penalty,
    block: RepetitionPenaltyMetalKernel,
    params {
        vocab_size: usize = model_shapes::profiling_vocabulary_sizes(),
        data_type: DataType = [DataType::F32, DataType::BF16],
    },
    buffers(parameters) {
        original_logits: [parameters.vocab_size] : parameters.data_type,
        logits_copy: [parameters.vocab_size] : parameters.data_type,
        context_ring: [2 + 64] : DataType::U32,
        token_ids: [1] : DataType::U64,
    },
    build(context, parameters) {
        RepetitionPenaltyMetalKernel::new(context, parameters.data_type).unwrap()
    },
    encode(kernel, buffers, parameters, encoder) {
        kernel.encode(
            &buffers.original_logits,
            &mut buffers.logits_copy,
            &buffers.context_ring,
            &buffers.token_ids,
            1.1,
            64,
            encoder,
        );
    },
}
