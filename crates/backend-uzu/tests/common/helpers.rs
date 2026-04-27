use std::{mem::size_of, rc::Rc};

use backend_uzu::{
    ArrayElement, allocation_copy_from_slice,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder,
        kernel::{
            ManualKernels,
            attention::{AttentionGemmArguments, AttentionGemmBlock},
            matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
        },
    },
};
use num_traits::Float;

pub fn allocation_size_bytes<T>(elements_count: usize) -> usize {
    elements_count * size_of::<T>()
}

pub fn alloc_allocation<B: Backend, T>(
    context: &B::Context,
    elements_count: usize,
) -> Allocation<B> {
    context
        .create_allocation(allocation_size_bytes::<T>(elements_count), AllocationType::Global)
        .expect("Failed to create allocation")
}

pub fn alloc_allocation_with_data<B: Backend, T: ArrayElement>(
    context: &B::Context,
    data: &[T],
) -> Allocation<B> {
    let allocation = context
        .create_allocation(allocation_size_bytes::<T>(data.len()), AllocationType::Global)
        .expect("Failed to create allocation");
    allocation_copy_from_slice(&allocation, data).expect("Failed to initialize allocation");
    allocation
}

pub fn allocation_to_vec<B: Backend, T: ArrayElement>(allocation: &Allocation<B>) -> Vec<T> {
    backend_uzu::allocation_to_vec(allocation)
}

pub fn allocation_prefix_to_vec<B: Backend, T: ArrayElement>(
    allocation: &Allocation<B>,
    elements_count: usize,
) -> Vec<T> {
    let mut values = allocation_to_vec::<B, T>(allocation);
    values.truncate(elements_count);
    values
}

pub fn write_allocation<B: Backend, T: ArrayElement>(
    allocation: &Allocation<B>,
    data: &[T],
) {
    allocation_copy_from_slice(allocation, data).expect("Failed to write allocation")
}

pub fn create_context<B: Backend>() -> Rc<<B as Backend>::Context> {
    B::Context::new().expect(format!("Failed to create context for {}", std::any::type_name::<B>()).as_str())
}

pub fn submit_encoder<B: Backend>(encoder: Encoder<B>) {
    encoder.end_encoding().submit().wait_until_completed().unwrap();
}

pub struct FlatAttentionInput<T: ArrayElement + Float> {
    pub queries: Box<[T]>,
    pub keys: Box<[T]>,
    pub values: Box<[T]>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub sequence_length: usize,
    pub suffix_length: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub do_causal: bool,
}

pub enum OutputStorage {
    Global,
    ScratchCopy,
}

pub fn generate_boxed_slice<T, F>(
    len: usize,
    mut generator: F,
) -> Box<[T]>
where
    T: ArrayElement + Float,
    F: FnMut(usize) -> f32,
{
    (0..len).map(|index| T::from(generator(index)).unwrap()).collect::<Vec<_>>().into_boxed_slice()
}

pub fn pad_attention_cache<T: ArrayElement + Float>(
    values: &[T],
    num_kv_heads: usize,
    sequence_length: usize,
    max_sequence_length: usize,
    head_dim: usize,
) -> Vec<T> {
    assert!(max_sequence_length >= sequence_length);

    let mut padded_values = vec![T::zero(); num_kv_heads * max_sequence_length * head_dim];
    let source_head_span = sequence_length * head_dim;
    let padded_head_span = max_sequence_length * head_dim;

    for head_index in 0..num_kv_heads {
        let source_start = head_index * source_head_span;
        let destination_start = head_index * padded_head_span;
        padded_values[destination_start..destination_start + source_head_span]
            .copy_from_slice(&values[source_start..source_start + source_head_span]);
    }

    padded_values
}

pub fn expected_row_major_matmul<T: ArrayElement + Float>(
    input: &[T],
    weights: &[T],
    input_dim: usize,
    output_dim: usize,
) -> Vec<T> {
    let row_count = input.len() / input_dim;
    let mut output = vec![T::zero(); row_count * output_dim];

    for row_index in 0..row_count {
        for output_column in 0..output_dim {
            let mut accumulator = 0.0_f32;
            for input_index in 0..input_dim {
                let input_value = input[row_index * input_dim + input_index].to_f32().unwrap();
                let weight_value = weights[output_column * input_dim + input_index].to_f32().unwrap();
                accumulator += input_value * weight_value;
            }
            output[row_index * output_dim + output_column] = T::from(accumulator).unwrap();
        }
    }

    output
}

fn flat_attention_output_elements<T: ArrayElement + Float>(input: &FlatAttentionInput<T>) -> usize {
    input.suffix_length * input.num_heads * input.head_dim
}

fn build_attention_allocations<T: ArrayElement + Float, B: Backend>(
    context: &B::Context,
    input: &FlatAttentionInput<T>,
    max_sequence_length: usize,
) -> (Allocation<B>, Allocation<B>, Allocation<B>) {
    let queries = alloc_allocation_with_data(context, &input.queries);

    if max_sequence_length == input.sequence_length {
        let keys = alloc_allocation_with_data(context, &input.keys);
        let values = alloc_allocation_with_data(context, &input.values);
        (queries, keys, values)
    } else {
        let padded_keys = pad_attention_cache(
            &input.keys,
            input.num_kv_heads,
            input.sequence_length,
            max_sequence_length,
            input.head_dim,
        );
        let padded_values = pad_attention_cache(
            &input.values,
            input.num_kv_heads,
            input.sequence_length,
            max_sequence_length,
            input.head_dim,
        );
        let keys = alloc_allocation_with_data(context, &padded_keys);
        let values = alloc_allocation_with_data(context, &padded_values);
        (queries, keys, values)
    }
}

fn encode_attention_gemm<T: ArrayElement + Float, B: Backend>(
    input: &FlatAttentionInput<T>,
    queries: &Allocation<B>,
    keys: &Allocation<B>,
    values: &Allocation<B>,
    output: &mut Allocation<B>,
    max_sequence_length: usize,
    encoder: &mut Encoder<B>,
) {
    let segment_prefix_length = input.sequence_length - input.suffix_length;

    AttentionGemmBlock::<B>::new(T::data_type())
        .encode(
            encoder,
            AttentionGemmArguments {
                queries_buffer: queries,
                keys_buffer: keys,
                values_buffer: values,
                output_buffer: output,
                trie_buffer: None,
                sinks_buffer: None,
                num_heads: input.num_heads,
                num_groups: input.num_kv_heads,
                suffix_length: input.suffix_length,
                sequence_length: input.sequence_length,
                segment_prefix_length,
                max_sequence_length,
                ring_params: None,
                head_dim: input.head_dim,
                sliding_window_size: None,
                is_causal: input.do_causal,
                scale: input.scale,
            },
        )
        .expect("Failed to encode AttentionGemm");
}

pub fn run_attention_gemm<T: ArrayElement + Float, B: Backend>(
    input: &FlatAttentionInput<T>,
    max_sequence_length: usize,
    output_storage: OutputStorage,
) -> Vec<T> {
    let context = create_context::<B>();
    let (queries, keys, values) = build_attention_allocations(context.as_ref(), input, max_sequence_length);
    let output_elements = flat_attention_output_elements(input);
    let output_size_bytes = allocation_size_bytes::<T>(output_elements);
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    match output_storage {
        OutputStorage::Global => {
            let mut output = alloc_allocation::<B, T>(context.as_ref(), output_elements);
            encode_attention_gemm(input, &queries, &keys, &values, &mut output, max_sequence_length, &mut encoder);
            submit_encoder(encoder);
            allocation_to_vec::<B, T>(&output)
        },
        OutputStorage::ScratchCopy => {
            {
                let mut dirty_scratch =
                    encoder.allocate_scratch(output_size_bytes).expect("Failed to allocate dirty scratch");
                encoder.encode_fill(&mut dirty_scratch, 0x7f);
            }

            let mut scratch_output =
                encoder.allocate_scratch(output_size_bytes).expect("Failed to allocate pooled output");
            let mut output = alloc_allocation::<B, T>(context.as_ref(), output_elements);

            encode_attention_gemm(
                input,
                &queries,
                &keys,
                &values,
                &mut scratch_output,
                max_sequence_length,
                &mut encoder,
            );

            encoder.encode_copy(&scratch_output, .., &mut output, ..);

            drop(scratch_output);
            submit_encoder(encoder);
            allocation_to_vec::<B, T>(&output)
        },
    }
}

pub fn followup_matmul_weights<T: ArrayElement + Float>(
    input_dim: usize,
    output_dim: usize,
) -> Box<[T]> {
    generate_boxed_slice(output_dim * input_dim, |index| (index as f32 * 0.017 + 0.25).cos() * 0.5)
}

pub fn run_attention_gemm_followed_by_matmul<T: ArrayElement + Float, B: Backend>(
    input: &FlatAttentionInput<T>,
    output_dim: usize,
) -> Vec<T>
where
    B::Kernels: ManualKernels,
{
    let context = create_context::<B>();
    let (queries, keys, values) = build_attention_allocations(context.as_ref(), input, input.sequence_length);
    let input_dim = input.num_heads * input.head_dim;
    let weights = followup_matmul_weights::<T>(input_dim, output_dim);
    let weights_allocation = alloc_allocation_with_data::<B, T>(context.as_ref(), &weights);
    let attention_output_elements = input.suffix_length * input_dim;
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let mut matmul = <B::Kernels as ManualKernels>::MatmulKernel::new(context.as_ref(), T::data_type())
        .expect("Failed to create matmul");

    {
        let mut dirty_scratch = encoder
            .allocate_scratch(allocation_size_bytes::<T>(attention_output_elements))
            .expect("Failed to allocate dirty scratch");
        encoder.encode_fill(&mut dirty_scratch, 0x5a);
    }

    let mut attention_output = encoder
        .allocate_scratch(allocation_size_bytes::<T>(attention_output_elements))
        .expect("Failed to allocate attention scratch output");
    let mut matmul_output = alloc_allocation::<B, T>(context.as_ref(), input.suffix_length * output_dim);

    encode_attention_gemm(input, &queries, &keys, &values, &mut attention_output, input.sequence_length, &mut encoder);
    matmul.encode(
        MatmulArguments {
            a: &attention_output,
            a_offset: 0,
            b: &weights_allocation,
            ab_scale: 1.0,
            c: MatmulArgumentC::None,
            d: &mut matmul_output,
            batch_dim: input.suffix_length as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        },
        &mut encoder,
    );

    drop(attention_output);
    submit_encoder(encoder);
    allocation_to_vec::<B, T>(&matmul_output)
}
