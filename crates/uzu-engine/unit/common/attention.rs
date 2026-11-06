#[cfg(any(backend = "metal", backend = "webgpu"))]
use half::bf16;

use crate::{
    backends::common::{
        Backend, Kernels,
        kernel::{AttentionKernel, AttentionKernelConfig, kv_layout::KVLayout},
    },
    data_type::DataType,
};
#[cfg(any(backend = "metal", backend = "webgpu"))]
use crate::{
    backends::common::{
        BufferRef, CommandBufferEncoding, Context,
        gpu_types::trie::TrieNode,
        kernel::{AttentionArguments, AttentionInput, AttentionPrepareKernel},
    },
    data_type::DataTypeRepr,
    encodable_block::mixer::attention::{KVCache, KVCacheView},
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, submit_command_buffer},
};

pub(crate) fn kv_layout<B: Backend>(
    context: &B::Context,
    head_dim: u32,
    num_kv_heads: u32,
) -> KVLayout {
    <B::Kernels as Kernels>::AttentionKernel::new(
        context,
        AttentionKernelConfig {
            head_dim,
            num_groups: num_kv_heads,
            num_q_heads: num_kv_heads,
            has_sinks: false,
            gate_row_stride: None,
            is_kv_cache_ring: false,
            is_causal: true,
            sliding_window_size: None,
            scale: None,
            data_type: DataType::BF16,
        },
    )
    .unwrap()
    .kv_layout()
}

#[cfg(any(backend = "metal", backend = "webgpu"))]
pub(crate) fn run_attention<B: Backend, T: DataTypeRepr>(
    config: AttentionKernelConfig,
    cache: KVCacheView,
    nodes: &[TrieNode],
    is_trie: bool,
) -> (Vec<T>, KVLayout) {
    let context = create_context::<B>();
    let suffix = nodes.len() as u32;
    let prefix = cache.prefix_len();
    let dim = config.head_dim;
    let stride = (config.num_q_heads + 2 * config.num_groups) * dim;
    let values: Vec<T> =
        (0..(prefix + suffix) * stride).map(|i| T::from(((i * 17 % 251) as f32 - 125.0) / 128.0).unwrap()).collect();
    let data = create_buffer_with_data::<B, T>(&context, &values);
    let kernel = <B::Kernels as Kernels>::AttentionKernel::new(&context, config).unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    let mut kv = KVCache::<B>::allocate_scratch(kernel.kv_layout(), prefix + suffix, &mut command).unwrap();
    let mut q = create_buffer::<B, i8>(&context, (config.num_q_heads * suffix * dim) as usize);
    let mut qs = create_buffer::<B, bf16>(&context, (config.num_q_heads * suffix) as usize);
    let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        config.data_type,
        DataType::F32,
        dim,
        false,
        config.num_groups,
        kernel.kv_layout().backend_opaque_layout,
        true,
        false,
        false,
        false,
    )
    .unwrap();
    for (buffer, value) in [
        (kv.k_codes.as_mut(), 0x7f),
        (kv.v_codes.as_mut(), 0xff),
        (kv.k_scales.as_mut(), 0xff),
        (kv.v_scales.as_mut(), 0xff),
    ] {
        let poison = create_buffer_with_data::<B, u8>(&context, &vec![value; buffer.size()]);
        command.encode_copy(&poison, buffer);
    }
    for (start, length, heads) in [(0, prefix, 0), (prefix, suffix, config.num_q_heads)] {
        if length == 0 {
            continue;
        }
        let offset = (start * stride
            + if heads == 0 {
                config.num_q_heads * dim
            } else {
                0
            }) as usize
            * config.data_type.size_in_bytes();
        prepare.encode(
            data.subrange(offset..),
            &mut q,
            &mut qs,
            Some(kv.k_codes.as_mut()),
            Some(kv.k_scales.as_mut()),
            Some(kv.v_codes.as_mut()),
            Some(kv.v_scales.as_mut()),
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            Default::default(),
            heads,
            None,
            start,
            stride,
            length,
            heads / config.num_groups * length,
            &mut command,
        );
    }
    let trie = create_buffer_with_data::<B, u32>(&context, bytemuck::cast_slice(nodes));
    let sinks = config
        .has_sinks
        .then(|| create_buffer_with_data::<B, T>(&context, &vec![T::from(1.5).unwrap(); config.num_q_heads as usize]));
    let gate = config.gate_row_stride.map(|stride| {
        create_buffer_with_data::<B, T>(
            &context,
            &(0..suffix * stride).map(|i| T::from((i % 37) as f32 / 17.0 - 1.0).unwrap()).collect::<Vec<_>>(),
        )
    });
    let result = kernel
        .encode(
            AttentionInput {
                query_codes: &q,
                query_scales: &qs,
                k_codes: kv.k_codes.as_ref(),
                k_scales: kv.k_scales.as_ref(),
                v_codes: kv.v_codes.as_ref(),
                v_scales: kv.v_scales.as_ref(),
            },
            AttentionArguments {
                suffix_length: suffix,
                trie: is_trie.then_some(&trie),
                sinks: sinks.as_ref(),
                cache,
                gate: gate.as_ref(),
            },
            &mut command,
        )
        .unwrap();
    let mut output = create_buffer::<B, T>(&context, (suffix * config.num_q_heads * dim) as usize);
    command.encode_copy(&result, &mut output);
    submit_command_buffer(command);
    (buffer_to_vec::<B, T>(&context, &output), kernel.kv_layout())
}
