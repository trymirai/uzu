use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Context, Kernels,
            gpu_types::trie::TrieNode as GpuTrieNode,
            kernel::{
                AttentionArguments, AttentionInput, AttentionKernel, AttentionPrepareInt8Kernel, Int8Input, KVFormat,
                tests::attention::attention_kernel_test::{attention_data, fill_attention, run_attention},
            },
        },
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataType,
    encodable_block::mixer::attention::{KVCacheView, kv_storage::KVStorage},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, submit_command_buffer},
    },
};

#[uzu_test]
fn int8_attention_matches_cpu() {
    let (metal, cpu) =
        (block_on(<Metal as Backend>::Context::new()).unwrap(), block_on(<Cpu as Backend>::Context::new()).unwrap());
    type Prepare = <<Metal as Backend>::Kernels as Kernels>::AttentionPrepareInt8Kernel;
    let trie: [GpuTrieNode; 3] = bytemuck::cast([0u32, 2, 0, 1, 1, 1, 2, 2, 1]); // (start, end, height) per token
    // (query heads, suffix, prefix, trie, shared KV, unwritten scale bits)
    const NAN: u16 = bf16::NAN.to_bits();
    const DENORMAL: u16 = 1; // 1 / scale overflows to inf
    for (q_heads, suffix, prefix, use_trie, reader, poison) in [
        (24, 1, 1024, false, false, NAN),
        (16, 16, 1024, false, false, DENORMAL),
        (24, 80, 16, false, false, NAN),
        (24, 3, 16, true, true, NAN),
        (24, 8, 32, false, false, DENORMAL),
        (16, 3, 37, true, false, DENORMAL),
        (32, 1, 1024, false, false, NAN),
        (32, 1024, 24, false, false, bf16::MAX.to_bits()),
        (32, 3, 1024, true, false, bf16::INFINITY.to_bits()),
    ] {
        let (head_dim, kv_heads) = if q_heads == 32 {
            (128, 2)
        } else {
            (256, 4)
        };
        let (shape, nodes) = ((head_dim, q_heads, kv_heads, suffix, prefix, true), use_trie.then_some(trie.as_slice()));
        let config = super::default_attention_config::<bf16>(head_dim, q_heads, kv_heads, true);
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(&metal, config).unwrap();
        if !kernel.supports_int8() {
            return;
        }
        // Fused QKV rows over the data `run_attention` uses; its queries are head-major.
        let ((keys, values), stride) = (attention_data(shape), (q_heads + 2 * kv_heads) * head_dim);
        let mut rows = vec![bf16::ZERO; (prefix + suffix) * stride];
        for (row, (k, v)) in
            rows.chunks_mut(stride).zip(keys.chunks(kv_heads * head_dim).zip(values.chunks(kv_heads * head_dim)))
        {
            let (key_row, value_row) = row[q_heads * head_dim..].split_at_mut(kv_heads * head_dim);
            key_row.copy_from_slice(k);
            value_row.copy_from_slice(v);
        }
        for (index, query) in fill_attention(q_heads * suffix * head_dim, 0.5).chunks(head_dim).enumerate() {
            rows[(prefix + index % suffix) * stride + index / suffix * head_dim..][..head_dim].copy_from_slice(query);
        }
        let mut cache = KVStorage::<Metal>::create_empty(
            KVFormat::INT8,
            kv_heads as u32,
            head_dim as u32,
            2048,
            DataType::BF16,
            &metal,
        )
        .unwrap();
        cache.prepare(2048).unwrap();
        let KVStorage::Int8(kv, _) = &mut cache else {
            unreachable!()
        };
        let mut fill = metal.create_command_buffer(None, None).unwrap();
        let pattern =
            create_buffer_with_data::<Metal, u16>(&metal, &vec![poison; kv.k_scales.size() / size_of::<bf16>()]);
        for buffer in [&mut kv.k_codes, &mut kv.v_codes] {
            fill.encode_fill(buffer.as_mut(), poison as u8);
        }
        for buffer in [&mut kv.k_scales, &mut kv.v_scales] {
            fill.encode_copy(&pattern, buffer.as_mut());
        }
        submit_command_buffer(fill);
        let mut step = |rows: &[bf16], prefix: usize, trie: Option<&[GpuTrieNode]>, append_kv: bool| {
            let suffix = rows.len() / stride;
            let query_rows = q_heads / kv_heads * suffix;
            let prepare = Prepare::new(
                &metal,
                DataType::BF16,
                DataType::F32,
                head_dim as u32,
                false,
                true,
                kv_heads as u32,
                append_kv,
            )
            .unwrap();
            let mut query_codes = create_buffer::<Metal, i8>(&metal, kv_heads * query_rows * head_dim);
            let mut query_scales = create_buffer::<Metal, bf16>(&metal, kv_heads * query_rows);
            let mut command_buffer = metal.create_command_buffer(None, None).unwrap();
            let trie = trie.map(|nodes| command_buffer.allocate_constant_from_slice(nodes).unwrap());
            prepare.encode(
                &create_buffer_with_data::<Metal, bf16>(&metal, rows),
                None::<&mut <Metal as Backend>::GlobalBuffer>,
                Some(&mut query_codes),
                Some(&mut query_scales),
                append_kv.then_some(kv.k_codes.as_mut()),
                append_kv.then_some(kv.k_scales.as_mut()),
                append_kv.then_some(kv.v_codes.as_mut()),
                append_kv.then_some(kv.v_scales.as_mut()),
                None::<&<Metal as Backend>::GlobalBuffer>,
                None::<&<Metal as Backend>::GlobalBuffer>,
                q_heads as u32,
                None,
                prefix as u32,
                stride as u32,
                suffix as u32,
                query_rows as u32,
                &mut command_buffer,
            );
            let input = AttentionInput::Int8(Int8Input {
                query_codes: &query_codes,
                query_scales: &query_scales,
                k_codes: kv.k_codes.as_ref(),
                k_scales: kv.k_scales.as_ref(),
                v_codes: kv.v_codes.as_ref(),
                v_scales: kv.v_scales.as_ref(),
            });
            let arguments = AttentionArguments {
                sinks: None,
                suffix_length: suffix as u32,
                trie: trie.as_ref(),
                cache: KVCacheView::full(prefix as u32),
                gate: None::<&<Metal as Backend>::GlobalBuffer>,
            };
            let pooled = kernel.encode(input, arguments, &mut command_buffer).unwrap();
            let mut output = create_buffer::<Metal, bf16>(&metal, suffix * q_heads * head_dim);
            command_buffer.encode_copy(&pooled, &mut output);
            drop((pooled, trie));
            submit_command_buffer(command_buffer);
            buffer_to_vec::<Metal, bf16>(&metal, &output)
        };
        step(&rows[..prefix * stride], 0, None, true);
        let mut actual = step(&rows[prefix * stride..], prefix, nodes, true);
        if reader {
            // The reader's own K/V rows are junk: appending them would change the output.
            rows[prefix * stride..].chunks_mut(stride).for_each(|row| row[q_heads * head_dim..].fill(bf16::ONE));
            actual = step(&rows[prefix * stride..], prefix, nodes, false);
        }
        let label = format!("int8 attention q{q_heads} S{suffix} L{prefix} trie={use_trie} reader={reader}");
        assert_eq_float::<bf16>(&run_attention::<Cpu>(cpu.as_ref(), shape, nodes, None), &actual, 1e-2, &label);
    }
}
