use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, Context, Kernels,
        gpu_types::kv_cache_update::Copy,
        kernel::{AttentionPrepareKernel, IndexerCacheUpdateKernel, KVCacheUpdateKernel, kv_layout::KVLayout},
    },
    data_type::DataType,
    encodable_block::mixer::attention::KVCache,
    tests::{
        attention::kv_layout,
        helpers::{
            buffer_readback, buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_backend,
            submit_command_buffer,
        },
    },
};

const HEAD_DIM: u32 = 256;
const NUM_HEADS: u32 = 2;
const KEYS: u32 = 64;
const ROW_WIDTH: usize = (2 * NUM_HEADS * HEAD_DIM) as usize;

fn prepare_cache<B: Backend>(
    context: &B::Context,
    layout: KVLayout,
    rows: &[bf16],
) -> KVCache<B> {
    let mut command = context.create_command_buffer(None, None).unwrap();
    let mut kv = KVCache::<B>::allocate_scratch(layout, KEYS, &mut command).unwrap();
    let input = create_buffer_with_data::<B, bf16>(context, rows);
    let mut q = create_buffer::<B, i8>(context, 1);
    let mut qs = create_buffer::<B, bf16>(context, 1);
    let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
        context,
        DataType::BF16,
        HEAD_DIM,
        false,
        NUM_HEADS,
        layout.backend_opaque_layout,
        true,
        false,
        false,
        false,
        false,
        0,
        false,
    )
    .unwrap();
    prepare.encode(
        &input,
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
        None::<&mut B::GlobalGpuBuffer>,
        Default::default(),
        0,
        None,
        None,
        None,
        0,
        ROW_WIDTH as u32,
        KEYS,
        0,
        None,
        None::<&B::GlobalGpuBuffer>,
        &mut command,
    );
    submit_command_buffer(command);
    kv
}

fn check_kv_copy<B: Backend>(keys_only: bool) {
    let context = create_context::<B>();
    let layout = kv_layout::<B>(&context, HEAD_DIM, NUM_HEADS);
    let mut rows: Vec<bf16> =
        (0..KEYS as usize * ROW_WIDTH).map(|i| bf16::from_f32(((i * 37 % 251) as f32 - 125.0) / 128.0)).collect();
    let mut actual = prepare_cache::<B>(&context, layout, &rows);
    let copies = [(6, 5), (7, 6), (9, 7)].map(|(source, destination)| Copy {
        source,
        destination,
    });
    for copy in &copies {
        rows.copy_within(
            copy.source as usize * ROW_WIDTH..(copy.source as usize + 1) * ROW_WIDTH,
            copy.destination as usize * ROW_WIDTH,
        );
    }
    let expected = prepare_cache::<B>(&context, layout, &rows);
    let kernel = <B::Kernels as Kernels>::KVCacheUpdateKernel::new(
        &context,
        HEAD_DIM,
        NUM_HEADS,
        layout.backend_opaque_layout,
        keys_only,
    )
    .unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        actual.k_codes.as_mut(),
        (!keys_only).then_some(actual.v_codes.as_mut()),
        actual.k_scales.as_mut(),
        (!keys_only).then_some(actual.v_scales.as_mut()),
        &copies,
        copies.len() as u32,
        &mut command,
    );
    submit_command_buffer(command);
    let mut buffers = vec![
        (actual.k_codes.as_ref(), expected.k_codes.as_ref()),
        (actual.k_scales.as_ref(), expected.k_scales.as_ref()),
    ];
    if !keys_only {
        buffers.extend([
            (actual.v_codes.as_ref(), expected.v_codes.as_ref()),
            (actual.v_scales.as_ref(), expected.v_scales.as_ref()),
        ]);
    }
    for (actual, expected) in buffers {
        assert_eq!(
            buffer_to_vec::<B, u8>(&context, &buffer_readback::<B>(&context, actual)),
            buffer_to_vec::<B, u8>(&context, &buffer_readback::<B>(&context, expected)),
        );
    }
}

#[uzu_test]
fn test_kv_copy() {
    for keys_only in [false, true] {
        for_each_backend!(|B| check_kv_copy::<B>(keys_only));
    }
}

#[uzu_test]
fn test_indexer_cache_update() {
    const DIM: usize = 128;
    const DRAFT_ROWS: usize = 16;
    const RATIO: u32 = 4;
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let mut codes: Vec<i8> = (0..2 * DRAFT_ROWS * DIM).map(|i| (i % 251) as u8 as i8).collect();
        let mut scales: Vec<bf16> = (0..2 * DRAFT_ROWS).map(|i| bf16::from_f32(i as f32 + 0.25)).collect();
        let mut unpooled: Vec<bf16> =
            (0..(DRAFT_ROWS + RATIO as usize - 1) * DIM).map(|i| bf16::from_f32(i as f32 / 16.0)).collect();
        let mut block_codes = create_buffer_with_data::<B, i8>(&context, &codes);
        let mut block_scales = create_buffer_with_data::<B, bf16>(&context, &scales);
        let mut unpooled_keys = create_buffer_with_data::<B, bf16>(&context, &unpooled);
        let kernel =
            <<B as Backend>::Kernels as Kernels>::IndexerCacheUpdateKernel::new(&context, DIM as u32, RATIO).unwrap();
        let accepted = [0, 2, 3, 5, 7, 8, 10];
        let mut command = context.create_command_buffer(None, None).unwrap();
        kernel.encode(
            &mut block_codes,
            &mut block_scales,
            &mut unpooled_keys,
            &accepted,
            accepted.len() as u32,
            2,
            DRAFT_ROWS as u32,
            &mut command,
        );
        submit_command_buffer(command);
        codes.copy_within(2 * DIM..3 * DIM, DRAFT_ROWS * DIM);
        codes.copy_within(8 * DIM..9 * DIM, (DRAFT_ROWS + 1) * DIM);
        scales[DRAFT_ROWS] = scales[2];
        scales[DRAFT_ROWS + 1] = scales[8];
        unpooled.copy_within(13 * DIM..14 * DIM, 0);
        assert_eq!(buffer_to_vec::<B, i8>(&context, &block_codes), codes);
        assert_eq!(buffer_to_vec::<B, bf16>(&context, &block_scales), scales);
        assert_eq!(buffer_to_vec::<B, bf16>(&context, &unpooled_keys), unpooled);
    });
}
