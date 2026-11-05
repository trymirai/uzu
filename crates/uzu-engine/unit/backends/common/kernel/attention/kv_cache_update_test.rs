use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, Context, Kernels,
        gpu_types::kv_cache_update::Copy,
        kernel::{AttentionPrepareKernel, KVCacheUpdateKernel, kv_layout::KVLayout},
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
        DataType::F32,
        HEAD_DIM,
        false,
        NUM_HEADS,
        layout.backend_opaque_layout,
        true,
        false,
        false,
        false,
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
        Default::default(),
        0,
        None,
        0,
        ROW_WIDTH as u32,
        KEYS,
        0,
        None::<&B::GlobalGpuBuffer>,
        &mut command,
    );
    submit_command_buffer(command);
    kv
}

fn check_kv_copy<B: Backend>() {
    let context = create_context::<B>();
    let layout = kv_layout::<B>(&context, HEAD_DIM, NUM_HEADS);
    let mut rows: Vec<bf16> =
        (0..KEYS as usize * ROW_WIDTH).map(|i| bf16::from_f32(((i * 37 % 251) as f32 - 125.0) / 128.0)).collect();
    let mut actual = prepare_cache::<B>(&context, layout, &rows);
    rows.copy_within(40 * ROW_WIDTH..41 * ROW_WIDTH, 5 * ROW_WIDTH);
    let expected = prepare_cache::<B>(&context, layout, &rows);
    let kernel =
        <B::Kernels as Kernels>::KVCacheUpdateKernel::new(&context, HEAD_DIM, NUM_HEADS, layout.backend_opaque_layout)
            .unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        actual.k_codes.as_mut(),
        actual.v_codes.as_mut(),
        actual.k_scales.as_mut(),
        actual.v_scales.as_mut(),
        &[Copy {
            source: 40,
            destination: 5,
        }],
        1,
        &mut command,
    );
    submit_command_buffer(command);
    for (actual, expected) in [
        (actual.k_codes.as_ref(), expected.k_codes.as_ref()),
        (actual.v_codes.as_ref(), expected.v_codes.as_ref()),
        (actual.k_scales.as_ref(), expected.k_scales.as_ref()),
        (actual.v_scales.as_ref(), expected.v_scales.as_ref()),
    ] {
        assert_eq!(
            buffer_to_vec::<B, u8>(&context, &buffer_readback::<B>(&context, actual)),
            buffer_to_vec::<B, u8>(&context, &buffer_readback::<B>(&context, expected)),
        );
    }
}

#[uzu_test]
fn test_kv_copy() {
    for_each_backend!(|B| check_kv_copy::<B>());
}
