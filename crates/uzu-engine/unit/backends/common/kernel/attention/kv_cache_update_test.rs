use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::kv_cache_update::Copy,
            kernel::{KVCacheUpdateKernel, kv_layout::Int8KVLayout},
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_non_cpu_backend},
};

#[uzu_test]
fn test_int8_copy() {
    for use_mxu in [false, true] {
        let expected = get_int8_output::<Cpu>(use_mxu);
        for_each_non_cpu_backend!(|B| {
            assert_eq!(
                expected,
                get_int8_output::<B>(use_mxu),
                "INT8 cache update mismatch on {}",
                std::any::type_name::<B>()
            );
        });
    }
}

fn get_int8_output<B: Backend>(use_mxu: bool) -> ([Vec<u8>; 2], [Vec<bf16>; 2]) {
    let layout = Int8KVLayout::new(256, 2, use_mxu);
    let scale_count = 64 * layout.num_kv_heads as usize;
    let code_bytes = scale_count * layout.head_dim as usize;
    let k_codes: Vec<i8> = (0..code_bytes).map(|i| ((i * 37) ^ (i >> 8)) as i8).collect();
    let v_codes: Vec<u8> = (0..code_bytes).map(|i| ((i * 19) ^ (i >> 7)) as u8).collect();
    let k_scales: Vec<bf16> = (0..scale_count).map(|i| bf16::from_f32(i as f32 + 1.0)).collect();
    let v_scales: Vec<bf16> = (0..scale_count).map(|i| bf16::from_f32(i as f32 + 129.0)).collect();
    let copies = [Copy {
        source: 40,
        destination: 5,
    }];
    let context = block_on(B::Context::new()).unwrap();
    let mut keys = create_buffer_with_data::<B, i8>(&context, &k_codes);
    let mut values = create_buffer_with_data::<B, u8>(&context, &v_codes);
    let mut key_scales = create_buffer_with_data::<B, bf16>(&context, &k_scales);
    let mut value_scales = create_buffer_with_data::<B, bf16>(&context, &v_scales);
    let kernel =
        <B::Kernels as Kernels>::KVCacheUpdateKernel::new(&context, layout.head_dim, layout.num_kv_heads, use_mxu)
            .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &mut keys,
        &mut values,
        &mut key_scales,
        &mut value_scales,
        &copies,
        copies.len() as u32,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (
        [&keys, &values].map(|buffer| buffer_to_vec::<B, u8>(&context, buffer)),
        [&key_scales, &value_scales].map(|buffer| buffer_to_vec::<B, bf16>(&context, buffer)),
    )
}
