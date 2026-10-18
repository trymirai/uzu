use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{BufferRef, Context},
        webgpu::{WebGPU, context::WebGPUContext},
    },
    tests::helpers::create_buffer_with_data,
};

#[uzu_test]
fn copyout_preserves_unaligned_ranges() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let data: Vec<u8> = (0..11).collect();
    let buffer = create_buffer_with_data::<WebGPU, u8>(&context, &data);
    for start in 0..data.len() {
        for end in start + 1..=data.len() {
            let actual = block_on(context.copyout::<u8>((&buffer).subrange(start..end))).unwrap();
            assert_eq!(actual, data[start..end], "range {start}..{end}");
        }
    }
}
