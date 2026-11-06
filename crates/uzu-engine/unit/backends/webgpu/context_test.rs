use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
        webgpu::{WebGPU, context::WebGPUContext, error::WebGPUError},
    },
    tests::helpers::create_buffer_with_data,
};

#[uzu_test]
fn device_error_is_returned_once_by_the_next_wait() {
    let context = block_on(WebGPUContext::new()).unwrap();
    let wait = || {
        let command_buffer = context.create_command_buffer(None, None).unwrap();
        block_on(command_buffer.end_encoding().submit().wait_until_completed())
    };

    let _ = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl("not wgsl".into()),
    });

    assert!(matches!(wait(), Err(WebGPUError::Device(_))));
    assert!(wait().is_ok());
}

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
