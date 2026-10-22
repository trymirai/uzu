use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferPending, Context, TimestampSpan,
        },
        metal::Metal,
    },
    tests::helpers::create_context,
};

type Encoding = <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding;

fn fill(command_buffer: &mut Encoding) {
    let mut buffer = command_buffer.allocate_scratch(256).unwrap();
    command_buffer.encode_fill(&mut buffer, 7);
}

fn record(mut command_buffer: Encoding) -> Box<[TimestampSpan]> {
    {
        let command_buffer = &mut *command_buffer.debug_group("decoder");
        fill(command_buffer);
        for layer in ["layer 0", "layer 1"] {
            let command_buffer = &mut *command_buffer.debug_group(layer);
            fill(command_buffer);
        }
    }
    let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    completed.timestamps().into()
}

#[uzu_test]
fn debug_groups_time_their_full_path_down_to_kernels() {
    let context = create_context::<Metal>();
    let spans = record(context.create_command_buffer(None, None).unwrap().with_timestamps());
    assert_eq!(
        spans.iter().map(|span| span.name.as_str()).collect::<Vec<_>>(),
        [
            "decoder",
            "decoder/BlitFill",
            "decoder/layer 0",
            "decoder/layer 0/BlitFill",
            "decoder/layer 1",
            "decoder/layer 1/BlitFill"
        ]
    );
    for span in &spans {
        assert!(span.start <= span.end, "{} ends before it starts", span.name);
        if let Some((parent_name, _)) = span.name.rsplit_once('/') {
            let parent = spans.iter().find(|parent| parent.name == parent_name).unwrap();
            assert!(parent.start <= span.start && span.end <= parent.end, "{} is outside {parent_name}", span.name);
        }
    }
}

#[uzu_test]
fn only_command_buffers_asked_for_timestamps_record_them() {
    let context = create_context::<Metal>();
    assert_eq!(record(context.create_command_buffer(None, None).unwrap().with_timestamps()).len(), 6);
    assert!(record(context.create_command_buffer(None, None).unwrap()).is_empty());
}
