use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, TimestampSpan,
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const GROUPS: [&str; 3] = ["decoder", "decoder/layer 0", "decoder/layer 1"];
/// Copies after the fill: enough commands for Vulkan to submit the recording in three segments.
const LONG_COPIES: usize = 99;

fn fill<B: Backend>(command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding) {
    let mut buffer = command_buffer.allocate_scratch(256).unwrap();
    command_buffer.encode_fill(&mut buffer, 7);
}

fn record<B: Backend>(mut command_buffer: <B::CommandBuffer as CommandBuffer>::Encoding) -> Box<[TimestampSpan]> {
    {
        let command_buffer = &mut *command_buffer.debug_group("decoder");
        fill::<B>(command_buffer);
        for layer in ["layer 0", "layer 1"] {
            let command_buffer = &mut *command_buffer.debug_group(layer);
            fill::<B>(command_buffer);
        }
    }
    let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    completed.timestamps().into()
}

fn assert_nested(spans: &[TimestampSpan]) {
    for span in spans {
        assert!(span.start <= span.end, "{} ends before it starts", span.name);
        if let Some((parent_name, _)) = span.name.rsplit_once('/') {
            let parent = spans.iter().find(|parent| parent.name == parent_name).unwrap();
            assert!(parent.start <= span.start && span.end <= parent.end, "{} is outside {parent_name}", span.name);
        }
    }
}

#[uzu_test]
fn debug_groups_time_their_full_path() {
    for_each_non_cpu_backend!(|B| {
        let context = create_context::<B>();
        let spans = record::<B>(context.create_command_buffer(None, None).unwrap().with_timestamps());
        // A backend may also time the kernels it encodes, as children of the group they run in.
        let groups =
            spans.iter().map(|span| span.name.as_str()).filter(|name| GROUPS.contains(name)).collect::<Vec<_>>();
        assert_eq!(groups, GROUPS);
        assert_nested(&spans);
    });
}

// Metal's generated kernels and its fill are debug groups too, so its spans reach each kernel.
#[cfg(backend = "metal")]
#[uzu_test]
fn debug_groups_time_their_full_path_down_to_kernels() {
    type B = crate::backends::metal::Metal;
    let context = create_context::<B>();
    let spans = record::<B>(context.create_command_buffer(None, None).unwrap().with_timestamps());
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
}

/// A fill, then copies back and forth in a debug group each: a recording long enough to be submitted in parts keeps
/// its commands in order and every span within its parent.
#[uzu_test]
fn long_recordings_keep_order_and_spans() {
    for_each_non_cpu_backend!(|B| {
        let context = create_context::<B>();
        let mut a = create_buffer_with_data::<B, u8>(&context, &[0; 256]);
        let mut b = create_buffer_with_data::<B, u8>(&context, &[1; 256]);
        let mut command_buffer = context.create_command_buffer(None, None).unwrap().with_timestamps();
        {
            let command_buffer = &mut *command_buffer.debug_group("decoder");
            command_buffer.encode_fill(&mut a, 7);
            for copy in 0..LONG_COPIES {
                let command_buffer = &mut *command_buffer.debug_group(&format!("layer {copy}"));
                if copy % 2 == 0 {
                    command_buffer.encode_copy(&a, &mut b);
                } else {
                    command_buffer.encode_copy(&b, &mut a);
                }
            }
        }
        let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        let spans = completed.timestamps();
        let mut layers = spans
            .iter()
            .filter_map(|span| Some((span.name.strip_prefix("decoder/layer ")?.parse::<usize>().ok()?, span.start)))
            .collect::<Vec<_>>();
        layers.sort_unstable();
        assert_eq!(layers.len(), LONG_COPIES);
        assert!(layers.windows(2).all(|pair| pair[0].1 <= pair[1].1), "a later copy starts before an earlier one");
        assert_nested(spans);
        assert_eq!(buffer_to_vec::<B, u8>(&context, &a), [7; 256]);
        assert_eq!(buffer_to_vec::<B, u8>(&context, &b), [7; 256]);
    });
}

#[uzu_test]
fn only_command_buffers_asked_for_timestamps_record_them() {
    for_each_non_cpu_backend!(|B| {
        let context = create_context::<B>();
        assert!(
            record::<B>(context.create_command_buffer(None, None).unwrap().with_timestamps()).len() >= GROUPS.len()
        );
        assert!(record::<B>(context.create_command_buffer(None, None).unwrap()).is_empty());
    });
}
