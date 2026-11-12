use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

#[cfg(backend = "metal")]
use crate::backends::metal::Metal;
use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, TimestampSpan,
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const GROUPS: [&str; 3] = ["decoder", "decoder/layer 0", "decoder/layer 1"];
const LONG_COPIES: usize = 99;

fn record<B: Backend>(mut command_buffer: <B::CommandBuffer as CommandBuffer>::Encoding) -> Box<[TimestampSpan]> {
    {
        let command_buffer = &mut *command_buffer.debug_group("decoder");
        let mut buffer = command_buffer.allocate_scratch(256).unwrap();
        command_buffer.encode_fill(&mut buffer, 7);
        for layer in ["layer 0", "layer 1"] {
            let command_buffer = &mut *command_buffer.debug_group(layer);
            let mut buffer = command_buffer.allocate_scratch(256).unwrap();
            command_buffer.encode_fill(&mut buffer, 7);
        }
    }
    let completed = block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    completed.timestamps().into()
}

// Metal's generated kernels and its fill are debug groups too, so its spans reach each kernel.
#[cfg(backend = "metal")]
#[uzu_test]
fn debug_groups_time_their_full_path_down_to_kernels() {
    let context = create_context::<Metal>();
    let spans = record::<Metal>(context.create_command_buffer(None, None).unwrap().with_timestamps());
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
        for span in spans {
            assert!(span.start <= span.end, "{} ends before it starts", span.name);
            if let Some((parent_name, _)) = span.name.rsplit_once('/') {
                let parent = spans.iter().find(|parent| parent.name == parent_name).unwrap();
                assert!(parent.start <= span.start && span.end <= parent.end, "{} is outside {parent_name}", span.name);
            }
        }
        assert_eq!(buffer_to_vec::<B, u8>(&context, &a), [7; 256]);
        assert_eq!(buffer_to_vec::<B, u8>(&context, &b), [7; 256]);
    });
}

#[uzu_test]
fn only_command_buffers_asked_for_timestamps_record_them() {
    for_each_non_cpu_backend!(|B| {
        let context = create_context::<B>();
        let spans = record::<B>(context.create_command_buffer(None, None).unwrap().with_timestamps());
        let groups =
            spans.iter().map(|span| span.name.as_str()).filter(|name| GROUPS.contains(name)).collect::<Vec<_>>();
        assert_eq!(groups, GROUPS);
        for span in &spans {
            assert!(span.start <= span.end, "{} ends before it starts", span.name);
            if let Some((parent_name, _)) = span.name.rsplit_once('/') {
                let parent = spans.iter().find(|parent| parent.name == parent_name).unwrap();
                assert!(parent.start <= span.start && span.end <= parent.end, "{} is outside {parent_name}", span.name);
            }
        }
        assert!(record::<B>(context.create_command_buffer(None, None).unwrap()).is_empty());
    });
}
