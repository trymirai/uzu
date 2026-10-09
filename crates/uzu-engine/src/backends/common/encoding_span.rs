use std::ops::{Deref, DerefMut};

use super::CommandBufferEncoding;

/// A named scope of encoded work; dropping it closes the span.
#[must_use = "keep the span alive while encoding its work"]
pub struct EncodingSpan<'a, C: CommandBufferEncoding> {
    encoding: &'a mut C,
}

impl<'a, C: CommandBufferEncoding> EncodingSpan<'a, C> {
    pub(super) fn new(
        encoding: &'a mut C,
        label: impl std::fmt::Display,
    ) -> Self {
        encoding.begin_span(label);
        Self {
            encoding,
        }
    }
}

impl<C: CommandBufferEncoding> Deref for EncodingSpan<'_, C> {
    type Target = C;

    fn deref(&self) -> &C {
        self.encoding
    }
}

impl<C: CommandBufferEncoding> DerefMut for EncodingSpan<'_, C> {
    fn deref_mut(&mut self) -> &mut C {
        self.encoding
    }
}

impl<C: CommandBufferEncoding> Drop for EncodingSpan<'_, C> {
    fn drop(&mut self) {
        self.encoding.end_span();
    }
}

#[cfg(all(test, backend = "cpu"))]
mod tests {
    use uzu_engine_macros::uzu_test;

    use crate::backends::{
        common::{
            Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferPending, Context,
        },
        cpu::Cpu,
    };

    type CpuEncoding = <<Cpu as Backend>::CommandBuffer as CommandBuffer>::Encoding;

    fn interrupted_work(encoding: &mut CpuEncoding) -> Result<(), ()> {
        let mut span = encoding.span("interrupted");
        let _nested = span.span("nested");
        Err(())?;
        Ok(())
    }

    #[uzu_test]
    fn encoding_span_nested_paths_and_early_return() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let mut encoding = context.create_command_buffer(None, None).unwrap().enable_timestamps();
        {
            let mut decoder = encoding.span("decoder");
            {
                let mut layer = decoder.span(format_args!("layer {}", 3));
                let _attention = layer.span("attention");
            }
            assert!(interrupted_work(&mut decoder).is_err());
            let _sampling = decoder.span("sampling");
        }
        let completed = encoding.end_encoding().submit().wait_until_completed().unwrap();
        let spans = completed.timestamps();
        assert_eq!(
            spans.iter().map(|span| span.name.as_str()).collect::<Vec<_>>(),
            [
                "decoder",
                "decoder/layer 3",
                "decoder/layer 3/attention",
                "decoder/interrupted",
                "decoder/interrupted/nested",
                "decoder/sampling",
            ]
        );
        assert!(spans.iter().all(|span| span.start <= span.end));
        for span in &spans[1..] {
            assert!(spans[0].start <= span.start && span.end <= spans[0].end);
        }
    }

    #[uzu_test]
    fn encoding_span_unwind_and_disabled_recording() {
        let context = <Cpu as Backend>::Context::new().unwrap();
        let mut encoding = context.create_command_buffer(None, None).unwrap().enable_timestamps();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut outer = encoding.span("outer");
            let _inner = outer.span("inner");
            panic!("interrupted encoding");
        }));
        assert!(result.is_err());
        {
            let _sibling = encoding.span("sibling");
        }
        let completed = encoding.end_encoding().submit().wait_until_completed().unwrap();
        assert_eq!(
            completed.timestamps().iter().map(|span| span.name.as_str()).collect::<Vec<_>>(),
            ["outer", "outer/inner", "sibling",]
        );

        let mut encoding = context.create_command_buffer(None, None).unwrap();
        {
            let mut outer = encoding.span("outer");
            let _inner = outer.span("inner");
        }
        let completed = encoding.end_encoding().submit().wait_until_completed().unwrap();
        assert!(completed.timestamps().is_empty());
    }
}
