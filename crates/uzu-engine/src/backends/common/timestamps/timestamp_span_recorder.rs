use std::time::Instant;

use super::{TimestampSpan, span_boundary::SpanBoundary};

#[derive(Default)]
pub struct TimestampSpanRecorder {
    boundaries: Vec<SpanBoundary>,
}

impl TimestampSpanRecorder {
    pub fn start(
        &mut self,
        name: String,
    ) -> usize {
        self.push(SpanBoundary::Start(name))
    }

    pub fn end(&mut self) -> usize {
        self.push(SpanBoundary::End)
    }

    fn push(
        &mut self,
        boundary: SpanBoundary,
    ) -> usize {
        self.boundaries.push(boundary);
        self.boundaries.len() - 1
    }

    pub fn slot_count(&self) -> usize {
        self.boundaries.len()
    }

    pub fn spans(
        &self,
        instants: &[Instant],
    ) -> Box<[TimestampSpan]> {
        let mut spans = Vec::new();
        let mut open_span_indices = Vec::new();
        for (boundary, &instant) in self.boundaries.iter().zip(instants) {
            match boundary {
                SpanBoundary::Start(name) => {
                    open_span_indices.push(spans.len());
                    spans.push(TimestampSpan {
                        name: name.clone(),
                        start: instant,
                        end: instant,
                    });
                },
                SpanBoundary::End => {
                    let index = open_span_indices.pop().expect("timestamp ended without a start");
                    spans[index].end = instant;
                },
            }
        }
        assert!(open_span_indices.is_empty(), "timestamps were started but never ended");
        spans.into_boxed_slice()
    }
}
