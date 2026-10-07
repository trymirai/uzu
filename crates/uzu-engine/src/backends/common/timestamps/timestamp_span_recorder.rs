use super::{BlockName, TimestampSlot, TimestampSpan, open_timestamp_span::OpenTimestampSpan};

#[derive(Default)]
pub struct TimestampSpanRecorder {
    open: Vec<OpenTimestampSpan>,
    closed: Vec<TimestampSpan<TimestampSlot>>,
}

impl TimestampSpanRecorder {
    fn next_slot(&self) -> TimestampSlot {
        self.open.len() + 2 * self.closed.len()
    }

    pub fn start(
        &mut self,
        name: BlockName,
    ) -> TimestampSlot {
        let start = self.next_slot();
        self.open.push(OpenTimestampSpan {
            name,
            start,
        });
        start
    }

    pub fn end(&mut self) -> TimestampSlot {
        let end = self.next_slot();
        let open = self.open.pop().expect("timestamp ended without a start");
        self.closed.push(open.close(end));
        end
    }

    pub fn finish(self) -> Box<[TimestampSpan<TimestampSlot>]> {
        assert!(self.open.is_empty(), "{} timestamps were started but never ended", self.open.len());
        let mut closed = self.closed;
        closed.sort_unstable_by_key(|span| span.start);
        closed.into_boxed_slice()
    }
}
