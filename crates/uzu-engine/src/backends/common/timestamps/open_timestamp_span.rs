use super::{BlockName, TimestampSlot, TimestampSpan};

pub struct OpenTimestampSpan {
    pub name: BlockName,
    pub start: TimestampSlot,
}

impl OpenTimestampSpan {
    pub fn close(
        self,
        end: TimestampSlot,
    ) -> TimestampSpan<TimestampSlot> {
        TimestampSpan {
            name: self.name,
            start: self.start,
            end,
        }
    }
}
