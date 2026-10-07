use super::BlockName;

#[derive(Debug, Clone)]
pub struct TimestampSpan<Time> {
    pub name: BlockName,
    pub start: Time,
    pub end: Time,
}

impl<Time> TimestampSpan<Time> {
    pub fn map<Mapped>(
        self,
        convert: impl Fn(Time) -> Mapped,
    ) -> TimestampSpan<Mapped> {
        TimestampSpan {
            name: self.name,
            start: convert(self.start),
            end: convert(self.end),
        }
    }
}
