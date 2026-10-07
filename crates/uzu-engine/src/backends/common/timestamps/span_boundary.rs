use super::BlockName;

pub enum SpanBoundary {
    Start(BlockName),
    End,
}
