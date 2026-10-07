mod span_boundary;
mod timestamp_span;
mod timestamp_span_recorder;

pub use timestamp_span::TimestampSpan;
pub use timestamp_span_recorder::TimestampSpanRecorder;

pub type BlockName = String;
pub type TimestampSlot = usize;
pub type CommandBufferTimestamps = Box<[TimestampSpan]>;
