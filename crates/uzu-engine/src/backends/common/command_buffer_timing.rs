use std::time::Duration;

use crate::backends::common::TimestampSpan;

pub struct CommandBufferTiming {
    pub gpu_time: Option<Duration>,
    pub spans: Box<[TimestampSpan]>,
}
