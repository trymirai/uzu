use crate::backends::{
    common::TimestampSpan,
    vulkan::{Error, VkCommandBufferError, VkContext},
};

#[derive(Default)]
pub struct VkTimestampRecorder {
    spans: Vec<(String, [(usize, u32); 2])>,
    open_spans: Vec<usize>,
}

impl VkTimestampRecorder {
    pub fn start(
        &mut self,
        label: &str,
        timestamp: (usize, u32),
    ) {
        let name = match self.open_spans.last() {
            Some(&parent) => format!("{}/{label}", self.spans[parent].0),
            None => label.to_owned(),
        };
        self.open_spans.push(self.spans.len());
        self.spans.push((name, [timestamp; 2]));
    }

    pub fn end(
        &mut self,
        timestamp: (usize, u32),
    ) {
        let span = self.open_spans.pop().expect("timestamp span ended without a start");
        self.spans[span].1[1] = timestamp;
    }

    pub fn resolve(
        &self,
        context: &VkContext,
        segment_timestamps: &[Box<[u64]>],
    ) -> Result<Box<[TimestampSpan]>, Error> {
        let (clock, now) = context.device_clock()?;
        let instant = |(segment, index): (usize, u32)| {
            let elapsed = context.timestamp_duration(clock.wrapping_sub(segment_timestamps[segment][index as usize]));
            now.checked_sub(elapsed).ok_or(VkCommandBufferError::UnrepresentableTimestamp {
                elapsed,
            })
        };
        self.spans
            .iter()
            .map(|(name, [start, end])| {
                Ok(TimestampSpan {
                    name: name.clone(),
                    start: instant(*start)?,
                    end: instant(*end)?,
                })
            })
            .collect()
    }
}
