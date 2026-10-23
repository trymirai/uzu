use std::time::Duration;

use crate::backends::{
    common::TimestampSpan,
    vulkan::{Error, VkCommandBufferError, VkContext},
};

/// The debug-group spans of a command buffer that records timestamps: `parent/child` names and, for each end, the
/// segment that wrote its span timestamp and the timestamp's index there.
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

    /// The spans in host time, from the span timestamps of each completed segment, and their uncertainty. A timestamp
    /// is one device clock reading after completion, less the device time since the timestamp, so every instant shares
    /// that reading's uncertainty while their differences are exact to a device tick.
    pub fn resolve(
        &self,
        context: &VkContext,
        segment_timestamps: &[Box<[u64]>],
    ) -> Result<(Box<[TimestampSpan]>, Duration), Error> {
        let (clock, now, uncertainty) = context.device_clock()?;
        let instant = |(segment, index): (usize, u32)| {
            let elapsed = context.timestamp_duration(clock.wrapping_sub(segment_timestamps[segment][index as usize]));
            now.checked_sub(elapsed).ok_or(VkCommandBufferError::UnrepresentableTimestamp {
                elapsed,
            })
        };
        let spans = self
            .spans
            .iter()
            .map(|(name, [start, end])| {
                Ok(TimestampSpan {
                    name: name.clone(),
                    start: instant(*start)?,
                    end: instant(*end)?,
                })
            })
            .collect::<Result<_, Error>>()?;
        Ok((spans, uncertainty))
    }
}
