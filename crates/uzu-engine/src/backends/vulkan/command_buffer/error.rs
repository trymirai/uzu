use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkCommandBufferError {
    #[error("the completion thread stopped before the command buffer completed")]
    Wait,
    #[error("buffer range {start}..{end} is reversed or exceeds a buffer of {size} bytes")]
    BufferRange {
        start: u64,
        end: u64,
        size: u64,
    },
    #[error("copy ranges overlap within the same buffer")]
    CopyOverlap,
    #[error("fill range {start}..{end} must be 4-byte aligned")]
    FillAlignment {
        start: u64,
        end: u64,
    },
    #[error("timestamps need VK_KHR_calibrated_timestamps with the device time domain to read the device clock")]
    UncalibratedTimestamps,
    #[error("cannot read the device clock: {0}")]
    DeviceClock(#[source] vk::Result),
    #[error("a timestamp {elapsed:?} before the device clock reading precedes what the host clock represents")]
    UnrepresentableTimestamp {
        elapsed: std::time::Duration,
    },
    #[error(
        "the GPU never wrote timestamp {query} of a completed submission, so it did not run it; a driver watchdog \
         resetting work it could not preempt is the usual cause"
    )]
    UnexecutedSubmission {
        query: u32,
    },
    #[error("dispatch of {groups:?} groups exceeds the device limit {limit:?}")]
    DispatchGroups {
        groups: [u32; 3],
        limit: [u32; 3],
    },
}
