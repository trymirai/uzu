use std::{sync::mpsc::Sender, time::Instant};

use crate::backends::common::{
    Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferPending, TimestampSource,
    TimestampSpan,
};

pub fn record<Encoding: CommandBufferEncoding>(
    command_buffer: Encoding,
    timestamps: Option<&Sender<Box<[TimestampSpan]>>>,
) -> Encoding {
    match timestamps {
        Some(_) => command_buffer.with_timestamps(),
        None => command_buffer,
    }
}

pub async fn wait<Pending: CommandBufferPending>(
    pending: Pending,
    timestamps: Option<&Sender<Box<[TimestampSpan]>>>,
) -> Result<
    <Pending::CommandBuffer as CommandBuffer>::Completed,
    <<Pending::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
> {
    let completed = pending.wait_until_completed().await?;
    if let Some(sender) = timestamps {
        let _ = sender.send(completed.timestamps().into());
    }
    Ok(completed)
}

#[must_use = "the span ends when dropped"]
pub struct CpuSpan {
    recording: Option<(Sender<Box<[TimestampSpan]>>, String, Instant)>,
}

impl CpuSpan {
    pub fn start(
        timestamps: Option<&Sender<Box<[TimestampSpan]>>>,
        name: &str,
    ) -> Self {
        Self {
            recording: timestamps.map(|sender| (sender.clone(), name.to_owned(), Instant::now())),
        }
    }
}

impl Drop for CpuSpan {
    fn drop(&mut self) {
        if let Some((sender, name, start)) = self.recording.take() {
            let _ = sender.send(Box::new([TimestampSpan {
                name,
                source: TimestampSource::Cpu,
                start,
                end: Instant::now(),
            }]));
        }
    }
}
