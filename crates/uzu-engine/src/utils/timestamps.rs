use std::{mem, sync::mpsc::Sender, thread::JoinHandle, time::Instant};

use crate::backends::common::{
    Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferPending, TimestampSource,
    TimestampSpan,
};

pub struct TimestampSender {
    // Always `Some` except on drop
    sender: Option<Sender<Box<[TimestampSpan]>>>,
    consumer: Option<JoinHandle<()>>,
}

impl TimestampSender {
    pub fn with_consumer(
        sender: Sender<Box<[TimestampSpan]>>,
        consumer: JoinHandle<()>,
    ) -> Self {
        Self {
            sender: Some(sender),
            consumer: Some(consumer),
        }
    }

    fn sender(&self) -> &Sender<Box<[TimestampSpan]>> {
        self.sender.as_ref().unwrap()
    }
}

impl From<Sender<Box<[TimestampSpan]>>> for TimestampSender {
    fn from(sender: Sender<Box<[TimestampSpan]>>) -> Self {
        Self {
            sender: Some(sender),
            consumer: None,
        }
    }
}

impl Drop for TimestampSender {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(consumer) = self.consumer.take() {
            let _ = consumer.join();
        }
    }
}

pub fn record<Encoding: CommandBufferEncoding>(
    command_buffer: Encoding,
    timestamps: Option<&TimestampSender>,
) -> Encoding {
    match timestamps {
        Some(_) => command_buffer.with_timestamps(),
        None => command_buffer,
    }
}

pub async fn wait<Pending: CommandBufferPending>(
    pending: Pending,
    timestamps: Option<&TimestampSender>,
) -> Result<
    <Pending::CommandBuffer as CommandBuffer>::Completed,
    <<Pending::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
> {
    let completed = pending.wait_until_completed().await?;
    if let Some(timestamps) = timestamps {
        let _ = timestamps.sender().send(completed.timestamps().into());
    }
    Ok(completed)
}

pub struct CpuSpan<'a> {
    timestamps: &'a TimestampSender,
    name: String,
    start: Instant,
}

impl<'a> CpuSpan<'a> {
    #[must_use = "the span ends when dropped"]
    pub fn start(
        timestamps: Option<&'a TimestampSender>,
        name: &str,
    ) -> Option<Self> {
        Some(Self {
            timestamps: timestamps?,
            name: name.to_owned(),
            start: Instant::now(),
        })
    }
}

impl Drop for CpuSpan<'_> {
    fn drop(&mut self) {
        let _ = self.timestamps.sender().send(Box::new([TimestampSpan {
            name: mem::take(&mut self.name),
            source: TimestampSource::Cpu,
            start: self.start,
            end: Instant::now(),
        }]));
    }
}
