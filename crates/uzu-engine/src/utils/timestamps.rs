use std::sync::mpsc::Sender;

use crate::backends::common::{
    Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferPending, TimestampSpan,
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
