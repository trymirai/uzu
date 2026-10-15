use std::sync::mpsc::Sender;

use crate::backends::common::{Backend, CommandBuffer, CommandBufferCompleted, CommandBufferPending, TimestampSpan};

pub fn wait<Pending: CommandBufferPending>(
    pending: Pending,
    timestamps: Option<&Sender<Box<[TimestampSpan]>>>,
) -> Result<
    <Pending::CommandBuffer as CommandBuffer>::Completed,
    <<Pending::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
> {
    let completed = pending.wait_until_completed()?;
    if let Some(sender) = timestamps {
        let _ = sender.send(completed.timestamps().into());
    }
    Ok(completed)
}
