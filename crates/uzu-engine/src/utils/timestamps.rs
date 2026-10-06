use std::{
    sync::{Arc, mpsc::Sender},
    time::Instant,
};

use crate::backends::common::{
    Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferPending, Context,
};

pub fn create_command_buffer<B: Backend>(
    context: &B::Context,
    name: &str,
    allocation_pool: &Arc<B::AllocationPool>,
    timestamps: Option<&Sender<Box<[(String, Instant)]>>>,
) -> Result<<B::CommandBuffer as CommandBuffer>::Encoding, B::Error> {
    let mut command_buffer = context.create_command_buffer(Some(name), Some(allocation_pool.clone()))?;
    if timestamps.is_some() {
        command_buffer.enable_timestamps()?;
        command_buffer.sample_timestamp(&format!("{name}/start"));
    }
    Ok(command_buffer)
}

pub fn wait<Pending: CommandBufferPending>(
    pending: Pending,
    timestamps: Option<&Sender<Box<[(String, Instant)]>>>,
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
