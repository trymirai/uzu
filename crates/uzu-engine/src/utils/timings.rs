use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferPending,
        CommandBufferTiming,
    },
    engine::language_model::stream::Timings,
};

pub fn record<Encoding: CommandBufferEncoding>(
    command_buffer: Encoding,
    timings: Option<&Timings>,
) -> Encoding {
    match timings {
        Some(timings) if timings.spans => command_buffer.with_timestamps(),
        _ => command_buffer,
    }
}

pub async fn wait<Pending: CommandBufferPending>(
    pending: Pending,
    timings: Option<&Timings>,
) -> Result<
    <Pending::CommandBuffer as CommandBuffer>::Completed,
    <<Pending::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
> {
    let completed = pending.wait_until_completed().await?;
    if let Some(timings) = timings {
        let _ = timings.sender.send(CommandBufferTiming {
            gpu_time: completed.gpu_execution_time(),
            spans: completed.timestamps().into(),
        });
    }
    Ok(completed)
}
