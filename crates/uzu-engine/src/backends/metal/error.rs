use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Cannot open device")]
    CannotOpenDevice,
    #[error("Cannot create residency set: {0}")]
    CannotCreateResidencySet(String),
    #[error("Cannot start GPU capture: {0}")]
    CannotStartGpuCapture(String),
    #[error("Cannot create library: {0}")]
    CannotCreateLibrary(String),
    #[error("Cannot create command queue")]
    CannotCreateCommandQueue,
    #[error("Cannot create event")]
    CannotCreateEvent,
    #[error("Cannot create buffer")]
    CannotCreateBuffer,
    #[error("Cannot create heap")]
    CannotCreateHeap,
    #[error("Cannot create command buffer")]
    CannotCreateCommandBuffer,
    #[error("Cannot create argument table: {0}")]
    CannotCreateArgumentTable(String),
    #[error("Error waiting for command buffer")]
    CommandBufferWait,
    #[error("Command buffer execution failed: {0}")]
    CommandBufferExecution(String),
    #[error("Cannot create function: {0}")]
    CannotCreateFunction(String),
    #[error("Cannot create pipeline state for {function_name}: {error}")]
    CannotCreatePipelineState {
        function_name: String,
        error: String,
    },
}
