use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Cannot open device")]
    CannotOpenDevice,
    #[error("Cannot start gpu capture {0}")]
    CannotStartGpuCapture(String),
    #[error("Cannot create library: {0}")]
    CannotCreateLibrary(String),
    #[error("Cannot create command queue")]
    CannotCreateCommandQueue,
    #[error("Cannot create command Metal 4 queue")]
    CannotCreateCommandQueueMtl4,
    #[error("Cannot create buffer")]
    CannotCreateBuffer,
    #[error("Cannot create command buffer")]
    CannotCreateCommandBuffer,
    #[error("Command buffer execution failed: {0}")]
    CommandBufferExecutionFailed(String),
    #[error("Cannot create event")]
    CannotCreateEvent,
    #[error("Cannot create function")]
    CannotCreateFunction,
    #[error("Cannot create pipeline state: {0}")]
    CannotCreatePipelineState(String),
    #[error("Can not allocate buffer with size={0}")]
    SparseBufferAlloc(usize),
    #[error("Can not allocate heap with size={0} nad page size={1}")]
    SparseHeapAlloc(usize, usize),
    #[error("Sparse buffer requires Metal 4 queue")]
    SparseRequireMtl4Queue,
}
