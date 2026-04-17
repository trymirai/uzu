use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("failed to create command buffer")]
    CommandBuffer,
    #[error("kernel creation failed: {0}")]
    Kernel(String),
    #[error("buffer allocation failed")]
    BufferAllocation,
    #[error("warmup iteration {iteration}: {source}")]
    Warmup {
        iteration: usize,
        source: Box<BenchError>,
    },
    #[error("benchmark iteration {iteration}: {source}")]
    Benchmark {
        iteration: usize,
        source: Box<BenchError>,
    },
}
