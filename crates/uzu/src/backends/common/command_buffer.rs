use super::Backend;

pub trait CommandBuffer {
    type Backend: Backend;

    fn with_compute_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::ComputeEncoder) -> T,
    ) -> T;

    fn with_copy_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::CopyEncoder) -> T,
    ) -> T;

    fn submit(&self);
    fn wait_until_completed(&self);

    fn gpu_execution_time_ms(&self) -> Option<f64>;
}
