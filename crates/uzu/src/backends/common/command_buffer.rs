use super::Backend;

pub trait CommandBuffer: Clone {
    type Backend: Backend;

    fn with_compute_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::ComputeEncoder) -> T,
    ) -> T;

    fn with_copy_encoder<T>(
        &self,
        callback: impl FnOnce(&<Self::Backend as Backend>::CopyEncoder) -> T,
    ) -> T;

    fn encode_wait_for_event(
        &self,
        event: &<Self::Backend as Backend>::Event,
        value: u64,
    );

    fn encode_signal_event(
        &self,
        event: &<Self::Backend as Backend>::Event,
        value: u64,
    );

    fn add_completed_handler(
        &self,
        handler: impl Fn() + 'static,
    );

    fn submit(&self);
    fn wait_until_completed(&self);

    fn gpu_execution_time_ms(&self) -> Option<f64>;
}
