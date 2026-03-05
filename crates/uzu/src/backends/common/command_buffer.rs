use super::Backend;

pub trait CommandBuffer {
    type Backend: Backend;

    fn with_compute_encoder<T>(
        &mut self,
        callback: impl FnOnce(&mut <Self::Backend as Backend>::ComputeEncoder) -> T,
    ) -> T;

    fn with_copy_encoder<T>(
        &mut self,
        callback: impl FnOnce(&mut <Self::Backend as Backend>::CopyEncoder) -> T,
    ) -> T;

    fn encode_wait_for_event(
        &mut self,
        event: &<Self::Backend as Backend>::Event,
        value: u64,
    );

    fn encode_signal_event(
        &mut self,
        event: &<Self::Backend as Backend>::Event,
        value: u64,
    );

    fn add_completion_handler(
        &mut self,
        handler: impl FnOnce(Result<(), <Self::Backend as Backend>::Error>) + 'static,
    );

    fn submit(&mut self);

    fn wait_until_completed(&self) -> Result<(), <Self::Backend as Backend>::Error>;

    fn gpu_execution_time_ms(&self) -> Option<f64>;
}
