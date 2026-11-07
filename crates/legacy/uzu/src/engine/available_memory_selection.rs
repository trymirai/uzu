use uzu_engine::{
    backends::{Backend, BackendSelection},
    engine::Engine,
};

/// The memory free on the selected backend's device, where the backend reports it.
pub struct AvailableMemorySelection;

impl BackendSelection for AvailableMemorySelection {
    type Output = Option<usize>;
    type Error = String;

    async fn select<B: Backend>(self) -> Result<Self::Output, Self::Error> {
        Ok(Engine::<B>::new().await.map_err(|error| error.to_string())?.available_memory())
    }
}
