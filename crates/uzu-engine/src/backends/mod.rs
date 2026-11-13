pub(crate) mod common;

#[cfg(backend = "metal")]
pub(crate) mod metal;

#[cfg(backend = "vulkan")]
pub mod vulkan;

#[cfg(backend = "webgpu")]
pub(crate) mod webgpu;

#[cfg(backend = "cpu")]
pub(crate) mod cpu;

mod selected_backend;

pub use common::Backend;
pub use selected_backend::selected_backend;

pub trait BackendSelection {
    type Output;
    type Error;

    fn select<B: Backend>(self) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

pub async fn select_backend<S: BackendSelection>(
    selection: S,
    unknown: S::Error,
) -> Result<S::Output, S::Error> {
    match selected_backend().as_str() {
        #[cfg(backend = "metal")]
        "metal" => selection.select::<metal::Metal>().await,
        #[cfg(backend = "vulkan")]
        "vulkan" => selection.select::<vulkan::Vulkan>().await,
        #[cfg(backend = "webgpu")]
        "webgpu" => selection.select::<webgpu::WebGPU>().await,
        #[cfg(backend = "cpu")]
        "cpu" => selection.select::<cpu::Cpu>().await,
        _ => Err(unknown),
    }
}
