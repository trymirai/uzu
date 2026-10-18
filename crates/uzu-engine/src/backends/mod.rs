pub(crate) mod common;

#[cfg(backend = "metal")]
pub(crate) mod metal;

#[cfg(backend = "webgpu")]
pub(crate) mod webgpu;

#[cfg(backend = "cpu")]
pub(crate) mod cpu;

pub use common::Backend;

pub trait BackendSelection {
    type Output;
    type Error;

    fn select<B: Backend>(self) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

pub async fn select_backend<S: BackendSelection>(
    selection: S,
    unknown: S::Error,
) -> Result<S::Output, S::Error> {
    let default = if cfg!(backend = "metal") {
        "metal"
    } else if cfg!(backend = "webgpu") {
        "webgpu"
    } else if cfg!(backend = "cpu") {
        "cpu"
    } else {
        unreachable!()
    };

    // TODO: remove magic env var
    match std::env::var("UZU_BACKEND").map(|s| s.to_lowercase()).as_deref().unwrap_or(default) {
        #[cfg(backend = "metal")]
        "metal" => selection.select::<metal::Metal>().await,
        #[cfg(backend = "webgpu")]
        "webgpu" => selection.select::<webgpu::WebGPU>().await,
        #[cfg(backend = "cpu")]
        "cpu" => selection.select::<cpu::Cpu>().await,
        _ => Err(unknown),
    }
}
