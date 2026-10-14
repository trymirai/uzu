pub mod common;

#[cfg(backend = "cpu")]
pub mod cpu;
#[cfg(backend = "metal")]
pub mod metal;
#[cfg(backend = "webgpu")]
pub mod webgpu;

use crate::backends::common::Backend;

pub trait BackendSelection {
    type Output;
    type Error;

    fn select<B: Backend>(self) -> Result<Self::Output, Self::Error>;
}

pub fn select_backend<S: BackendSelection>(
    selection: S,
    unknown: S::Error,
) -> Result<S::Output, S::Error> {
    let default = if cfg!(backend = "webgpu") && cfg!(target_arch = "wasm32") {
        "webgpu"
    } else if cfg!(backend = "metal") {
        "metal"
    } else if cfg!(backend = "cpu") {
        "cpu"
    } else if cfg!(backend = "webgpu") {
        "webgpu"
    } else {
        unreachable!()
    };

    // TODO: remove magic env var
    match std::env::var("UZU_BACKEND").map(|s| s.to_lowercase()).as_deref().unwrap_or(default) {
        #[cfg(backend = "cpu")]
        "cpu" => selection.select::<cpu::Cpu>(),
        #[cfg(backend = "metal")]
        "metal" => selection.select::<metal::Metal>(),
        #[cfg(backend = "webgpu")]
        "webgpu" => selection.select::<webgpu::WebGpu>(),
        _ => Err(unknown),
    }
}
