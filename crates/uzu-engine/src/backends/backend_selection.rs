use crate::backends::common::Backend;

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
        "metal" => selection.select::<crate::backends::metal::Metal>().await,
        #[cfg(backend = "vulkan")]
        "vulkan" => selection.select::<crate::backends::vulkan::Vulkan>().await,
        #[cfg(backend = "webgpu")]
        "webgpu" => selection.select::<crate::backends::webgpu::WebGPU>().await,
        #[cfg(backend = "cpu")]
        "cpu" => selection.select::<crate::backends::cpu::Cpu>().await,
        _ => Err(unknown),
    }
}

pub fn selected_backend() -> String {
    let default = if cfg!(backend = "metal") {
        "metal"
    } else if cfg!(backend = "vulkan") {
        "vulkan"
    } else if cfg!(backend = "webgpu") {
        "webgpu"
    } else if cfg!(backend = "cpu") {
        "cpu"
    } else {
        unreachable!()
    };

    // TODO: remove magic env var
    std::env::var("UZU_BACKEND").map(|s| s.to_lowercase()).unwrap_or_else(|_| default.to_string())
}
