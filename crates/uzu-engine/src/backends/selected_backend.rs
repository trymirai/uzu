/// The backend `select_backend` opens: `UZU_BACKEND`, lowercased, else the first built of Metal, Vulkan, WebGPU and the
/// CPU.
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
