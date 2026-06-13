use crate::tests::env_vars;

pub fn enable_benchmark_gpu_capture_if_requested() {
    if env_vars::enabled(env_vars::UZU_CAPTURE_BENCH) {
        unsafe {
            std::env::set_var(env_vars::METAL_CAPTURE_ENABLED, "1");
        }
    }
}

#[cfg(metal_backend)]
pub fn shared_metal_context() -> std::rc::Rc<backend_uzu::backends::metal::MetalContext> {
    use backend_uzu::backends::{common::Context, metal::MetalContext};
    thread_local! {
        static CTX: std::cell::OnceCell<std::rc::Rc<MetalContext>> = const { std::cell::OnceCell::new() };
    }
    CTX.with(|cell| cell.get_or_init(|| MetalContext::new().expect("Metal context")).clone())
}

pub fn type_short_name<T>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
}
