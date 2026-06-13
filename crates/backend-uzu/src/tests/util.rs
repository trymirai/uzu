use test_runner::env_vars;

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
