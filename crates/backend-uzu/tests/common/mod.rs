#![allow(dead_code, unused_imports, unused_macros)]

pub mod assert;
pub mod audio;
pub mod helpers;
pub mod matmul;
pub mod metrics;
pub mod path;
pub mod perf;
pub mod proptest;

pub(crate) use proptest::for_each_context;

pub fn type_short_name<T>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
}

pub fn env_var_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("yes") || v.eq_ignore_ascii_case("true"))
}

pub fn enable_benchmark_gpu_capture_if_requested() {
    if env_var_enabled("UZU_CAPTURE_BENCH") {
        unsafe {
            std::env::set_var("METAL_CAPTURE_ENABLED", "1");
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

/// Invokes `$body` once per available backend, with `$B` bound to each backend type.
macro_rules! for_each_backend {
    (|$B:ident| $body:expr) => {{
        {
            type $B = backend_uzu::backends::cpu::Cpu;
            $body
        }
        #[cfg(metal_backend)]
        {
            type $B = backend_uzu::backends::metal::Metal;
            $body
        }
    }};
}
pub(crate) use for_each_backend;

macro_rules! for_each_non_cpu_backend {
    (|$B:ident| $body:expr) => {{
        #[cfg(metal_backend)]
        {
            type $B = backend_uzu::backends::metal::Metal;
            $body
        }
        {
            if false {
                type $B = backend_uzu::backends::cpu::Cpu;
                $body
            }
        }
    }};
}
pub(crate) use for_each_non_cpu_backend;
