#![allow(dead_code, unused_imports, unused_macros)]

pub mod audio;
pub mod matmul;
pub mod metrics;
pub mod path;
pub mod proptest;

pub(crate) use proptest::for_each_context;

pub fn type_short_name<T>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
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
