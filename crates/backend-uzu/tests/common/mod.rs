#![allow(dead_code, unused_imports, unused_macros)]

pub mod assert;
pub mod audio;
pub mod helpers;
pub mod path;
pub mod perf;
pub mod proptest;

pub(crate) use proptest::{dispatch_dtype, for_each_context};

pub fn type_short_name<T>() -> &'static str {
    std::any::type_name::<T>().rsplit("::").next().unwrap()
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

macro_rules! for_each_float_type {
    (|$F:ident| $body:expr) => {{
        {
            type $F = f32;
            $body
        }
        {
            type $F = half::f16;
            $body
        }
        {
            type $F = half::bf16;
            $body
        }
    }};
}
pub(crate) use for_each_float_type;
