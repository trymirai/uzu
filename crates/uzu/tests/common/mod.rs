#![allow(dead_code)]

pub mod assert;
pub mod audio;
pub mod helpers;
pub mod path;
pub mod perf;
pub mod proptest;

/// Invokes `$body` once per available backend, with `$B` bound to each backend type.
#[macro_export]
macro_rules! for_each_backend {
    (|$B:ident| $body:expr) => {{
        {
            type $B = uzu::backends::cpu::Cpu;
            $body
        }
        #[cfg(metal_backend)]
        {
            type $B = uzu::backends::metal::Metal;
            $body
        }
    }};
}

#[macro_export]
macro_rules! for_each_non_cpu_backend {
    (|$B:ident| $body:expr) => {{
        #[cfg(metal_backend)]
        {
            type $B = uzu::backends::metal::Metal;
            $body
        }
        {
            if false {
                type $B = uzu::backends::cpu::Cpu;
                $body
            }
        }
    }};
}

#[macro_export]
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
