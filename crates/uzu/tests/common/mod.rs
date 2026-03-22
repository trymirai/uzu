#![allow(dead_code)]

pub mod assert;
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
