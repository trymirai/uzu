#![cfg(test)]
#![allow(dead_code, unused_imports, unused_macros)]

pub extern crate test;

pub mod assert;
pub mod audio;
pub mod env_vars;
pub mod helpers;
pub mod matmul;
pub mod metrics;
pub mod path;
pub mod perf;
pub mod proptest;
pub mod util;

mod harness;

pub use harness::{UzuTest, uzu_harness};

/// Invokes `$body` once per available backend, with `$B` bound to each backend type.
macro_rules! for_each_backend {
    (|$B:ident| $body:expr) => {{
        {
            type $B = crate::backends::cpu::Cpu;
            $body
        }
        #[cfg(metal_backend)]
        {
            type $B = crate::backends::metal::Metal;
            $body
        }
    }};
}
pub(crate) use for_each_backend;

macro_rules! for_each_non_cpu_backend {
    (|$B:ident| $body:expr) => {{
        #[cfg(metal_backend)]
        {
            type $B = crate::backends::metal::Metal;
            $body
        }
        {
            if false {
                type $B = crate::backends::cpu::Cpu;
                $body
            }
        }
    }};
}
pub(crate) use for_each_non_cpu_backend;
