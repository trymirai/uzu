#![feature(test)]

mod harness;

pub mod env_vars;
pub mod metrics;
pub mod path;
pub mod perf;
pub mod util;

pub extern crate test;

pub use harness::{UzuTest, uzu_harness};

/// Invokes `$body` once per available backend, with `$B` bound to each backend type.
#[macro_export]
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

#[macro_export]
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
