#![cfg_attr(feature = "nightly-harness", feature(test))]

pub mod env_vars;
pub mod metrics;
pub mod path;
pub mod perf;
pub mod util;

#[cfg(feature = "nightly-harness")]
pub extern crate test;

#[cfg(feature = "nightly-harness")]
mod harness;

#[cfg(feature = "nightly-harness")]
pub use harness::{UzuTest, uzu_harness};

/// Invokes `$body` once per available backend, with `$B` bound to each backend type.
#[macro_export]
macro_rules! for_each_backend {
    (|$B:ident| $body:expr) => {{
        {
            type $B = crate::backends::cpu::Cpu;
            $body
        }
        #[cfg(backend = "metal")]
        {
            type $B = crate::backends::metal::Metal;
            $body
        }
    }};
}

#[macro_export]
macro_rules! for_each_non_cpu_backend {
    (|$B:ident| $body:expr) => {{
        #[cfg(backend = "metal")]
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
