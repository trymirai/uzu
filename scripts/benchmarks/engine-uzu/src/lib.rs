#![cfg(any(target_vendor = "apple", windows))]

mod args;
mod bench;
mod common;
mod engine;
#[cfg_attr(target_vendor = "apple", path = "memory/mach.rs")]
#[cfg_attr(windows, path = "memory/windows.rs")]
mod memory;
mod run;

pub use args::Args;
pub use run::{benchmarks_uzu_main, run};
