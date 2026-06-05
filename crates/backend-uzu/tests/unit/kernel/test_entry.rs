use proc_macros::{__internal_uzu_ignored as uzu_bench, __internal_uzu_test as uzu_test};

#[macro_use]
#[path = "../../common/mod.rs"]
mod common;

#[path = "mod.rs"]
mod kernel;
