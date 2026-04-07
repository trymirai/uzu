#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use dsl::{__internal_uzu_bench as uzu_bench, __internal_uzu_ignored as uzu_test};

include!("mod.rs");
