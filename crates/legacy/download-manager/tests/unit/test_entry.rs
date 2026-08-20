extern crate self as download_manager;

include!("../../src/lib.rs");

#[path = "../common/mod.rs"]
mod common;

mod unit_tests {
    include!("mod.rs");
}
