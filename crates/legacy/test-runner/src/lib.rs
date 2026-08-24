#![cfg_attr(feature = "nightly-harness", feature(test))]

#[cfg(feature = "nightly-harness")]
pub extern crate test;

#[cfg(feature = "nightly-harness")]
mod harness;

#[cfg(feature = "nightly-harness")]
pub use harness::{UzuTest, uzu_harness};
