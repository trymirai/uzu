#![cfg(metal_backend)]

use std::{path::PathBuf, time::Instant};

use backend_uzu::session::{Session, config::DecodingConfig};
use proc_macros::uzu_test;
use test_tag::tag;

#[ignore = "requires THINKING_TEST_MODEL pointing at a downloaded model"]
#[tag(heavy)]
#[uzu_test]
fn load_time() {
    let path = PathBuf::from(std::env::var("THINKING_TEST_MODEL").expect("set THINKING_TEST_MODEL"));
    let start = Instant::now();
    let _session = Session::new(path, DecodingConfig::default()).expect("load model");
    println!("load_time: {:.3} s", start.elapsed().as_secs_f64());
}
