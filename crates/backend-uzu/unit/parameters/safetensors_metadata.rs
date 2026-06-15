use std::fs::File;

use proc_macros::uzu_test;
use test_runner::path::get_test_weights_path;

use crate::parameters::safetensors_metadata::read_metadata;

#[uzu_test]
fn test_metadata_loading() {
    let path = get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_metadata(&file).expect("read metadata");
    assert!(!metadata.tensors.is_empty());
}
