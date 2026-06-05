use std::fs::File;

use proc_macros::uzu_test;

use crate::{common::path::get_test_weights_path, parameters::safetensors_metadata::read_metadata};

#[uzu_test]
fn test_metadata_loading() {
    let path = get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_metadata(&file).expect("read metadata");
    assert!(!metadata.tensors.is_empty());
}
