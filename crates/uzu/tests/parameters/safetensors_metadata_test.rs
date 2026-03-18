#[macro_use]
#[path = "../common/mod.rs"]
mod common;

use std::fs::File;

use uzu::parameters::read_safetensors_metadata;

#[test]
fn test_metadata_loading() {
    let path = common::get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_safetensors_metadata(&file).expect("read metadata");
    assert!(metadata.tensors.len() > 0);
}
