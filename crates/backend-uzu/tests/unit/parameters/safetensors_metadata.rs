use std::fs::File;

use proc_macros::uzu_test;
use test_runner::path::get_test_weights_path;

use crate::parameters::safetensors_metadata::{read_metadata, summarize_header};

#[uzu_test]
fn test_metadata_loading() {
    let path = get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_metadata(&file).expect("read metadata");
    assert!(!metadata.tensors.is_empty());
}

#[uzu_test]
fn test_header_summary() {
    let path = get_test_weights_path();
    let summary = summarize_header(&path).expect("summarize header");
    assert!(summary.tensor_count > 0);
    assert!(summary.logical_payload_bytes > 0);
}
