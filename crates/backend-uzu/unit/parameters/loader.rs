use std::fs::File;

use half::bf16;
use proc_macros::uzu_test;
use test_runner::{for_each_backend, path::get_test_weights_path};
use test_tag::tag;

use crate::{
    backends::common::{Backend, Context},
    data_type::DataType,
    parameters::{ParameterLoader, safetensors_metadata::read_metadata},
};

const EMBEDDING_PATH: &str = "decoder.embedding.embedding.weights";
const EMBEDDING_TREE_PATH: &str = "decoder.embedding.embedding";

fn check_tree_api<B: Backend>(context: &B::Context) {
    let weights_path = get_test_weights_path();
    let file = File::open(&weights_path).expect("Weights file not found; run download script");
    let (_header_len, metadata) = read_metadata(&file).expect("read weights metadata");
    let embedding_shape = metadata.tensors.get(EMBEDDING_PATH).expect("weights embeddings metadata").shape.clone();
    assert_eq!(embedding_shape.len(), 2);

    let loader = ParameterLoader::<B>::new(&file, context).expect("create loader");
    let tree = loader.tree();
    let embeddings_leaf =
        tree.leaf(EMBEDDING_PATH).expect("weights embeddings").validate(&embedding_shape, DataType::BF16).unwrap();
    let embeddings = embeddings_leaf.read_array().unwrap();
    let view = embeddings.as_view::<bf16>();

    // tree API check
    let subtree = tree.subtree(EMBEDDING_TREE_PATH).unwrap();
    let weights_leaf = subtree.leaf("weights").unwrap().validate(&embedding_shape, DataType::BF16).unwrap();
    let same = weights_leaf.read_array().unwrap();
    assert_eq!(view, same.as_view::<bf16>());
}

#[tag(heavy)]
#[uzu_test]
fn test_parameter_loader_basic() {
    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().expect("Failed to create context");
        check_tree_api::<B>(&context);
    })
}
