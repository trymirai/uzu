#![cfg(metal_backend)]

// New integration test for ParameterLoader
use backend_uzu::{
    backends::{
        common::{Backend, Context},
        metal::Metal,
    },
    data_type::DataType,
    parameters::{ParameterLoader, read_safetensors_metadata},
};
use half::bf16;
use test_tag::tag;

use crate::common::path::get_test_weights_path;

const EMBEDDING_PATH: &str = "decoder.embedding.embedding.weights";
const EMBEDDING_TREE_PATH: &str = "decoder.embedding.embedding";

#[tag(heavy)]
#[test]
fn test_parameter_loader_basic() {
    let weights_path = get_test_weights_path();
    let context = <Metal as Backend>::Context::new().expect("Failed to create MetalContext");
    let file = std::fs::File::open(&weights_path).expect("Weights file not found; run download script");
    let (_header_len, metadata) = read_safetensors_metadata(&file).expect("read weights metadata");
    let embedding_shape = metadata.tensors.get(EMBEDDING_PATH).expect("weights embeddings metadata").shape.clone();
    assert_eq!(embedding_shape.len(), 2);

    let loader = ParameterLoader::<Metal>::new(&file, context.as_ref()).expect("create loader");
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
