mod common;

use is_close::is_close;
use uzu::{backends::cpu::CPUContext, parameters::ParameterLoader};
use uzu::Array;

// New integration test for ParameterLoader
use half::f16;

#[test]
fn test_parameter_loader_basic() {
    let weights_path = crate::common::get_test_weights_path();
    let context = CPUContext::new();
    let file = std::fs::File::open(&weights_path)
        .expect("Weights file not found; run download script");

    let loader = ParameterLoader::new(&file, &context).expect("create loader");
    let embeddings = loader
        .get("embedding.token_embeddings")
        .expect("load embeddings");
    let view = embeddings.as_view::<f16>().unwrap();
    assert!(is_close!(view[[5usize, 3usize]], f16::from_f32(-0.01819)));

    // tree API check
    let subtree = loader.tree().subtree("embedding").unwrap();
    let same = subtree.leaf("token_embeddings").unwrap();
    assert_eq!(view, same.as_view::<f16>().unwrap());
} 