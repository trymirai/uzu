mod common;

// New integration test for ParameterLoader
use half::bf16;
use is_close::is_close;
use uzu::{
    backends::{common::Context, metal::MetalContext},
    parameters::ParameterLoader,
};

#[test]
fn test_parameter_loader_basic() {
    let weights_path = crate::common::get_test_weights_path();
    let context = MetalContext::new().expect("Failed to create MetalContext");
    let file = std::fs::File::open(&weights_path).expect("Weights file not found; run download script");

    let loader = ParameterLoader::new(&file, context.as_ref()).expect("create loader");
    let embeddings = loader.get("embedding.weights").expect("weights embeddings");
    let view = embeddings.as_view::<bf16>();
    assert!(is_close!(view[[5usize, 3usize]], bf16::from_f32(-0.01819)));

    // tree API check
    let subtree = loader.tree().subtree("embedding").unwrap();
    let same = subtree.leaf("weights").unwrap();
    assert_eq!(view, same.as_view::<bf16>());
}
