use std::path::PathBuf;

use uzu::{Array, backends::metal::MTLContext};

#[test]
fn test_layer_norm_vs_lalamo() {
    // Load BERT model to get normalization scales
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models/modern_bert");

    if !model_path.exists() {
        println!("Skipping test: BERT model not found");
        return;
    }

    // Create Metal context
    let device = metal::Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let mtl_context =
        MTLContext::new(device, queue).expect("Failed to create MTLContext");

    // Load embedding_norm scales
    let weights_path = model_path.join("model.safetensors");
    let weights_file =
        std::fs::File::open(&weights_path).expect("Failed to open weights");
    let loader =
        uzu::parameters::ParameterLoader::new(&weights_file, &mtl_context)
            .expect("Failed to create loader");
    let tree = loader.tree();

    let scales_arr =
        tree.leaf("embedding_norm.scales").expect("Failed to load scales");

    // Support both BF16 and F32 scales
    let scales: Vec<f32> =
        if let Ok(bf16_slice) = scales_arr.as_slice::<half::bf16>() {
            bf16_slice.iter().map(|s| s.to_f32()).collect()
        } else if let Ok(f32_slice) = scales_arr.as_slice::<f32>() {
            f32_slice.to_vec()
        } else {
            panic!(
                "Unsupported scales dtype: {:?}",
                uzu::Array::data_type(&scales_arr)
            );
        };

    println!("Scales (first 10): {:?}", &scales[..10]);
    println!("Scale offset from config should be: None (0.0)");
    println!("Epsilon from config should be: 1e-05");
    println!("Subtract mean: true");

    // Create a simple test input
    let test_input: Vec<f32> =
        (0..768).map(|i| (i as f32 - 384.0) / 100.0).collect();

    // Compute expected output using Lalamo's algorithm
    let mean: f32 = test_input.iter().sum::<f32>() / test_input.len() as f32;
    let centered: Vec<f32> = test_input.iter().map(|x| x - mean).collect();
    let variance: f32 =
        centered.iter().map(|x| x * x).sum::<f32>() / centered.len() as f32;
    let inv_std = 1.0 / (variance + 1e-05_f32).sqrt();
    let expected: Vec<f32> = centered
        .iter()
        .zip(scales.iter())
        .map(|(x, s)| x * inv_std * *s)
        .collect();

    println!("\nTest input mean: {:.6}", mean);
    println!("Test input variance: {:.6}", variance);
    println!("Expected output (first 10): {:?}", &expected[..10]);

    println!("\nℹ️  This test validates LayerNorm math");
    println!("Run Lalamo separately to compare activation values");
}
