#![cfg(target_os = "macos")]

use std::path::PathBuf;

use metal::{MTLDevice, MTLDeviceExt, MTLResourceOptions};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use uzu::{DataType, array::Array, backends::metal::Metal};
use xgrammar::{DLDevice, DLDeviceType, DLTensor, Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: u32,
}

#[test]
fn person_schema_metal_bitmask() {
    // Locate the tokenizer.json under {repo}/models/{version}/Llama-3.2-1B-Instruct
    let crate_version = env!("CARGO_PKG_VERSION");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let model_dir = repo_root.join("models").join(crate_version).join("Llama-3.2-1B-Instruct");

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!(
            "Skipping test: tokenizer.json not found at {}.\nRun scripts/download_test_model.sh first.",
            tokenizer_path.display()
        );
        return;
    }

    let schema_root = schemars::schema_for!(Person);
    let schema_str = serde_json::to_string(&schema_root).expect("schema to string");

    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer.json");
    let tokenizer_info = TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();

    let grammar = Grammar::from_json_schema(&schema_str, true, Some(2), Some((",", ":")), true, None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();

    let device = match <dyn MTLDevice>::system_default() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        },
    };
    let device_id = device.registry_id() as i32;
    let batch: usize = 1;
    let vocab = tokenizer_info.vocab_size() as usize;
    // xgrammar uses a dynamic bitset over the vocab, packed into i32 words
    let buffer_size = (vocab + 31) / 32; // number of i32s required
    let shape = [batch, buffer_size];

    // xgrammar expects an int32 mask tensor of shape [buffer_size]
    let elems = batch * buffer_size;
    let bytes = elems * core::mem::size_of::<i32>();
    let buffer = device.new_buffer(bytes, MTLResourceOptions::STORAGE_MODE_SHARED).expect("Failed to create buffer");
    let metal_bitmask = unsafe { Array::<Metal>::from_parts(buffer, 0, &shape, DataType::I32) };

    let mut shape_i64 = [buffer_size as i64];
    let mut bitmask_tensor = DLTensor {
        data: metal_bitmask.cpu_ptr().as_ptr(),
        device: DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id,
        },
        ndim: 1,
        dtype: DataType::I32.into(),
        shape: shape_i64.as_mut_ptr(),
        strides: core::ptr::null_mut(),
        byte_offset: 0,
    };

    let _ok = matcher.fill_next_token_bitmask(&mut bitmask_tensor, 0, false);

    assert_eq!(metal_bitmask.shape(), &shape);
    assert_eq!(metal_bitmask.as_slice::<i32>().len(), elems);
}
