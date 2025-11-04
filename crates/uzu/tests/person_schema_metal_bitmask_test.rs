#![cfg(target_os = "macos")]

use std::path::PathBuf;

use metal::{Device as MTLDevice, MTLResourceOptions};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use uzu::{Array, DataType, backends::metal::MetalArray};
use xgrammar::{
    DLDataType, DLDevice, DLDeviceType, DLTensor, Grammar, GrammarCompiler,
    GrammarMatcher, TokenizerInfo,
};

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
    let model_dir = repo_root
        .join("models")
        .join(crate_version)
        .join("Llama-3.2-1B-Instruct");

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!(
            "Skipping test: tokenizer.json not found at {}.\nRun scripts/download_test_model.sh first.",
            tokenizer_path.display()
        );
        return;
    }

    if std::env::var("RUN_XGRAMMAR_JSON_SCHEMA").ok().as_deref() != Some("1") {
        eprintln!(
            "Skipping JSON-schema compile part. Set RUN_XGRAMMAR_JSON_SCHEMA=1 to run it."
        );
        return;
    }

    let schema_root = schemars::schema_for!(Person);
    let schema_str =
        serde_json::to_string(&schema_root).expect("schema to string");

    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).expect("load tokenizer.json");
    let tokenizer_info =
        TokenizerInfo::from_huggingface(&tokenizer, None, None);

    let grammar = Grammar::from_json_schema(
        &schema_str,
        true,             // any_whitespace
        Some(2),          // indent
        Some((",", ":")), // separators
        true,             // strict_mode
        false,            // print_converted_ebnf
    );
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1);
    let compiled = compiler.compile_grammar(&grammar);

    let mut matcher = GrammarMatcher::new(&compiled, None, true, -1);

    let device = match MTLDevice::system_default() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        },
    };
    let batch: usize = 1;
    let vocab = tokenizer_info.vocab_size() as usize;
    let shape = [batch, vocab];

    let bytes = batch * vocab; // assume u8 bitmask
    let buffer =
        device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let mut metal_bitmask =
        unsafe { MetalArray::new(buffer, &shape, DataType::U8) };

    let mut cpu_mask: Vec<u8> = vec![0u8; vocab];

    let mut shape_i64 = [vocab as i64];
    let mut bitmask_tensor = DLTensor {
        data: cpu_mask.as_mut_ptr() as *mut core::ffi::c_void,
        device: DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: DLDataType {
            code: 1,
            bits: 8,
            lanes: 1,
        },
        shape: shape_i64.as_mut_ptr(),
        strides: core::ptr::null_mut(),
        byte_offset: 0,
    };

    let _ok = matcher.fill_next_token_bitmask(&mut bitmask_tensor, 0, false);

    metal_bitmask
        .as_slice_mut::<u8>()
        .expect("slice U8")
        .copy_from_slice(&cpu_mask);

    assert_eq!(metal_bitmask.shape(), &shape);
    assert_eq!(metal_bitmask.as_slice::<u8>().unwrap().len(), bytes);
}
