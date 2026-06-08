use std::{fs::File, path::PathBuf};

use backend_uzu::{
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    data_type::DataType,
    parameters::{Dtype, ParameterLoader, read_safetensors_metadata},
};
use tokenizers::Tokenizer;

use crate::{
    common::{
        for_each_non_cpu_backend,
        path::{get_test_model_path, get_traces_path},
    },
    tracer::trace_validator::{TraceValidator, export_tokenization_trace},
};

fn test_tracer_internal<B: Backend>() {
    let model_path = get_test_model_path();
    let mut tracer = TraceValidator::<B>::new(&model_path).expect("Failed to create TraceValidator");
    let input_file = File::open(get_traces_path()).expect("open input trace");
    let (_offset, input_trace) = read_safetensors_metadata(&input_file).expect("read input trace metadata");
    let (export_path, _temp_file) = match std::env::var_os("UZU_TRACE_EXPORT_PATH") {
        Some(path) => (PathBuf::from(path), None),
        None => {
            let directory = tempfile::TempDir::new().expect("create exported trace directory");
            (directory.path().join("uzu-traces.safetensors"), Some(directory))
        },
    };
    tracer.export_trace(&export_path).expect("export uzu trace");

    let file = File::open(&export_path).expect("open exported trace");
    let (_offset, metadata) = read_safetensors_metadata(&file).expect("read exported trace metadata");
    assert_eq!(metadata.metadata, input_trace.metadata);
    assert!(metadata.tensors.contains_key("activation_trace.token_ids"));
    assert!(metadata.tensors.contains_key("activation_trace.token_positions"));
    assert!(metadata.tensors.contains_key("activation_trace.output_norm"));
    assert!(metadata.tensors.contains_key("logits"));
    assert!(metadata.tensors.keys().any(|path| path.ends_with(".activation_trace.inputs")));
    assert!(metadata.tensors.keys().any(|path| path.ends_with(".updated_state.keys")));
}

#[test]
#[ignore = "requires a Lalamo traces.safetensors fixture in the test model directory"]
fn test_tracer() {
    let traces_path = get_traces_path();
    assert!(traces_path.exists(), "Traces file missing at {:?}", traces_path);

    for_each_non_cpu_backend!(|B| {
        test_tracer_internal::<B>();
    })
}

#[test]
fn test_tokenization_tracer() {
    let model_path = get_test_model_path();
    let (export_path, _temp_directory) = match std::env::var_os("UZU_TOKENIZATION_TRACE_EXPORT_PATH") {
        Some(path) => (PathBuf::from(path), None),
        None => {
            let directory = tempfile::TempDir::new().expect("create tokenization trace directory");
            (directory.path().join("uzu-tokenization-trace.safetensors"), Some(directory))
        },
    };
    let message = std::env::var("UZU_TRACE_MESSAGE").unwrap_or_else(|_| "Tell me about London".to_string());

    export_tokenization_trace(&model_path, &export_path, &message).expect("export tokenization trace");

    let file = File::open(&export_path).expect("open tokenization trace");
    let (_offset, trace) = read_safetensors_metadata(&file).expect("read tokenization trace metadata");
    let token_ids_info = &trace.tensors["activation_trace.token_ids"];
    let token_positions_info = &trace.tensors["activation_trace.token_positions"];
    assert_eq!(token_ids_info.dtype, Dtype::I32);
    assert_eq!(token_positions_info.dtype, Dtype::I32);
    assert_eq!(token_ids_info.shape[0], 1);
    assert_eq!(token_positions_info.shape, token_ids_info.shape);
    let metadata = trace.metadata.expect("tokenization trace metadata");
    assert_eq!(metadata["add_special_tokens"], "false");
    assert!(metadata["rendered_request"].contains(&message));
    let request: serde_json::Value = serde_json::from_str(&metadata["request"]).expect("parse request metadata");
    assert_eq!(request["add_generation_prompt"], true);
    let messages = request["messages"].as_array().expect("request messages");
    assert_eq!(messages.last().expect("user message"), &serde_json::json!({"role": "user", "content": message}));
    assert_eq!(request["enable_thinking"], true);
    assert!(request.get("bos_token").is_some());
    assert!(request.get("eos_token").is_some());

    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let loader = ParameterLoader::<Cpu>::new(&file, context.as_ref()).expect("load tokenization trace");
    let token_shape = token_ids_info.shape.as_slice();
    let token_ids = loader
        .tree()
        .leaf("activation_trace.token_ids")
        .unwrap()
        .validate(token_shape, DataType::I32)
        .unwrap()
        .read_array()
        .unwrap();
    let token_positions = loader
        .tree()
        .leaf("activation_trace.token_positions")
        .unwrap()
        .validate(token_shape, DataType::I32)
        .unwrap()
        .read_array()
        .unwrap();
    let expected_tokens = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .expect("load tokenizer")
        .encode(metadata["rendered_request"].as_str(), false)
        .expect("encode rendered request");
    let expected_token_ids =
        expected_tokens.get_ids().iter().map(|token| i32::try_from(*token).unwrap()).collect::<Vec<_>>();
    let expected_positions = (0..expected_token_ids.len()).map(|position| position as i32).collect::<Vec<_>>();

    assert_eq!(token_ids.as_slice::<i32>(), expected_token_ids.as_slice());
    assert_eq!(token_positions.as_slice::<i32>(), expected_positions.as_slice());
    assert_eq!(
        serde_json::from_str::<Vec<String>>(&metadata["tokens"]).unwrap(),
        expected_tokens.get_tokens().to_vec()
    );
}
