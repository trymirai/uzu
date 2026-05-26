use std::{fs::File, path::PathBuf};

use backend_uzu::{backends::common::Backend, read_safetensors_metadata};

use crate::{
    common::{
        for_each_non_cpu_backend,
        path::{get_test_model_path, get_traces_path},
    },
    tracer::trace_validator::TraceValidator,
};

fn test_tracer_internal<B: Backend>() {
    let model_path = get_test_model_path();
    let mut tracer = TraceValidator::<B>::new(&model_path).expect("Failed to create TraceValidator");
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
    assert!(metadata.tensors.contains_key("activation_trace.token_ids"));
    assert!(metadata.tensors.contains_key("activation_trace.token_positions"));
    assert!(metadata.tensors.contains_key("activation_trace.output_norm"));
    assert!(metadata.tensors.contains_key("logits"));
    assert!(metadata.tensors.keys().any(|path| path.ends_with(".activation_trace.inputs")));
}

#[test]
fn test_tracer() {
    let traces_path = get_traces_path();
    assert!(traces_path.exists(), "Traces file missing at {:?}", traces_path);

    for_each_non_cpu_backend!(|B| {
        test_tracer_internal::<B>();
    })
}
