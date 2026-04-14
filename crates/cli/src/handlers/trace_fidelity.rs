use std::{ffi::OsString, fs, path::PathBuf};

use serde::Serialize;
use uzu::{
    TraceDebugSnapshot,
    session::{ChatSession, config::DecodingConfig, types::Input},
};

use super::kv_fidelity::{build_run_config, install_kv_env, load_task};

#[derive(Debug, Clone, Serialize)]
struct TraceMetric {
    cosine: f64,
    mean_abs: f64,
    max_abs: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TraceLayerReport {
    layer_index: usize,
    row_index: usize,
    baseline_active_row_count: usize,
    candidate_active_row_count: usize,
    candidate_sparse_value_single_decode_has_kv_cache: bool,
    candidate_sparse_value_single_decode_has_sparse_value: bool,
    candidate_sparse_value_single_decode_suffix_length: usize,
    candidate_sparse_value_single_decode_projection_step: usize,
    candidate_sparse_value_single_decode_is_trie: bool,
    candidate_sparse_value_single_decode_is_kv_cache_ring: bool,
    candidate_attempted_sparse_value_single_decode: bool,
    candidate_used_sparse_value_single_decode: bool,
    pre_attention_norm: TraceMetric,
    attention: TraceMetric,
    candidate_expected_attention: Option<TraceMetric>,
    candidate_attention_vs_expected: Option<TraceMetric>,
    outputs: TraceMetric,
}

#[derive(Debug, Clone, Serialize)]
struct TraceReport {
    method: String,
    generated_tokens: usize,
    row_index: usize,
    baseline_text: String,
    candidate_text: String,
    layers: Vec<TraceLayerReport>,
}

#[derive(Default)]
struct MetricAccumulator {
    dot: f64,
    left_sq: f64,
    right_sq: f64,
    abs_sum: f64,
    max_abs: f64,
    count: usize,
}

impl MetricAccumulator {
    fn update(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) {
        assert_eq!(left.len(), right.len(), "trace vectors must have the same length");
        for (&left, &right) in left.iter().zip(right.iter()) {
            let left = left as f64;
            let right = right as f64;
            let diff = (left - right).abs();
            self.dot += left * right;
            self.left_sq += left * left;
            self.right_sq += right * right;
            self.abs_sum += diff;
            self.max_abs = self.max_abs.max(diff);
            self.count += 1;
        }
    }

    fn finish(&self) -> TraceMetric {
        let cosine = if self.left_sq == 0.0 || self.right_sq == 0.0 {
            0.0
        } else {
            self.dot / (self.left_sq.sqrt() * self.right_sq.sqrt())
        };
        TraceMetric {
            cosine,
            mean_abs: self.abs_sum / self.count as f64,
            max_abs: self.max_abs,
        }
    }
}

struct AsyncEnvGuard {
    old_disable_async_generation: Option<OsString>,
}

impl Drop for AsyncEnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.old_disable_async_generation {
                Some(value) => std::env::set_var("UZU_DISABLE_ASYNC_GENERATION", value),
                None => std::env::remove_var("UZU_DISABLE_ASYNC_GENERATION"),
            }
        }
    }
}

fn install_trace_env() -> AsyncEnvGuard {
    let guard = AsyncEnvGuard {
        old_disable_async_generation: std::env::var_os("UZU_DISABLE_ASYNC_GENERATION"),
    };
    unsafe {
        std::env::set_var("UZU_DISABLE_ASYNC_GENERATION", "1");
    }
    guard
}

fn run_snapshot(
    model_path: &str,
    task_path: &str,
    method: Option<&str>,
    turboquant_bits: Option<usize>,
    turboquant_target: Option<&str>,
    generated_tokens: usize,
) -> Result<(TraceDebugSnapshot, String), Box<dyn std::error::Error>> {
    let task = load_task(task_path)?;
    let _trace_env = install_trace_env();
    let _env = install_kv_env(method, turboquant_bits, turboquant_target);
    let decoding_config = DecodingConfig::default().with_allow_pre_encode(false);
    let mut session = ChatSession::new(PathBuf::from(model_path), decoding_config)?;
    let input = Input::Messages(task.messages.clone());
    let run_config = build_run_config(&task, true);
    let (output, snapshot) = session.run_capture_generated_trace_debug(input, run_config, generated_tokens)?;
    Ok((snapshot, output.text.original))
}

fn select_row(
    flat: &[f32],
    active_row_count: usize,
    model_dim: usize,
    row_index: usize,
) -> &[f32] {
    assert!(row_index < active_row_count, "trace row index out of bounds");
    let start = row_index * model_dim;
    &flat[start..start + model_dim]
}

fn compare_snapshots(
    method: String,
    generated_tokens: usize,
    row_index: usize,
    baseline_text: String,
    candidate_text: String,
    baseline: &TraceDebugSnapshot,
    candidate: &TraceDebugSnapshot,
) -> TraceReport {
    assert_eq!(baseline.layers.len(), candidate.layers.len(), "trace layer count must match");
    let layers = baseline
        .layers
        .iter()
        .zip(candidate.layers.iter())
        .map(|(baseline_layer, candidate_layer)| {
            assert_eq!(baseline_layer.layer_index, candidate_layer.layer_index, "trace layer index mismatch");
            assert_eq!(baseline_layer.model_dim, candidate_layer.model_dim, "trace model dim mismatch");
            let mut pre_attention_norm = MetricAccumulator::default();
            let mut attention = MetricAccumulator::default();
            let mut outputs = MetricAccumulator::default();
            pre_attention_norm.update(
                select_row(
                    &baseline_layer.pre_attention_norm,
                    baseline_layer.active_row_count,
                    baseline_layer.model_dim,
                    row_index,
                ),
                select_row(
                    &candidate_layer.pre_attention_norm,
                    candidate_layer.active_row_count,
                    candidate_layer.model_dim,
                    row_index,
                ),
            );
            attention.update(
                select_row(
                    &baseline_layer.attention,
                    baseline_layer.active_row_count,
                    baseline_layer.model_dim,
                    row_index,
                ),
                select_row(
                    &candidate_layer.attention,
                    candidate_layer.active_row_count,
                    candidate_layer.model_dim,
                    row_index,
                ),
            );
            let (candidate_expected_attention, candidate_attention_vs_expected) = candidate_layer
                .sparse_expected_attention
                .as_ref()
                .map(|candidate_expected_attention| {
                    let baseline_attention = select_row(
                        &baseline_layer.attention,
                        baseline_layer.active_row_count,
                        baseline_layer.model_dim,
                        row_index,
                    );
                    let candidate_attention = select_row(
                        &candidate_layer.attention,
                        candidate_layer.active_row_count,
                        candidate_layer.model_dim,
                        row_index,
                    );
                    let candidate_expected_attention = select_row(
                        candidate_expected_attention,
                        candidate_layer.active_row_count,
                        candidate_layer.model_dim,
                        row_index,
                    );
                    let mut expected_vs_baseline = MetricAccumulator::default();
                    expected_vs_baseline.update(baseline_attention, candidate_expected_attention);
                    let mut actual_vs_expected = MetricAccumulator::default();
                    actual_vs_expected.update(candidate_attention, candidate_expected_attention);
                    (Some(expected_vs_baseline.finish()), Some(actual_vs_expected.finish()))
                })
                .unwrap_or((None, None));
            outputs.update(
                select_row(
                    &baseline_layer.outputs,
                    baseline_layer.active_row_count,
                    baseline_layer.model_dim,
                    row_index,
                ),
                select_row(
                    &candidate_layer.outputs,
                    candidate_layer.active_row_count,
                    candidate_layer.model_dim,
                    row_index,
                ),
            );
            TraceLayerReport {
                layer_index: baseline_layer.layer_index,
                row_index,
                baseline_active_row_count: baseline_layer.active_row_count,
                candidate_active_row_count: candidate_layer.active_row_count,
                candidate_sparse_value_single_decode_has_kv_cache: candidate_layer
                    .sparse_value_single_decode_has_kv_cache,
                candidate_sparse_value_single_decode_has_sparse_value: candidate_layer
                    .sparse_value_single_decode_has_sparse_value,
                candidate_sparse_value_single_decode_suffix_length: candidate_layer
                    .sparse_value_single_decode_suffix_length,
                candidate_sparse_value_single_decode_projection_step: candidate_layer
                    .sparse_value_single_decode_projection_step,
                candidate_sparse_value_single_decode_is_trie: candidate_layer.sparse_value_single_decode_is_trie,
                candidate_sparse_value_single_decode_is_kv_cache_ring: candidate_layer
                    .sparse_value_single_decode_is_kv_cache_ring,
                candidate_attempted_sparse_value_single_decode: candidate_layer.attempted_sparse_value_single_decode,
                candidate_used_sparse_value_single_decode: candidate_layer.used_sparse_value_single_decode,
                pre_attention_norm: pre_attention_norm.finish(),
                attention: attention.finish(),
                candidate_expected_attention,
                candidate_attention_vs_expected,
                outputs: outputs.finish(),
            }
        })
        .collect();
    TraceReport {
        method,
        generated_tokens,
        row_index,
        baseline_text,
        candidate_text,
        layers,
    }
}

pub fn handle_trace_fidelity(
    model_path: String,
    task_path: String,
    output_path: String,
    method: String,
    turboquant_bits: Option<usize>,
    turboquant_target: Option<String>,
    generated_tokens: usize,
    row_index: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(cfg!(feature = "tracing"), "trace-fidelity requires the cli tracing feature");
    let (baseline_snapshot, baseline_text) = run_snapshot(&model_path, &task_path, None, None, None, generated_tokens)?;
    let (candidate_snapshot, candidate_text) = run_snapshot(
        &model_path,
        &task_path,
        Some(&method),
        turboquant_bits,
        turboquant_target.as_deref(),
        generated_tokens,
    )?;
    let report = compare_snapshots(
        method,
        generated_tokens,
        row_index,
        baseline_text,
        candidate_text,
        &baseline_snapshot,
        &candidate_snapshot,
    );
    fs::write(output_path, serde_json::to_string_pretty(&report)?)?;
    println!("Generated tokens: {}", report.generated_tokens);
    println!("Compared row: {}", report.row_index);
    println!("Baseline text: {}", report.baseline_text);
    println!("Candidate text: {}", report.candidate_text);
    for layer in report.layers.iter() {
        println!(
            "layer {} pre_norm={:.6} attention={:.6} outputs={:.6} sparse(has_kv={}, has_state={}, suffix={}, proj={}, trie={}, ring={}, attempted={}, used={})",
            layer.layer_index,
            layer.pre_attention_norm.cosine,
            layer.attention.cosine,
            layer.outputs.cosine,
            layer.candidate_sparse_value_single_decode_has_kv_cache,
            layer.candidate_sparse_value_single_decode_has_sparse_value,
            layer.candidate_sparse_value_single_decode_suffix_length,
            layer.candidate_sparse_value_single_decode_projection_step,
            layer.candidate_sparse_value_single_decode_is_trie,
            layer.candidate_sparse_value_single_decode_is_kv_cache_ring,
            layer.candidate_attempted_sparse_value_single_decode,
            layer.candidate_used_sparse_value_single_decode,
        );
        if let Some(metric) = &layer.candidate_expected_attention {
            println!("layer {} expected_attention={:.6}", layer.layer_index, metric.cosine,);
        }
        if let Some(metric) = &layer.candidate_attention_vs_expected {
            println!("layer {} actual_vs_expected={:.6}", layer.layer_index, metric.cosine,);
        }
    }
    Ok(())
}
