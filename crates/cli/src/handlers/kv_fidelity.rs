use std::{collections::HashMap, ffi::OsString, fs, path::PathBuf};

use benchmarks::types::Task;
use serde::Serialize;
use uzu::{
    KvDebugSnapshot,
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig},
        parameter::{SamplingMethod, SamplingPolicy},
        types::Input,
    },
};

#[derive(Debug, Clone, Serialize)]
struct FidelityMetric {
    cosine: f64,
    mean_abs: f64,
    max_abs: f64,
}

#[derive(Debug, Clone, Serialize)]
struct FidelityLayerResult {
    layer_index: usize,
    baseline_token_count: usize,
    candidate_token_count: usize,
    shared_token_count: usize,
    shared_token_ratio: f64,
    baseline_storage_bytes: usize,
    candidate_storage_bytes: usize,
    compression_ratio: f64,
    key: FidelityMetric,
    value: FidelityMetric,
    recent_value: Option<FidelityMetric>,
    candidate_recent_consistency: Option<FidelityMetric>,
    pending_value: Option<FidelityMetric>,
    candidate_pending_consistency: Option<FidelityMetric>,
}

#[derive(Debug, Clone, Serialize)]
struct FidelityReport {
    method: String,
    stage: String,
    turboquant_bits: Option<usize>,
    prompt_tokens: u64,
    baseline_kv_storage_bytes: u64,
    candidate_kv_storage_bytes: u64,
    compression_ratio: f64,
    baseline_token_count: usize,
    candidate_token_count: usize,
    shared_token_count: usize,
    shared_token_ratio: f64,
    baseline_text: String,
    candidate_text: String,
    key: FidelityMetric,
    value: FidelityMetric,
    recent_value: Option<FidelityMetric>,
    candidate_recent_consistency: Option<FidelityMetric>,
    pending_value: Option<FidelityMetric>,
    candidate_pending_consistency: Option<FidelityMetric>,
    layers: Vec<FidelityLayerResult>,
}

#[derive(Default)]
struct MetricAccumulator {
    dot: f64,
    reference_sq: f64,
    candidate_sq: f64,
    abs_sum: f64,
    max_abs: f64,
    count: usize,
}

impl MetricAccumulator {
    fn update(
        &mut self,
        reference: &[f32],
        candidate: &[f32],
    ) {
        assert_eq!(reference.len(), candidate.len(), "fidelity vectors must have the same length");
        for (&left, &right) in reference.iter().zip(candidate.iter()) {
            let left = left as f64;
            let right = right as f64;
            let diff = (left - right).abs();
            self.dot += left * right;
            self.reference_sq += left * left;
            self.candidate_sq += right * right;
            self.abs_sum += diff;
            self.max_abs = self.max_abs.max(diff);
            self.count += 1;
        }
    }

    fn finish(&self) -> FidelityMetric {
        let cosine = if self.reference_sq == 0.0 || self.candidate_sq == 0.0 {
            0.0
        } else {
            self.dot / (self.reference_sq.sqrt() * self.candidate_sq.sqrt())
        };
        FidelityMetric {
            cosine,
            mean_abs: if self.count == 0 {
                0.0
            } else {
                self.abs_sum / self.count as f64
            },
            max_abs: self.max_abs,
        }
    }
}

pub(super) struct KvEnvGuard {
    old_compression: Option<OsString>,
    old_bits: Option<OsString>,
    old_shearkv_bits: Option<OsString>,
    old_target: Option<OsString>,
}

impl Drop for KvEnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.old_compression {
                Some(value) => std::env::set_var("UZU_KV_COMPRESSION", value),
                None => std::env::remove_var("UZU_KV_COMPRESSION"),
            }
            match &self.old_bits {
                Some(value) => std::env::set_var("UZU_KV_TURBOQUANT_BITS", value),
                None => std::env::remove_var("UZU_KV_TURBOQUANT_BITS"),
            }
            match &self.old_shearkv_bits {
                Some(value) => std::env::set_var("UZU_KV_SHEARKV_VALUE_BITS", value),
                None => std::env::remove_var("UZU_KV_SHEARKV_VALUE_BITS"),
            }
            match &self.old_target {
                Some(value) => std::env::set_var("UZU_KV_TURBOQUANT_TARGET", value),
                None => std::env::remove_var("UZU_KV_TURBOQUANT_TARGET"),
            }
        }
    }
}

pub(super) fn install_kv_env(
    method: Option<&str>,
    turboquant_bits: Option<usize>,
    turboquant_target: Option<&str>,
) -> KvEnvGuard {
    let guard = KvEnvGuard {
        old_compression: std::env::var_os("UZU_KV_COMPRESSION"),
        old_bits: std::env::var_os("UZU_KV_TURBOQUANT_BITS"),
        old_shearkv_bits: std::env::var_os("UZU_KV_SHEARKV_VALUE_BITS"),
        old_target: std::env::var_os("UZU_KV_TURBOQUANT_TARGET"),
    };
    unsafe {
        match method {
            Some(method) => std::env::set_var("UZU_KV_COMPRESSION", method),
            None => std::env::remove_var("UZU_KV_COMPRESSION"),
        }
        match turboquant_bits {
            Some(bits) => std::env::set_var("UZU_KV_TURBOQUANT_BITS", bits.to_string()),
            None => std::env::remove_var("UZU_KV_TURBOQUANT_BITS"),
        }
        match (method, turboquant_bits) {
            (Some(method), Some(bits))
                if method.eq_ignore_ascii_case("shearkv")
                    || method.eq_ignore_ascii_case("shear-kv")
                    || method.eq_ignore_ascii_case("shear_kv") =>
            {
                std::env::set_var("UZU_KV_SHEARKV_VALUE_BITS", bits.to_string())
            },
            _ => std::env::remove_var("UZU_KV_SHEARKV_VALUE_BITS"),
        }
        match turboquant_target {
            Some(target) => std::env::set_var("UZU_KV_TURBOQUANT_TARGET", target),
            None => std::env::remove_var("UZU_KV_TURBOQUANT_TARGET"),
        }
    }
    guard
}

pub(super) fn load_task(task_path: &str) -> Result<Task, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(task_path)?)?)
}

pub(super) fn build_run_config(
    task: &Task,
    run_generation_prefill: bool,
) -> RunConfig {
    let tokens_limit = if run_generation_prefill {
        task.tokens_limit.max(1)
    } else {
        0
    };
    let mut run_config = RunConfig::default().tokens_limit(tokens_limit);
    if task.greedy {
        run_config = run_config.sampling_policy(SamplingPolicy::Custom {
            value: SamplingMethod::Greedy,
        });
    }
    run_config
}

fn select_rows(
    flat: &[f32],
    num_groups: usize,
    head_dim: usize,
    total_rows: usize,
    selected_rows: &[usize],
) -> Vec<f32> {
    let mut selected = Vec::with_capacity(num_groups * selected_rows.len() * head_dim);
    let group_stride = total_rows * head_dim;
    for group_index in 0..num_groups {
        let group_start = group_index * group_stride;
        for &row_index in selected_rows {
            let start = group_start + row_index * head_dim;
            let end = start + head_dim;
            selected.extend_from_slice(&flat[start..end]);
        }
    }
    selected
}

fn run_snapshot(
    model_path: &str,
    task: &Task,
    method: Option<&str>,
    turboquant_bits: Option<usize>,
    turboquant_target: Option<&str>,
    after_prefill: bool,
    generated_tokens_after_prefill: Option<usize>,
) -> Result<(KvDebugSnapshot, u64, u64, String), Box<dyn std::error::Error>> {
    let _env = install_kv_env(method, turboquant_bits, turboquant_target);
    let decoding_config = DecodingConfig::default();
    let mut session = ChatSession::new(PathBuf::from(model_path), decoding_config)?;
    let input = Input::Messages(task.messages.clone());
    let run_config = build_run_config(task, after_prefill || generated_tokens_after_prefill.is_some());
    let (output, snapshot) = if let Some(generated_tokens) = generated_tokens_after_prefill {
        session.run_capture_generated_kv_debug(input, run_config, generated_tokens)?
    } else if after_prefill {
        session.run_capture_prefill_kv_debug(input, run_config)?
    } else {
        session.run_capture_kv_debug(input, run_config)?
    };
    let kv_storage_bytes = session.kv_storage_bytes();
    Ok((snapshot, kv_storage_bytes, output.stats.total_stats.tokens_count_input, output.text.original))
}

fn compare_snapshots(
    method: String,
    stage: String,
    turboquant_bits: Option<usize>,
    prompt_tokens: u64,
    baseline_kv_storage_bytes: u64,
    candidate_kv_storage_bytes: u64,
    baseline_text: String,
    candidate_text: String,
    baseline: &KvDebugSnapshot,
    candidate: &KvDebugSnapshot,
    compare_positions_before: Option<usize>,
) -> FidelityReport {
    assert_eq!(baseline.layers.len(), candidate.layers.len(), "transformer layer count must match");

    let mut all_keys = MetricAccumulator::default();
    let mut all_values = MetricAccumulator::default();
    let mut all_recent_values = MetricAccumulator::default();
    let mut all_recent_consistency = MetricAccumulator::default();
    let mut all_pending_values = MetricAccumulator::default();
    let mut all_pending_consistency = MetricAccumulator::default();
    let mut saw_recent_values = false;
    let mut saw_recent_consistency = false;
    let mut saw_pending_values = false;
    let mut saw_pending_consistency = false;
    let mut layers = Vec::with_capacity(baseline.layers.len());
    let mut baseline_token_count = 0usize;
    let mut candidate_token_count = 0usize;
    let mut shared_token_count = 0usize;

    for (baseline_layer, candidate_layer) in baseline.layers.iter().zip(candidate.layers.iter()) {
        assert_eq!(baseline_layer.layer_index, candidate_layer.layer_index, "layer index mismatch");
        assert_eq!(baseline_layer.num_groups, candidate_layer.num_groups, "layer group count mismatch");
        assert_eq!(baseline_layer.head_dim, candidate_layer.head_dim, "layer head dim mismatch");

        let baseline_rows = baseline_layer
            .positions
            .iter()
            .enumerate()
            .filter_map(|(row_index, &position)| {
                compare_positions_before.map_or(true, |limit| position < limit).then_some(row_index)
            })
            .collect::<Vec<_>>();
        let candidate_rows = candidate_layer
            .positions
            .iter()
            .enumerate()
            .filter_map(|(row_index, &position)| {
                compare_positions_before.map_or(true, |limit| position < limit).then_some(row_index)
            })
            .collect::<Vec<_>>();
        let candidate_row_by_position = candidate_rows
            .iter()
            .map(|&row_index| (candidate_layer.positions[row_index], row_index))
            .collect::<HashMap<_, _>>();
        let shared_rows = baseline_rows
            .iter()
            .filter_map(|&baseline_row_index| {
                let position = baseline_layer.positions[baseline_row_index];
                candidate_row_by_position
                    .get(&position)
                    .copied()
                    .map(|candidate_row_index| (baseline_row_index, candidate_row_index))
            })
            .collect::<Vec<_>>();
        let shared_baseline_rows =
            shared_rows.iter().map(|&(baseline_row_index, _)| baseline_row_index).collect::<Vec<_>>();
        let shared_candidate_rows =
            shared_rows.iter().map(|&(_, candidate_row_index)| candidate_row_index).collect::<Vec<_>>();

        let baseline_keys = select_rows(
            &baseline_layer.keys,
            baseline_layer.num_groups,
            baseline_layer.head_dim,
            baseline_layer.positions.len(),
            &shared_baseline_rows,
        );
        let candidate_keys = select_rows(
            &candidate_layer.keys,
            candidate_layer.num_groups,
            candidate_layer.head_dim,
            candidate_layer.positions.len(),
            &shared_candidate_rows,
        );
        let baseline_values = select_rows(
            &baseline_layer.values,
            baseline_layer.num_groups,
            baseline_layer.head_dim,
            baseline_layer.positions.len(),
            &shared_baseline_rows,
        );
        let candidate_values = select_rows(
            &candidate_layer.values,
            candidate_layer.num_groups,
            candidate_layer.head_dim,
            candidate_layer.positions.len(),
            &shared_candidate_rows,
        );

        let mut key_acc = MetricAccumulator::default();
        let mut value_acc = MetricAccumulator::default();
        key_acc.update(&baseline_keys, &candidate_keys);
        value_acc.update(&baseline_values, &candidate_values);
        all_keys.update(&baseline_keys, &candidate_keys);
        all_values.update(&baseline_values, &candidate_values);
        let (recent_value, candidate_recent_consistency) = candidate_layer
            .sparse_recent_positions
            .as_ref()
            .zip(candidate_layer.sparse_recent_values.as_ref())
            .map(|(recent_positions, recent_values)| {
                let recent_row_by_position = recent_positions
                    .iter()
                    .enumerate()
                    .map(|(row_index, &position)| (position, row_index))
                    .collect::<HashMap<_, _>>();
                let shared_recent_rows = baseline_rows
                    .iter()
                    .filter_map(|&baseline_row_index| {
                        let position = baseline_layer.positions[baseline_row_index];
                        recent_row_by_position
                            .get(&position)
                            .copied()
                            .map(|recent_row_index| (baseline_row_index, recent_row_index))
                    })
                    .collect::<Vec<_>>();
                let recent_value = (!shared_recent_rows.is_empty()).then(|| {
                    let shared_baseline_recent_rows = shared_recent_rows
                        .iter()
                        .map(|&(baseline_row_index, _)| baseline_row_index)
                        .collect::<Vec<_>>();
                    let shared_candidate_recent_rows =
                        shared_recent_rows.iter().map(|&(_, recent_row_index)| recent_row_index).collect::<Vec<_>>();
                    let baseline_recent_values = select_rows(
                        &baseline_layer.values,
                        baseline_layer.num_groups,
                        baseline_layer.head_dim,
                        baseline_layer.positions.len(),
                        &shared_baseline_recent_rows,
                    );
                    let candidate_recent_values = select_rows(
                        recent_values,
                        candidate_layer.num_groups,
                        candidate_layer.head_dim,
                        recent_positions.len(),
                        &shared_candidate_recent_rows,
                    );
                    let mut recent_acc = MetricAccumulator::default();
                    recent_acc.update(&baseline_recent_values, &candidate_recent_values);
                    all_recent_values.update(&baseline_recent_values, &candidate_recent_values);
                    saw_recent_values = true;
                    recent_acc.finish()
                });

                let candidate_shared_recent_rows = candidate_layer
                    .positions
                    .iter()
                    .enumerate()
                    .filter_map(|(row_index, &position)| {
                        recent_row_by_position
                            .get(&position)
                            .copied()
                            .map(|recent_row_index| (row_index, recent_row_index))
                    })
                    .collect::<Vec<_>>();
                let candidate_recent_consistency = (!candidate_shared_recent_rows.is_empty()).then(|| {
                    let candidate_value_rows =
                        candidate_shared_recent_rows.iter().map(|&(row_index, _)| row_index).collect::<Vec<_>>();
                    let candidate_recent_rows =
                        candidate_shared_recent_rows.iter().map(|&(_, row_index)| row_index).collect::<Vec<_>>();
                    let candidate_value_rows = select_rows(
                        &candidate_layer.values,
                        candidate_layer.num_groups,
                        candidate_layer.head_dim,
                        candidate_layer.positions.len(),
                        &candidate_value_rows,
                    );
                    let candidate_recent_values = select_rows(
                        recent_values,
                        candidate_layer.num_groups,
                        candidate_layer.head_dim,
                        recent_positions.len(),
                        &candidate_recent_rows,
                    );
                    let mut consistency_acc = MetricAccumulator::default();
                    consistency_acc.update(&candidate_value_rows, &candidate_recent_values);
                    all_recent_consistency.update(&candidate_value_rows, &candidate_recent_values);
                    saw_recent_consistency = true;
                    consistency_acc.finish()
                });

                (recent_value, candidate_recent_consistency)
            })
            .unwrap_or((None, None));
        let (pending_value, candidate_pending_consistency) = candidate_layer
            .sparse_pending_positions
            .as_ref()
            .zip(candidate_layer.sparse_pending_values.as_ref())
            .map(|(pending_positions, pending_values)| {
                let pending_row_by_position = pending_positions
                    .iter()
                    .enumerate()
                    .map(|(row_index, &position)| (position, row_index))
                    .collect::<HashMap<_, _>>();
                let shared_pending_rows = baseline_rows
                    .iter()
                    .filter_map(|&baseline_row_index| {
                        let position = baseline_layer.positions[baseline_row_index];
                        pending_row_by_position
                            .get(&position)
                            .copied()
                            .map(|pending_row_index| (baseline_row_index, pending_row_index))
                    })
                    .collect::<Vec<_>>();
                let pending_value = (!shared_pending_rows.is_empty()).then(|| {
                    let shared_baseline_pending_rows = shared_pending_rows
                        .iter()
                        .map(|&(baseline_row_index, _)| baseline_row_index)
                        .collect::<Vec<_>>();
                    let shared_candidate_pending_rows =
                        shared_pending_rows.iter().map(|&(_, pending_row_index)| pending_row_index).collect::<Vec<_>>();
                    let baseline_pending_values = select_rows(
                        &baseline_layer.values,
                        baseline_layer.num_groups,
                        baseline_layer.head_dim,
                        baseline_layer.positions.len(),
                        &shared_baseline_pending_rows,
                    );
                    let candidate_pending_values = select_rows(
                        pending_values,
                        candidate_layer.num_groups,
                        candidate_layer.head_dim,
                        pending_positions.len(),
                        &shared_candidate_pending_rows,
                    );
                    let mut pending_acc = MetricAccumulator::default();
                    pending_acc.update(&baseline_pending_values, &candidate_pending_values);
                    all_pending_values.update(&baseline_pending_values, &candidate_pending_values);
                    saw_pending_values = true;
                    pending_acc.finish()
                });

                let candidate_shared_pending_rows = candidate_layer
                    .positions
                    .iter()
                    .enumerate()
                    .filter_map(|(row_index, &position)| {
                        pending_row_by_position
                            .get(&position)
                            .copied()
                            .map(|pending_row_index| (row_index, pending_row_index))
                    })
                    .collect::<Vec<_>>();
                let candidate_pending_consistency = (!candidate_shared_pending_rows.is_empty()).then(|| {
                    let candidate_value_rows =
                        candidate_shared_pending_rows.iter().map(|&(row_index, _)| row_index).collect::<Vec<_>>();
                    let candidate_pending_rows =
                        candidate_shared_pending_rows.iter().map(|&(_, row_index)| row_index).collect::<Vec<_>>();
                    let candidate_value_rows = select_rows(
                        &candidate_layer.values,
                        candidate_layer.num_groups,
                        candidate_layer.head_dim,
                        candidate_layer.positions.len(),
                        &candidate_value_rows,
                    );
                    let candidate_pending_values = select_rows(
                        pending_values,
                        candidate_layer.num_groups,
                        candidate_layer.head_dim,
                        pending_positions.len(),
                        &candidate_pending_rows,
                    );
                    let mut consistency_acc = MetricAccumulator::default();
                    consistency_acc.update(&candidate_value_rows, &candidate_pending_values);
                    all_pending_consistency.update(&candidate_value_rows, &candidate_pending_values);
                    saw_pending_consistency = true;
                    consistency_acc.finish()
                });

                (pending_value, candidate_pending_consistency)
            })
            .unwrap_or((None, None));

        baseline_token_count += baseline_rows.len();
        candidate_token_count += candidate_rows.len();
        shared_token_count += shared_rows.len();

        layers.push(FidelityLayerResult {
            layer_index: baseline_layer.layer_index,
            baseline_token_count: baseline_rows.len(),
            candidate_token_count: candidate_rows.len(),
            shared_token_count: shared_rows.len(),
            shared_token_ratio: if baseline_rows.is_empty() {
                0.0
            } else {
                shared_rows.len() as f64 / baseline_rows.len() as f64
            },
            baseline_storage_bytes: baseline_layer.storage_bytes,
            candidate_storage_bytes: candidate_layer.storage_bytes,
            compression_ratio: baseline_layer.storage_bytes as f64 / candidate_layer.storage_bytes as f64,
            key: key_acc.finish(),
            value: value_acc.finish(),
            recent_value,
            candidate_recent_consistency,
            pending_value,
            candidate_pending_consistency,
        });
    }

    FidelityReport {
        method,
        stage,
        turboquant_bits,
        prompt_tokens,
        baseline_kv_storage_bytes,
        candidate_kv_storage_bytes,
        compression_ratio: baseline_kv_storage_bytes as f64 / candidate_kv_storage_bytes as f64,
        baseline_token_count,
        candidate_token_count,
        shared_token_count,
        shared_token_ratio: if baseline_token_count == 0 {
            0.0
        } else {
            shared_token_count as f64 / baseline_token_count as f64
        },
        baseline_text,
        candidate_text,
        key: all_keys.finish(),
        value: all_values.finish(),
        recent_value: saw_recent_values.then(|| all_recent_values.finish()),
        candidate_recent_consistency: saw_recent_consistency.then(|| all_recent_consistency.finish()),
        pending_value: saw_pending_values.then(|| all_pending_values.finish()),
        candidate_pending_consistency: saw_pending_consistency.then(|| all_pending_consistency.finish()),
        layers,
    }
}

pub fn handle_kv_fidelity(
    model_path: String,
    task_path: String,
    output_path: String,
    method: String,
    turboquant_bits: Option<usize>,
    turboquant_target: Option<String>,
    after_prefill: bool,
    after_first_generate: bool,
    after_generate_tokens: Option<usize>,
    compare_prompt_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let generated_tokens_after_prefill = if after_first_generate {
        assert!(after_generate_tokens.is_none(), "use either --after-first-generate or --after-generate-tokens");
        Some(1)
    } else {
        after_generate_tokens
    };
    assert!(!(after_prefill && generated_tokens_after_prefill.is_some()), "fidelity stages are mutually exclusive");
    let task = load_task(&task_path)?;
    let (baseline_snapshot, baseline_kv_storage_bytes, prompt_tokens, baseline_text) =
        run_snapshot(&model_path, &task, None, None, None, after_prefill, generated_tokens_after_prefill)?;
    let (candidate_snapshot, candidate_kv_storage_bytes, candidate_prompt_tokens, candidate_text) = run_snapshot(
        &model_path,
        &task,
        Some(&method),
        turboquant_bits,
        turboquant_target.as_deref(),
        after_prefill,
        generated_tokens_after_prefill,
    )?;

    assert_eq!(prompt_tokens, candidate_prompt_tokens, "prompt token count must match");

    let stage = if let Some(generated_tokens) = generated_tokens_after_prefill {
        format!("after_generate_tokens_{generated_tokens}")
    } else if after_prefill {
        "generation_prefill".to_string()
    } else {
        "prefill_only".to_string()
    };
    let compare_positions_before = if generated_tokens_after_prefill.is_some() && compare_prompt_only {
        Some(prompt_tokens as usize)
    } else if let Some(generated_tokens) = generated_tokens_after_prefill {
        Some(prompt_tokens as usize + generated_tokens)
    } else if after_prefill {
        Some(prompt_tokens as usize)
    } else {
        None
    };

    let report = compare_snapshots(
        method,
        stage,
        turboquant_bits,
        prompt_tokens,
        baseline_kv_storage_bytes,
        candidate_kv_storage_bytes,
        baseline_text,
        candidate_text,
        &baseline_snapshot,
        &candidate_snapshot,
        compare_positions_before,
    );

    fs::write(output_path, serde_json::to_string_pretty(&report)?)?;

    println!("Stage: {}", report.stage);
    println!("Prompt tokens: {}", report.prompt_tokens);
    println!("KV compression ratio: {:.3}x", report.compression_ratio);
    println!(
        "Shared positions: {}/{} ({:.3})",
        report.shared_token_count, report.baseline_token_count, report.shared_token_ratio
    );
    println!(
        "Keys  cosine={:.6} mean_abs={:.6} max_abs={:.6}",
        report.key.cosine, report.key.mean_abs, report.key.max_abs
    );
    println!(
        "Values cosine={:.6} mean_abs={:.6} max_abs={:.6}",
        report.value.cosine, report.value.mean_abs, report.value.max_abs
    );
    if let Some(recent_value) = &report.recent_value {
        println!(
            "Recent cosine={:.6} mean_abs={:.6} max_abs={:.6}",
            recent_value.cosine, recent_value.mean_abs, recent_value.max_abs
        );
    }
    if let Some(recent_consistency) = &report.candidate_recent_consistency {
        println!(
            "Candidate recent/materialized cosine={:.6} mean_abs={:.6} max_abs={:.6}",
            recent_consistency.cosine, recent_consistency.mean_abs, recent_consistency.max_abs
        );
    }
    if let Some(pending_value) = &report.pending_value {
        println!(
            "Pending cosine={:.6} mean_abs={:.6} max_abs={:.6}",
            pending_value.cosine, pending_value.mean_abs, pending_value.max_abs
        );
    }
    if let Some(pending_consistency) = &report.candidate_pending_consistency {
        println!(
            "Candidate pending/materialized cosine={:.6} mean_abs={:.6} max_abs={:.6}",
            pending_consistency.cosine, pending_consistency.mean_abs, pending_consistency.max_abs
        );
    }
    if after_prefill || generated_tokens_after_prefill.is_some() {
        println!("Baseline prefill text: {}", report.baseline_text);
        println!("Candidate prefill text: {}", report.candidate_text);
    }

    Ok(())
}
