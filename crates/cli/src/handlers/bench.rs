use std::{collections::BTreeSet, env, fs, path::PathBuf, process::Command};

use benchmarks::{
    runner::Runner,
    types::{BenchRunMode, Result as TaskResult, Task},
};
use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::ProgressBar;
use uuid::Uuid;

pub fn handle_bench(
    model_path: String,
    task_path: String,
    output_path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let task_data = fs::read_to_string(task_path)?;
    let task: Task = serde_json::from_str(&task_data)?;

    println!("Model: {}", task.repo_id);
    let results = match task.run_mode {
        BenchRunMode::FreshProcess => run_fresh_process(&model_path, &task)?,
        BenchRunMode::WarmedProcess | BenchRunMode::FreshSession => run_in_process(&model_path, &task)?,
    };
    write_results(&output_path, &results)?;
    print_summary(&results);

    Ok(())
}

fn run_in_process(
    model_path: &str,
    task: &Task,
) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>> {
    let progress_bar = ProgressBar::new(task.number_of_runs);
    let runner = Runner::new(task.clone(), model_path.to_string(), None);
    let results = runner.run(Some(|_: f64| {
        progress_bar.inc(1);
    }))?;
    progress_bar.finish();
    Ok(results)
}

fn run_fresh_process(
    model_path: &str,
    task: &Task,
) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>> {
    let progress_bar = ProgressBar::new(task.number_of_runs);
    let temp_dir = env::temp_dir().join(format!("uzu-bench-{}", Uuid::new_v4()));
    fs::create_dir(&temp_dir)?;
    let mut results = Vec::with_capacity(task.number_of_runs as usize);
    for run_index in 0..task.number_of_runs {
        let child_task_path = temp_dir.join(format!("task-{run_index}.json"));
        let child_output_path = temp_dir.join(format!("result-{run_index}.json"));
        let child_task = Task {
            number_of_runs: 1,
            run_mode: BenchRunMode::FreshSession,
            ..task.clone()
        };
        write_results(&child_task_path, &child_task)?;
        let output = Command::new(env::current_exe()?)
            .arg("bench")
            .arg(model_path)
            .arg(&child_task_path)
            .arg(&child_output_path)
            .output()?;
        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("fresh-process bench run failed\nstdout:\n{stdout}\nstderr:\n{stderr}").into());
        }
        let mut child_results: Vec<TaskResult> = read_json(&child_output_path)?;
        assert!(child_results.len() == 1, "fresh-process child must produce exactly one result");
        let mut child_result = child_results.pop().expect("fresh-process child produced no result");
        child_result.task = task.clone();
        results.push(child_result);
        progress_bar.inc(1);
    }
    progress_bar.finish();
    fs::remove_dir_all(temp_dir)?;
    Ok(results)
}

fn write_results<T: serde::Serialize>(
    output_path: impl Into<PathBuf>,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = output_path.into();
    let data = serde_json::to_string_pretty(value)?;
    fs::write(output_path, data)?;
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: impl Into<PathBuf>) -> Result<T, Box<dyn std::error::Error>> {
    let path = path.into();
    let data = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}

fn print_summary(results: &[TaskResult]) {
    let tokens_count_input = results.first().expect("No results").tokens_count_input;
    println!("Input tokens count: {}", tokens_count_input);

    let memory_used_list = results
        .iter()
        .filter_map(|result| result.memory_used)
        .map(|memory_used| (memory_used as f64) / 1024.0 / 1024.0 / 1024.0)
        .collect::<Vec<f64>>();
    let kv_storage_list =
        results.iter().map(|result| (result.kv_storage_bytes as f64) / 1024.0 / 1024.0 / 1024.0).collect::<Vec<f64>>();
    let time_to_first_token_list = results.iter().map(|result| result.time_to_first_token).collect::<Vec<f64>>();
    let prompt_tokens_per_second_list =
        results.iter().map(|result| result.prompt_tokens_per_second).collect::<Vec<f64>>();
    let generate_tokens_per_second_list =
        results.iter().filter_map(|result| result.generate_tokens_per_second).collect::<Vec<f64>>();

    let memory_used_metric = calculate_metric(memory_used_list);
    let kv_storage_metric = calculate_metric(kv_storage_list);
    let time_to_first_token_metric = calculate_metric(time_to_first_token_list);
    let prompt_tokens_per_second_metric = calculate_metric(prompt_tokens_per_second_list);
    let generate_tokens_per_second_metric = calculate_metric(generate_tokens_per_second_list);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Metric", "Mean ± Std", "Median"])
        .add_row(metric_row("Used memory, GB", &memory_used_metric))
        .add_row(metric_row("KV storage, GB", &kv_storage_metric))
        .add_row(metric_row("TTFT, s", &time_to_first_token_metric))
        .add_row(metric_row("Prompt, t/s", &prompt_tokens_per_second_metric))
        .add_row(metric_row("Generate, t/s", &generate_tokens_per_second_metric));
    for index in [1, 2] {
        let column = table.column_mut(index).expect("Column not found");
        column.set_cell_alignment(CellAlignment::Right);
    }
    println!("{table}");

    let unique_outputs = results.iter().map(|result| result.text_blake3.as_str()).collect::<BTreeSet<_>>();
    println!("Unique outputs: {}", unique_outputs.len());

    if let Some(reference) = results.first().and_then(|result| result.task.reference_text_blake3.as_ref()) {
        let exact_matches = results.iter().filter(|result| result.matches_reference_output == Some(true)).count();
        println!("Reference blake3: {reference}");
        println!("Exact runs: {exact_matches}/{}", results.len());
    }
}

struct SummaryMetric {
    mean_std: String,
    median: String,
}

fn metric_row(
    name: &'static str,
    metric: &SummaryMetric,
) -> Vec<String> {
    vec![name.to_string(), metric.mean_std.clone(), metric.median.clone()]
}

fn calculate_metric(data: Vec<f64>) -> SummaryMetric {
    SummaryMetric {
        mean_std: match (mean(&data), std_dev(&data)) {
            (Some(mean), Some(std_dev)) => format!("{mean:.3} ± {std_dev:.3}"),
            (Some(mean), None) => format!("{mean:.3}"),
            (None, _) => "-".to_string(),
        },
        median: median(&data).map(|value| format!("{value:.3}")).unwrap_or_else(|| "-".to_string()),
    }
}

fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    Some(data.iter().sum::<f64>() / data.len() as f64)
}

fn std_dev(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 2 {
        return None;
    }

    let mean = mean(data)?;

    let variance = data
        .iter()
        .map(|x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / (n as f64 - 1.0);

    Some(variance.sqrt())
}

fn median(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("benchmark metric must be finite"));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] + sorted[mid]) * 0.5)
    } else {
        Some(sorted[mid])
    }
}
