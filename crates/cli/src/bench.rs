use std::fs;

use anyhow::{Context, Result, bail};
use benchmarks::runner::{
    Runner,
    types::{Result as BenchmarkResult, Task},
};
use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::ProgressBar;

pub fn run_bench(
    model_path: String,
    task_path: String,
    output_path: String,
) -> Result<()> {
    let task_data = fs::read_to_string(&task_path).with_context(|| format!("Failed to read task file: {task_path}"))?;
    let task: Task =
        serde_json::from_str(&task_data).with_context(|| format!("Failed to parse task file: {task_path}"))?;

    println!("Model: {}", task.repo_id);

    let progress_bar = ProgressBar::new(task.number_of_runs);
    progress_bar.set_position(0);

    let runner = Runner::new(task.clone(), model_path, None);
    let results = runner
        .run(Some(|_progress| {
            progress_bar.inc(1);
        }))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    progress_bar.finish();

    let results_data = serde_json::to_string_pretty(&results).context("Failed to serialize benchmark results")?;
    fs::write(&output_path, results_data).with_context(|| format!("Failed to write output file: {output_path}"))?;

    print_summary(&results)
}

fn print_summary(results: &[BenchmarkResult]) -> Result<()> {
    let Some(first_result) = results.first() else {
        bail!("Benchmark produced no results");
    };

    println!("Input tokens count: {}", first_result.tokens_count_input);

    let memory_used_list = results
        .iter()
        .filter_map(|result| result.memory_used)
        .map(|memory_used| (memory_used as f64) / 1024.0 / 1024.0 / 1024.0)
        .collect::<Vec<f64>>();
    let time_to_first_token_list = results.iter().map(|result| result.time_to_first_token).collect::<Vec<f64>>();
    let prompt_tokens_per_second_list =
        results.iter().map(|result| result.prompt_tokens_per_second).collect::<Vec<f64>>();
    let generate_tokens_per_second_list =
        results.iter().filter_map(|result| result.generate_tokens_per_second).collect::<Vec<f64>>();

    let memory_used_metric = calculate_metric(&memory_used_list);
    let time_to_first_token_metric = calculate_metric(&time_to_first_token_list);
    let prompt_tokens_per_second_metric = calculate_metric(&prompt_tokens_per_second_list);
    let generate_tokens_per_second_metric = calculate_metric(&generate_tokens_per_second_list);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Metric", "Value"])
        .add_row(vec!["Used memory, GB", memory_used_metric.as_str()])
        .add_row(vec!["TTFT, s", time_to_first_token_metric.as_str()])
        .add_row(vec!["Prompt, t/s", prompt_tokens_per_second_metric.as_str()])
        .add_row(vec!["Generate, t/s", generate_tokens_per_second_metric.as_str()]);

    let Some(column) = table.column_mut(1) else {
        bail!("Benchmark summary table value column is missing");
    };
    column.set_cell_alignment(CellAlignment::Right);

    println!("{table}");

    Ok(())
}

fn calculate_metric(data: &[f64]) -> String {
    let mean = mean(data);
    let standard_deviation = standard_deviation(data);
    if let (Some(mean), Some(standard_deviation)) = (mean, standard_deviation) {
        format!("{mean:.3} ± {standard_deviation:.3}")
    } else if let Some(mean) = mean {
        format!("{mean:.3}")
    } else {
        "-".to_string()
    }
}

fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    Some(data.iter().sum::<f64>() / data.len() as f64)
}

fn standard_deviation(data: &[f64]) -> Option<f64> {
    let samples_count = data.len();
    if samples_count < 2 {
        return None;
    }

    let mean = mean(data)?;

    let variance = data
        .iter()
        .map(|value| {
            let difference = value - mean;
            difference * difference
        })
        .sum::<f64>()
        / (samples_count as f64 - 1.0);

    Some(variance.sqrt())
}
