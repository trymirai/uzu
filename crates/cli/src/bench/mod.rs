mod model;
mod runner;
mod stat;

use anyhow::{Context, Result, bail};
use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::ProgressBar;
use tokio::fs;

use crate::bench::{model::BenchTask, runner::BenchRunner, stat::calculate_metric};

pub async fn run_bench(
    model_path: String,
    task_path: String,
    output_path: String,
) -> Result<()> {
    let task_content =
        fs::read_to_string(&task_path).await.with_context(|| format!("Failed to read task file: {task_path}"))?;
    let task: BenchTask =
        serde_json::from_str(&task_content).with_context(|| format!("Failed to parse task file: {task_path}"))?;
    println!("Model: {}", task.repo_id);

    let progress_bar = ProgressBar::new(task.number_of_runs);
    progress_bar.set_position(0);

    let runner = BenchRunner::new(task.clone(), model_path);
    let results = runner
        .run(Some(|_progress| {
            progress_bar.inc(1);
        }))
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    progress_bar.finish();

    // Results
    let results_data = serde_json::to_string_pretty(&results).context("Failed to serialize benchmark results")?;
    fs::write(&output_path, results_data)
        .await
        .with_context(|| format!("Failed to write output file: {output_path}"))?;

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
