use std::{
    ops::Add,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use comfy_table::{CellAlignment, ContentArrangement, Table, presets::UTF8_FULL};
use indexmap::IndexMap;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use tokio::fs;
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::{
        chat::ReplyEnergy,
        decision::{DecisionInput, DecisionResult},
    },
};
use uzu_engine::VERSION;

use crate::{
    bench::{model::BenchDevice, stat::calculate_metric},
    common::decision::DecisionRequestQuestion,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionBenchTask {
    pub identifier: String,
    pub repo_id: String,
    pub number_of_runs: u64,
    pub state: String,
    // Keyed by question id, as in the System One request.
    pub questions: IndexMap<String, DecisionRequestQuestion>,
}

// Totals over all questions of one run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionBenchResult {
    pub task: DecisionBenchTask,
    pub device: BenchDevice,
    pub engine_version: String,
    pub timestamp: u64,
    pub tokens_count: u64,
    pub prefill_duration: f64,
    pub total_duration: f64,
    pub prefill_tokens_per_second: Option<f64>,
    pub energy: Option<ReplyEnergy>,
    pub joules_per_token: Option<f64>,
    pub results: IndexMap<String, DecisionResult>,
}

pub async fn run(
    task: DecisionBenchTask,
    model_path: String,
    output_path: String,
) -> Result<()> {
    println!("Model: {}", task.repo_id);

    let model_path = kiban::fs::asyn::canonicalize(&model_path)
        .await
        .with_context(|| format!("Can not open model path: {model_path}"))?
        .to_string_lossy()
        .into_owned();
    let engine = Engine::new(EngineConfig::default()).await.context("Can not create engine")?;
    let model = engine
        .model_by_path(model_path.clone())
        .await?
        .with_context(|| format!("Model not found at path: {model_path}"))?;
    let session = engine.decision(model).await.context("Can not create decision session")?;

    let input = DecisionInput {
        state: task.state.clone(),
        questions: task.questions.values().cloned().map(Into::into).collect(),
    };
    let device = BenchDevice::current();

    // The first decision pays for kernel setup.
    session.decide(input.clone()).await?;

    let progress_bar = ProgressBar::new(task.number_of_runs);
    let mut results = Vec::<DecisionBenchResult>::new();
    for _ in 0..task.number_of_runs {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let output = session.decide(input.clone()).await?;

        let tokens_count = output.results.iter().map(|result| result.stats.tokens_count as u64).sum::<u64>();
        let prefill_duration = output.results.iter().map(|result| result.stats.prefill_duration).sum::<f64>();
        let total_duration = output.results.iter().map(|result| result.stats.total_duration).sum::<f64>();
        let prefill_tokens_per_second = (prefill_duration > 0.0).then_some(tokens_count as f64 / prefill_duration);
        // A partial sum would understate the run, so energy is reported only when every question has it.
        let energy = output
            .results
            .iter()
            .map(|result| result.stats.energy.clone())
            .collect::<Option<Vec<_>>>()
            .and_then(|energy| energy.into_iter().reduce(ReplyEnergy::add));
        let joules_per_token =
            energy.as_ref().and_then(|energy| (tokens_count > 0).then(|| energy.total() / tokens_count as f64));

        results.push(DecisionBenchResult {
            task: task.clone(),
            device: device.clone(),
            engine_version: VERSION.to_string(),
            timestamp,
            tokens_count,
            prefill_duration,
            total_duration,
            prefill_tokens_per_second,
            energy,
            joules_per_token,
            results: task.questions.keys().cloned().zip(output.results).collect(),
        });
        progress_bar.inc(1);
    }
    progress_bar.finish();

    let results_data = serde_json::to_string_pretty(&results).context("Failed to serialize benchmark results")?;
    fs::write(&output_path, results_data)
        .await
        .with_context(|| format!("Failed to write output file: {output_path}"))?;

    let Some(first_result) = results.first() else {
        bail!("Benchmark produced no results");
    };
    println!("Questions count: {}", task.questions.len());
    println!("Tokens count: {}", first_result.tokens_count);

    let prefill_duration_list = results.iter().map(|result| result.prefill_duration).collect::<Vec<f64>>();
    let total_duration_list = results.iter().map(|result| result.total_duration).collect::<Vec<f64>>();
    let prefill_tokens_per_second_list =
        results.iter().filter_map(|result| result.prefill_tokens_per_second).collect::<Vec<f64>>();
    let energy_joules_list =
        results.iter().filter_map(|result| result.energy.as_ref().map(ReplyEnergy::total)).collect::<Vec<f64>>();
    let joules_per_token_list = results.iter().filter_map(|result| result.joules_per_token).collect::<Vec<f64>>();

    let prefill_duration_metric = calculate_metric(&prefill_duration_list);
    let total_duration_metric = calculate_metric(&total_duration_list);
    let prefill_tokens_per_second_metric = calculate_metric(&prefill_tokens_per_second_list);
    let energy_joules_metric = calculate_metric(&energy_joules_list);
    let joules_per_token_metric = calculate_metric(&joules_per_token_list);

    let mut table = Table::new();
    table
        .load_style(UTF8_FULL.with_rounded_corners())
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Metric", "Value"])
        .add_row(vec!["Prefill, s", prefill_duration_metric.as_str()])
        .add_row(vec!["Total, s", total_duration_metric.as_str()])
        .add_row(vec!["Prefill, t/s", prefill_tokens_per_second_metric.as_str()])
        .add_row(vec!["Total energy, J", energy_joules_metric.as_str()])
        .add_row(vec!["Energy per token, J/tok", joules_per_token_metric.as_str()]);
    let Some(column) = table.column_mut(1) else {
        bail!("Benchmark summary table value column is missing");
    };
    column.set_cell_alignment(CellAlignment::Right);
    println!("{table}");

    Ok(())
}
