use std::fs;

use benchmarks::matmul_runner::{MatmulBenchmarkTask, MatmulRunner};
use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::ProgressBar;

pub fn handle_bench_matmul(
    task_path: Option<String>,
    output_path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let task = match task_path {
        Some(path) => {
            let task_data = fs::read_to_string(path)?;
            serde_json::from_str(&task_data)?
        },
        None => {
            eprintln!("[bench_matmul] Using default MPP task grid");
            MatmulBenchmarkTask::default_mpp()
        },
    };

    let total_cases = task.combos.len() * task.shapes.len();
    eprintln!(
        "[bench_matmul] {} combos x {} shapes = {} total cases, {} warmup + {} benchmark iterations each",
        task.combos.len(),
        task.shapes.len(),
        total_cases,
        task.warmup_iterations,
        task.benchmark_iterations,
    );

    let progress_bar = ProgressBar::new(total_cases as u64);
    progress_bar.set_position(0);

    let runner = MatmulRunner::new(task);
    let results = runner.run(Some(|_: f64| {
        progress_bar.inc(1);
    }))?;
    progress_bar.finish();

    let results_data = serde_json::to_string_pretty(&results)?;
    fs::write(&output_path, &results_data)?;
    eprintln!("[bench_matmul] Results written to {output_path}");

    let ok_results: Vec<_> = results.iter().filter(|r| r.status == "ok").collect();
    let error_results: Vec<_> = results.iter().filter(|r| r.status != "ok").collect();

    if !ok_results.is_empty() {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["Dtype Combo", "Shape", "GFLOPS", "ms/iter"]);

        for r in &ok_results {
            table.add_row(vec![
                format!("{}", r.combo),
                format!("{}", r.shape),
                format!("{:.1}", r.gflops),
                format!("{:.2}", r.duration_ms),
            ]);
        }

        let column = table.column_mut(2).expect("Column not found");
        column.set_cell_alignment(CellAlignment::Right);
        let column = table.column_mut(3).expect("Column not found");
        column.set_cell_alignment(CellAlignment::Right);

        println!("{table}");
    }

    if !error_results.is_empty() {
        eprintln!("\n{} cases failed:", error_results.len());
        for r in &error_results {
            eprintln!(
                "  {} {}: {}",
                r.combo,
                r.shape,
                r.error_message.as_deref().unwrap_or("unknown error")
            );
        }
    }

    eprintln!(
        "\nSummary: {} ok, {} errors out of {} total",
        ok_results.len(),
        error_results.len(),
        results.len()
    );

    Ok(())
}
