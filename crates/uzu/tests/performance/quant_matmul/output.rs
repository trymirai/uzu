use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct PerfResult {
    pub combo: String,
    pub shape: String,
    pub dispatch_path: String,
    pub duration_ms: f64,
    pub gflops: f64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn print_results_table(results: &[PerfResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Config", "Shape (MxKxN)", "Dispatch", "GFLOPS", "ms/iter", "Status"]);

    for r in results {
        let (gflops_str, ms_str, status_str) = if r.status == "ok" {
            (format!("{:.1}", r.gflops), format!("{:.3}", r.duration_ms), "ok".into())
        } else {
            ("-".into(), "-".into(), format!("ERR: {}", r.error.as_deref().unwrap_or("?")))
        };
        table.add_row(vec![r.combo.clone(), r.shape.clone(), r.dispatch_path.clone(), gflops_str, ms_str, status_str]);
    }

    for col in [3, 4] {
        if let Some(column) = table.column_mut(col) {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }

    println!("{table}");
}

pub fn print_comparison_table(
    results: &[PerfResult],
    batches: &[usize],
) {
    let mut dispatch_paths: Vec<String> = Vec::new();
    for r in results {
        if r.status == "ok" && !dispatch_paths.contains(&r.dispatch_path) {
            dispatch_paths.push(r.dispatch_path.clone());
        }
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic);

    let mut header = vec!["M".to_string()];
    for dp in &dispatch_paths {
        header.push(dp.clone());
    }
    table.set_header(header);

    for &batch in batches {
        let mut row = vec![format!("{batch}")];
        for dp in &dispatch_paths {
            let cell = results
                .iter()
                .find(|r| r.dispatch_path == *dp && r.status == "ok" && r.shape.starts_with(&format!("{batch}x")))
                .map(|r| format!("{:.3}", r.duration_ms))
                .unwrap_or_else(|| "-".into());
            row.push(cell);
        }
        table.add_row(row);
    }

    for col in 1..=dispatch_paths.len() {
        if let Some(column) = table.column_mut(col) {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }

    println!("\nComparison: Latency (ms) by M x kernel");
    println!("{table}");
}
