use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use serde::Serialize;

#[derive(Serialize)]
pub struct TestResult {
    pub combo: String,
    pub shape: String,
    pub dispatch_path: String,
    pub passed: bool,
    pub max_diff: f64,
    pub tolerance: f64,
}

pub fn print_results_table(results: &[TestResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Dtype Combo", "Shape (BxKxN)", "Dispatch", "Status", "Max Diff", "Tolerance"]);

    for result in results {
        table.add_row(vec![
            result.combo.clone(),
            result.shape.clone(),
            result.dispatch_path.clone(),
            if result.passed { "PASS".into() } else { "FAIL".into() },
            format!("{:.6}", result.max_diff),
            format!("{:.6}", result.tolerance),
        ]);
    }

    for col in [4, 5] {
        if let Some(column) = table.column_mut(col) {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }

    println!("{table}");
}
