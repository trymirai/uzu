use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use serde::Serialize;

#[derive(Serialize)]
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

pub fn write_json_results(
    test_name: &str,
    device: &str,
    results: &[PerfResult],
) {
    if let Ok(dir) = std::env::var("UZU_TEST_RESULTS_DIR") {
        let path = std::path::Path::new(&dir);
        std::fs::create_dir_all(path).expect("create results dir");
        let file = path.join(format!("{test_name}.json"));
        let wrapper = serde_json::json!({ "device": device, "results": results });
        let json = serde_json::to_string_pretty(&wrapper).expect("serialize");
        std::fs::write(&file, json).expect("write results");
        eprintln!("Results written to {}", file.display());
    }
}

pub fn print_results_table(results: &[PerfResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Dtype Combo", "Shape (MxKxN)", "Dispatch", "GFLOPS", "ms/iter", "Status"]);

    for r in results {
        let (gflops_str, ms_str, status_str) = if r.status == "ok" {
            (format!("{:.1}", r.gflops), format!("{:.3}", r.duration_ms), "ok".into())
        } else {
            ("-".into(), "-".into(), format!("ERR: {}", r.error.as_deref().unwrap_or("?")))
        };
        table.add_row(vec![
            r.combo.clone(),
            r.shape.clone(),
            r.dispatch_path.clone(),
            gflops_str,
            ms_str,
            status_str,
        ]);
    }

    for col in [3, 4] {
        if let Some(column) = table.column_mut(col) {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }

    println!("{table}");
}
