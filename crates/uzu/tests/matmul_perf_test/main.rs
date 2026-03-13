#![cfg(target_os = "macos")]

mod bench;
#[path = "../common/mod.rs"]
mod common;
mod error;
mod output;
mod shapes;

use common::matmul::{TestShape, applicable_variants, test_combos};
use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLDeviceExt, MTLResourceOptions};
use output::print_results_table;
use shapes::test_shapes;
use uzu::backends::{
    common::Context,
    metal::MetalContext,
};

#[test]
#[ignore]
fn matmul_perf() {
    let context = MetalContext::new().expect("Metal context required");

    let dtype_combos = test_combos();
    let test_shapes = test_shapes();

    eprintln!(
        "Matmul perf: {} combos x {} shapes, testing all applicable dispatch paths",
        dtype_combos.len(),
        test_shapes.len(),
    );

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::with_template("{spinner} {pos} benchmarks [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::new();

    for combo in &dtype_combos {
        for shape in &test_shapes {
            let variants = applicable_variants(&context, combo, shape);

            for (path_name, variant) in &variants {
                progress_bar.set_message(format!("{} {} {}", combo, shape, path_name));
                let result = bench::benchmark_single(&context, combo, shape, path_name, *variant);
                results.push(result);
                progress_bar.inc(1);
            }
        }
    }

    progress_bar.finish_with_message("done");
    print_results_table(&results);
    common::matmul::write_json_results("matmul_perf", &context.device.name(), &results);

    let error_results: Vec<_> = results.iter().filter(|r| r.status != "ok").collect();
    if !error_results.is_empty() {
        eprintln!("{} / {} cases had errors", error_results.len(), results.len());
    }
}
