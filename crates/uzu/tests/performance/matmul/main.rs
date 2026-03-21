#![cfg(all(target_os = "macos", metal_backend))]

use indicatif::{ProgressBar, ProgressStyle};
use metal::MTLDeviceExt;
use uzu::{
    DataType,
    backends::{
        common::{Backend, Context},
        metal::Metal,
    },
};

use crate::matmul::{bench, output::print_results_table, shapes::test_shapes};

type Ctx = <Metal as Backend>::Context;

fn write_json_results<T: serde::Serialize>(
    test_name: &str,
    device: &str,
    results: &[T],
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

#[test]
#[ignore]
fn matmul_perf() {
    let context = Ctx::new().expect("Metal context required");

    let data_types = [DataType::BF16, DataType::F16];
    let test_shapes = test_shapes();

    eprintln!("Matmul perf: {} dtypes x {} shapes", data_types.len(), test_shapes.len(),);

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::with_template("{spinner} {pos} benchmarks [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::new();

    for &data_type in &data_types {
        for shape in &test_shapes {
            progress_bar.set_message(format!("{data_type:?} {shape}"));
            let result = bench::benchmark_single(&context, data_type, shape);
            results.push(result);
            progress_bar.inc(1);
        }
    }

    progress_bar.finish_with_message("done");
    print_results_table(&results);
    write_json_results("matmul_perf", &context.device.name(), &results);

    let error_results: Vec<_> = results.iter().filter(|r| r.status != "ok").collect();
    if !error_results.is_empty() {
        eprintln!("{} / {} cases had errors", error_results.len(), results.len());
    }
}
