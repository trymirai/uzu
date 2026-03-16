#![cfg(all(target_os = "macos", feature = "metal"))]

mod bench;
mod error;
mod output;
mod shapes;

use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLDeviceExt, MTLResourceOptions};
use output::print_results_table;
use shapes::test_shapes;
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, Context,
            kernel::matmul::{MatmulDispatchDescriptor, choose_matmul_dispatch_descriptor},
        },
        metal::Metal,
    },
};

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
        let elem_size = data_type.size_in_bytes();

        for shape in &test_shapes {
            let a_byte_count = shape.batch * shape.input_dim * elem_size;
            let b_byte_count = shape.output_dim * shape.input_dim * elem_size;
            let d_byte_count = shape.batch * shape.output_dim * elem_size;

            let (a_buffer, b_buffer, mut d_buffer) = match (
                context.device.new_buffer(a_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
                context.device.new_buffer(b_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
                context.device.new_buffer(d_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
            ) {
                (Some(a), Some(b), Some(d)) => (a, b, d),
                _ => continue,
            };

            let arguments = bench::make_arguments(&a_buffer, &b_buffer, &mut d_buffer, shape);

            let descriptor = match choose_matmul_dispatch_descriptor::<Metal>(&context, data_type, &arguments) {
                Ok(d) => d,
                Err(e) => {
                    results.push(output::PerfResult {
                        combo: format!("{data_type:?}"),
                        shape: format!("{shape}"),
                        dispatch_path: "auto".into(),
                        duration_ms: 0.0,
                        gflops: 0.0,
                        status: "error".into(),
                        error: Some(format!("dispatch: {e}")),
                    });
                    continue;
                },
            };

            let path_name = match &descriptor {
                MatmulDispatchDescriptor::Gemv(_) => "Gemv",
                MatmulDispatchDescriptor::Gemm(_) => "Gemm",
            };

            progress_bar.set_message(format!("{data_type:?} {shape} {path_name}"));
            let result = bench::benchmark_single(&context, data_type, shape, path_name, &descriptor);
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
