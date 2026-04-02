#![cfg(metal_backend)]

use bench::DispatchPath;
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
    context: &Ctx,
    results: &[T],
) {
    if let Ok(dir) = std::env::var("UZU_TEST_RESULTS_DIR") {
        let caps = context.device_capabilities();
        let path = std::path::Path::new(&dir);
        std::fs::create_dir_all(path).expect("create results dir");
        let file = path.join(format!("{test_name}.json"));
        let wrapper = serde_json::json!({
            "device": context.device.name(),
            "generation": caps.generation.generation_number(),
            "gpu_core_count": caps.gpu_core_count,
            "max_threadgroup_memory_bytes": caps.max_threadgroup_memory.as_u64(),
            "shared_memory_bytes": caps.shared_memory_size.as_u64(),
            "supports_mxu": caps.supports_mxu,
            "supports_simd_group_matrix": caps.supports_simd_group_matrix,
            "results": results,
        });
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
    let dispatch_paths = DispatchPath::available_paths();

    let caps = context.device_capabilities();
    eprintln!(
        "Device: {} (gen{}, {} cores, {}, mxu={})",
        context.device.name(),
        caps.generation.generation_number(),
        caps.gpu_core_count,
        caps.shared_memory_size,
        caps.supports_mxu,
    );

    let total_benchmarks = data_types.len() * test_shapes.len() * dispatch_paths.len();
    eprintln!(
        "Matmul perf: {} dtypes x {} shapes x {} kernels = {} benchmarks",
        data_types.len(),
        test_shapes.len(),
        dispatch_paths.len(),
        total_benchmarks,
    );

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::with_template("{spinner} {pos}/{len} benchmarks [{elapsed_precise}] {msg}")
            .expect("progress style"),
    );
    progress_bar.set_length(total_benchmarks as u64);

    let mut results = Vec::new();

    for &data_type in &data_types {
        for shape in &test_shapes {
            for &dispatch_path in &dispatch_paths {
                progress_bar.set_message(format!("{data_type:?} {shape} {}", dispatch_path.name()));
                let result = bench::benchmark_single(&context, data_type, dispatch_path, shape);
                results.push(result);
                progress_bar.inc(1);
            }
        }
    }

    progress_bar.finish_with_message("done");
    print_results_table(&results);
    write_json_results("matmul_perf", &context, &results);

    let error_results: Vec<_> = results.iter().filter(|r| r.status != "ok").collect();
    if !error_results.is_empty() {
        eprintln!("{} / {} cases had errors", error_results.len(), results.len());
    }
}
