#![cfg(target_os = "macos")]

#[path = "../common/mod.rs"]
mod common;
mod bench;
mod error;
mod output;
mod shapes;

use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLDeviceExt, MTLResourceOptions};
use uzu::backends::{common::Context, metal::MetalContext};

use common::matmul::{make_arguments, test_combos, try_all_descriptors, DtypeCombo, TestShape};
use output::print_results_table;
use shapes::test_shapes;
use uzu::{
    DataType,
    backends::common::kernel::matmul::{MatmulDispatchDescriptor, gemm_mpp},
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
            let a_byte_count = shape.batch * shape.input_dim * combo.a_dtype.size_in_bytes();
            let b_byte_count = shape.output_dim * shape.input_dim * combo.b_dtype.size_in_bytes();
            let d_byte_count = shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes();

            let (a_buffer, b_buffer, mut d_buffer) = match (
                context.device.new_buffer(a_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
                context.device.new_buffer(b_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
                context.device.new_buffer(d_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
            ) {
                (Some(a), Some(b), Some(d)) => (a, b, d),
                _ => continue,
            };

            let arguments = make_arguments(&a_buffer, &b_buffer, &mut d_buffer, shape);
            let dispatch_descriptors = try_all_descriptors(&context, combo, &arguments);

            for (path_name, dispatch_descriptor) in &dispatch_descriptors {
                progress_bar.set_message(format!("{} {} {}", combo, shape, path_name));
                let result = bench::benchmark_single(&context, combo, shape, path_name, dispatch_descriptor);
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

fn mpp_perf_shapes() -> Vec<TestShape> {
    vec![
        TestShape { batch: 128, input_dim: 128, output_dim: 128 },
        TestShape { batch: 256, input_dim: 256, output_dim: 256 },
        TestShape { batch: 128, input_dim: 896, output_dim: 896 },
        TestShape { batch: 64, input_dim: 896, output_dim: 4864 },
        TestShape { batch: 1, input_dim: 4096, output_dim: 4096 },
        TestShape { batch: 64, input_dim: 2048, output_dim: 8192 },
        TestShape { batch: 100, input_dim: 128, output_dim: 200 },
        TestShape { batch: 128, input_dim: 17, output_dim: 128 },
    ]
}

#[test]
#[ignore]
fn matmul_mpp_perf() {
    let context = MetalContext::new().expect("Metal context required");

    let combos = vec![
        DtypeCombo { a_dtype: DataType::F16, b_dtype: DataType::F16, output_dtype: DataType::F16 },
        DtypeCombo { a_dtype: DataType::I8, b_dtype: DataType::F16, output_dtype: DataType::F16 },
    ];
    let shapes = mpp_perf_shapes();

    eprintln!("MPP matmul perf: {} combos x {} shapes", combos.len(), shapes.len());

    let mut results = Vec::new();

    for combo in &combos {
        for shape in &shapes {
            let a_byte_count = shape.batch * shape.input_dim * combo.a_dtype.size_in_bytes();
            let b_byte_count = shape.output_dim * shape.input_dim * combo.b_dtype.size_in_bytes();
            let d_byte_count = shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes();

            let (a_buffer, b_buffer, mut d_buffer) = match (
                context.device.new_buffer(a_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
                context.device.new_buffer(b_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
                context.device.new_buffer(d_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
            ) {
                (Some(a), Some(b), Some(d)) => (a, b, d),
                _ => continue,
            };

            let arguments = make_arguments(&a_buffer, &b_buffer, &mut d_buffer, shape);
            if let Ok(descriptor) = gemm_mpp::DispatchDescriptor::new(combo.output_dtype, &arguments) {
                let dispatch_descriptor = MatmulDispatchDescriptor::GemmMpp(descriptor);
                eprintln!("Benchmarking: {} {} GemmMpp", combo, shape);
                let result = bench::benchmark_single(&context, combo, shape, "GemmMpp", &dispatch_descriptor);
                eprintln!("  {:.1} GFLOPS  {:.3} ms", result.gflops, result.duration_ms);
                results.push(result);
            }
        }
    }

    print_results_table(&results);
    common::matmul::write_json_results("matmul_mpp_perf", &context.device.name(), &results);
}
