#![cfg(target_os = "macos")]

mod bench;
mod combos;
mod descriptors;
mod error;
mod output;
mod shapes;

use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLDeviceExt, MTLResourceOptions};
use uzu::backends::{common::Context, metal::MetalContext};

use bench::{benchmark_single, make_arguments};
use combos::test_combos;
use descriptors::try_all_descriptors;
use output::{print_results_table, write_json_results};
use shapes::test_shapes;

#[test]
#[ignore]
fn matmul_perf() {
    let context = MetalContext::new().expect("Metal context required");

    eprintln!("MPP available: {}", context.is_mpp_available());

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
                let result = benchmark_single(&context, combo, shape, path_name, dispatch_descriptor);
                results.push(result);
                progress_bar.inc(1);
            }
        }
    }

    progress_bar.finish_with_message("done");
    print_results_table(&results);
    write_json_results("matmul_perf", &context.device.name(), context.is_mpp_available(), &results);

    let error_results: Vec<_> = results.iter().filter(|r| r.status != "ok").collect();
    if !error_results.is_empty() {
        eprintln!("{} / {} cases had errors", error_results.len(), results.len());
    }
}
