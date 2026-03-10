#![cfg(target_os = "macos")]

#[path = "../common/mod.rs"]
mod common;
mod output;
mod reference;
mod verify;

use common::matmul::{DtypeCombo, TestShape, make_arguments, try_all_descriptors};
use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLDeviceExt, MTLResourceOptions};
use output::print_results_table;
use uzu::{
    DataType,
    backends::{
        common::{Context, kernel::matmul::{MatmulDispatchDescriptor, gemm_mpp}},
        metal::MetalContext,
    },
};
use verify::run_correctness_case;

const MODEL_DIMS: &[(usize, usize)] = &[
    (896, 896),
    (896, 4864),
    (4864, 896),
    (1024, 1024),
    (1024, 4096),
    (4096, 1024),
    (1152, 1152),
    (1152, 6912),
    (6912, 1152),
    (1536, 1536),
    (1536, 8960),
    (8960, 1536),
    (2048, 2048),
    (2048, 8192),
    (8192, 2048),
    (2560, 2560),
    (2560, 10240),
    (10240, 2560),
    (3072, 3072),
    (3072, 8192),
    (8192, 3072),
    (3584, 3584),
    (3584, 18944),
    (18944, 3584),
    (4096, 4096),
    (4096, 14336),
    (14336, 4096),
    (5120, 5120),
    (5120, 17408),
    (17408, 5120),
];

fn test_shapes() -> Vec<TestShape> {
    let small_dim_shapes = [512usize, 1024, 2048].iter().flat_map(|&batch| {
        [512, 1024, 2048].iter().flat_map(move |&output_dim| {
            [1, 2, 4, 8, 16, 32, 64].iter().map(move |&input_dim| TestShape {
                batch,
                input_dim,
                output_dim,
            })
        })
    });

    let model_shapes = MODEL_DIMS.iter().flat_map(|&(input_dim, output_dim)| {
        [1, 128].iter().map(move |&batch| TestShape {
            batch,
            input_dim,
            output_dim,
        })
    });

    small_dim_shapes.chain(model_shapes).collect()
}

fn mpp_test_shapes() -> Vec<TestShape> {
    vec![
        // Small aligned
        TestShape { batch: 64, input_dim: 64, output_dim: 64 },
        TestShape { batch: 128, input_dim: 128, output_dim: 128 },
        TestShape { batch: 256, input_dim: 256, output_dim: 256 },
        // Unaligned K
        TestShape { batch: 128, input_dim: 1, output_dim: 128 },
        TestShape { batch: 128, input_dim: 17, output_dim: 128 },
        // Unaligned M/N
        TestShape { batch: 100, input_dim: 128, output_dim: 200 },
        // Model-like
        TestShape { batch: 1, input_dim: 896, output_dim: 896 },
        TestShape { batch: 128, input_dim: 896, output_dim: 896 },
    ]
}

#[test]
#[ignore]
fn matmul_correctness() {
    let context = MetalContext::new().expect("Metal context required");

    let combos = vec![common::matmul::DtypeCombo {
        a_dtype: uzu::DataType::F16,
        b_dtype: uzu::DataType::F16,
        output_dtype: uzu::DataType::F16,
    }];
    let shapes = test_shapes();

    eprintln!(
        "Matmul correctness: {} combos x {} shapes, testing all applicable dispatch paths",
        combos.len(),
        shapes.len(),
    );

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::with_template("{spinner} {pos} tests [{elapsed_precise}] {msg}").expect("progress style"),
    );

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
            let dispatch_descriptors = try_all_descriptors(&context, combo, &arguments);

            for (path_name, dispatch_descriptor) in &dispatch_descriptors {
                progress_bar.set_message(format!("{} {} {}", combo, shape, path_name));
                let result = run_correctness_case(&context, combo, shape, path_name, dispatch_descriptor);
                results.push(result);
                progress_bar.inc(1);
            }
        }
    }

    progress_bar.finish_with_message("done");
    print_results_table(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failures.is_empty() {
        eprintln!("\n{} / {} cases failed:", failures.len(), results.len());
        for failure in &failures {
            eprintln!(
                "  {} {} [{}] max_diff={:.6} > tol={:.6}",
                failure.combo, failure.shape, failure.dispatch_path, failure.max_diff, failure.tolerance
            );
        }
        panic!("{} matmul correctness cases failed", failures.len());
    }
}

#[test]
#[ignore]
fn matmul_mpp_correctness() {
    let context = MetalContext::new().expect("Metal context required");

    let combo = DtypeCombo {
        a_dtype: DataType::F16,
        b_dtype: DataType::F16,
        output_dtype: DataType::F16,
    };
    let shapes = mpp_test_shapes();

    eprintln!("MPP matmul correctness: {} shapes", shapes.len());

    let mut results = Vec::new();

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
            eprintln!("Testing: {} {}", combo, shape);
            let result = run_correctness_case(&context, &combo, shape, "GemmMpp", &dispatch_descriptor);
            eprintln!(
                "  {} max_diff={:.6} tol={:.6}",
                if result.passed { "PASS" } else { "FAIL" },
                result.max_diff,
                result.tolerance,
            );
            results.push(result);
        }
    }

    print_results_table(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failures.is_empty() {
        eprintln!("\n{} / {} cases failed:", failures.len(), results.len());
        for failure in &failures {
            eprintln!(
                "  {} {} [{}] max_diff={:.6} > tol={:.6}",
                failure.combo, failure.shape, failure.dispatch_path, failure.max_diff, failure.tolerance
            );
        }
        panic!("{} MPP matmul correctness cases failed", failures.len());
    }
}
