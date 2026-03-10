#![cfg(target_os = "macos")]

#[path = "../common/mod.rs"]
mod common;
mod output;
mod reference;
mod verify;

use common::matmul::{DtypeCombo, TestShape, make_arguments, try_all_descriptors};
use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use output::print_results_table;
use uzu::{
    DataType,
    backends::{
        common::{
            Context,
            CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferInitial, CommandBufferPending,
            kernel::matmul::{MatmulDispatchDescriptor, MatmulKernel, gemm_mpp},
        },
        metal::{Metal, MetalContext},
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

/// Diagnostic: probe the cooperative tensor layout vs BaseMppFrag layout.
/// Dispatches a matmul with K=-999 which triggers probe mode in the shader.
/// The shader writes CT and BaseMppFrag coordinates into the output buffer.
#[test]
#[ignore]
fn mpp_layout_probe() {
    let context = MetalContext::new().expect("Metal context required");

    // Use a shape where M >= 16, N >= 32 (SM=16, SN=32 subtile size)
    // and K doesn't matter (probe mode ignores it)
    let shape = TestShape { batch: 128, input_dim: 128, output_dim: 128 };
    let combo = DtypeCombo {
        a_dtype: DataType::F16,
        b_dtype: DataType::F16,
        output_dtype: DataType::F16,
    };

    let a_byte_count = shape.batch * shape.input_dim * combo.a_dtype.size_in_bytes();
    let b_byte_count = shape.output_dim * shape.input_dim * combo.b_dtype.size_in_bytes();
    // Output buffer needs to be large enough for probe data (ints, not halfs)
    // Probe writes at most: 3 + 32 * ((16+16+16)*2 + 8*2) = 3 + 32 * 112 = 3587 ints = ~14KB
    let d_byte_count = std::cmp::max(
        shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes(),
        16 * 1024, // ensure enough space for probe data
    );

    let (a_buffer, b_buffer, mut d_buffer) = match (
        context.device.new_buffer(a_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
        context.device.new_buffer(b_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
        context.device.new_buffer(d_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
    ) {
        (Some(a), Some(b), Some(d)) => (a, b, d),
        _ => panic!("Failed to allocate buffers"),
    };

    // Zero the output buffer
    unsafe {
        let ptr = d_buffer.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(ptr, 0, d_byte_count);
    }

    let arguments = make_arguments(&a_buffer, &b_buffer, &mut d_buffer, &shape);
    let mut descriptor = gemm_mpp::DispatchDescriptor::new(combo.output_dtype, &arguments)
        .expect("MPP dispatch descriptor");

    // Set K = -999 to trigger probe mode
    descriptor.params.K = -999;

    let dispatch_descriptor = MatmulDispatchDescriptor::GemmMpp(descriptor);

    // Dispatch the kernel
    let mut kernel = MatmulKernel::<Metal>::new_mixed(combo.a_dtype, combo.b_dtype, combo.output_dtype)
        .expect("create kernel");
    let arguments = make_arguments(&a_buffer, &b_buffer, &mut d_buffer, &shape);
    let mut command_buffer = context
        .create_command_buffer()
        .expect("create command buffer")
        .start_encoding();

    kernel
        .encode_with_descriptor(&context, arguments, &dispatch_descriptor, &mut command_buffer)
        .expect("encode");

    command_buffer
        .end_encoding()
        .submit()
        .wait_until_completed()
        .expect("wait");

    // Read probe data from output buffer
    let probe_data: &[i32] = unsafe {
        let ptr = d_buffer.contents().as_ptr() as *const i32;
        std::slice::from_raw_parts(ptr, d_byte_count / 4)
    };

    let a_cap = probe_data[0];
    let b_cap = probe_data[1];
    let c_cap = probe_data[2];
    eprintln!("CT capacities: A={}, B={}, C={}", a_cap, b_cap, c_cap);

    let per_lane = ((a_cap + b_cap + c_cap) * 2 + 8 * 2) as usize;

    // Print mapping for lane 0
    for lane in 0..1 {
        let base = 3 + lane * per_lane;
        let mut off = base;

        eprintln!("\n=== Lane {} ===", lane);
        eprintln!("\nCT_A coordinates (cap={}):", a_cap);
        for i in 0..a_cap as usize {
            let col = probe_data[off];
            let row = probe_data[off + 1];
            off += 2;
            eprintln!("  ct_a[{}] -> (col={}, row={})", i, col, row);
        }

        eprintln!("\nCT_B coordinates (cap={}):", b_cap);
        for i in 0..b_cap as usize {
            let col = probe_data[off];
            let row = probe_data[off + 1];
            off += 2;
            eprintln!("  ct_b[{}] -> (col={}, row={})", i, col, row);
        }

        eprintln!("\nCT_C coordinates (cap={}):", c_cap);
        for i in 0..c_cap as usize {
            let col = probe_data[off];
            let row = probe_data[off + 1];
            off += 2;
            eprintln!("  ct_c[{}] -> (col={}, row={})", i, col, row);
        }

        eprintln!("\nBaseMppFrag coordinates (8 elements):");
        for j in 0..8 {
            let col = probe_data[off];
            let row = probe_data[off + 1];
            off += 2;
            eprintln!("  frag[{}] -> (col={}, row={})", j, col, row);
        }

        // Compute permutation: for each CT index i, find frag index j with matching coords
        eprintln!("\n--- Permutation A ---");
        let a_base = 3 + lane * per_lane;
        let frag_base = a_base + ((a_cap + b_cap + c_cap) * 2) as usize;
        for i in 0..a_cap as usize {
            let ct_col = probe_data[a_base + i * 2];
            let ct_row = probe_data[a_base + i * 2 + 1];
            let mut match_j: i32 = -1;
            for j in 0..8 {
                let f_col = probe_data[frag_base + j * 2];
                let f_row = probe_data[frag_base + j * 2 + 1];
                if ct_col == f_col && ct_row == f_row {
                    match_j = j as i32;
                    break;
                }
            }
            eprintln!("  ct_a[{}] (col={}, row={}) -> frag[{}]", i, ct_col, ct_row, match_j);
        }

        eprintln!("\n--- Permutation C ---");
        let c_base = a_base + ((a_cap + b_cap) * 2) as usize;
        for i in 0..c_cap as usize {
            let ct_col = probe_data[c_base + i * 2];
            let ct_row = probe_data[c_base + i * 2 + 1];
            // C is 16x32, spans two 16x16 fragments (cols 0-15 and 16-31)
            let mut match_j: i32 = -1;
            for frag_idx in 0..2 {
                let col_offset = frag_idx * 16;
                for j in 0..8 {
                    let f_col = probe_data[frag_base + j * 2] + col_offset;
                    let f_row = probe_data[frag_base + j * 2 + 1];
                    if ct_col == f_col && ct_row == f_row {
                        match_j = (frag_idx * 8 + j as i32) as i32;
                        break;
                    }
                }
                if match_j >= 0 { break; }
            }
            eprintln!("  ct_c[{}] (col={}, row={}) -> frag_elem[{}]", i, ct_col, ct_row, match_j);
        }
    }

    // Check if permutation is consistent across all 32 lanes
    eprintln!("\n=== Checking permutation consistency across lanes ===");
    let mut consistent = true;
    let lane0_base = 3usize;
    for lane in 1..32usize {
        let lane_base = 3 + lane * per_lane;
        let frag_base_0 = lane0_base + ((a_cap + b_cap + c_cap) * 2) as usize;
        let frag_base_l = lane_base + ((a_cap + b_cap + c_cap) * 2) as usize;

        for i in 0..a_cap as usize {
            let ct_col_0 = probe_data[lane0_base + i * 2];
            let ct_row_0 = probe_data[lane0_base + i * 2 + 1];
            let ct_col_l = probe_data[lane_base + i * 2];
            let ct_row_l = probe_data[lane_base + i * 2 + 1];

            // Find perm for lane 0
            let mut perm0: i32 = -1;
            let mut perml: i32 = -1;
            for j in 0..8 {
                if probe_data[frag_base_0 + j * 2] == ct_col_0 && probe_data[frag_base_0 + j * 2 + 1] == ct_row_0 {
                    perm0 = j as i32;
                }
                if probe_data[frag_base_l + j * 2] == ct_col_l && probe_data[frag_base_l + j * 2 + 1] == ct_row_l {
                    perml = j as i32;
                }
            }
            if perm0 != perml {
                eprintln!("  MISMATCH: ct_a[{}] lane0->frag[{}] vs lane{}->frag[{}]", i, perm0, lane, perml);
                consistent = false;
            }
        }
    }
    if consistent {
        eprintln!("  Permutation is CONSISTENT across all 32 lanes!");
    } else {
        eprintln!("  Permutation VARIES across lanes.");
    }
}
