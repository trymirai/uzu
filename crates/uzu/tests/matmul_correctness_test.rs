//! Correctness tests comparing Metal matmul kernels against MPSGraph

use std::collections::HashMap;

use half::bf16;
use metal::{Device, MTLResourceOptions};
use mpsgraph::{
    CommandBuffer, Device as MPSDevice, ExecutableExecutionDescriptor, Graph,
    ShapedType, TensorData,
};
use objc2::rc::autoreleasepool;
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::{MatmulArguments, MatmulKernel},
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

fn run_metal_matmul(
    ctx: &MTLContext,
    a_data: &[bf16],
    b_data: &[bf16],
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) -> Vec<bf16> {
    let a_buf = ctx.device.new_buffer_with_data(
        a_data.as_ptr() as *const _,
        (a_data.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b_data.as_ptr() as *const _,
        (b_data.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (m * n * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let ldb = if transpose_b {
        k
    } else {
        n
    };

    let mut kernel = MatmulKernel::new(ctx, DataType::BF16, false, transpose_b)
        .expect("kernel");

    let cb = ctx.command_queue.new_command_buffer().to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel
        .encode(
            ctx,
            &enc,
            MatmulArguments {
                a: &a_buf,
                a_offset: 0,
                b: &b_buf,
                c: None,
                d: &d_buf,
                batch: m as i32,
                input_dim: k as i32,
                output_dim: n as i32,
                lda: k as i32,
                ldb: ldb as i32,
                ldd: n as i32,
                batch_count: 1,
                alpha: 1.0,
                beta: 0.0,
            },
        )
        .expect("encode");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    unsafe {
        let ptr = d_buf.contents() as *const bf16;
        std::slice::from_raw_parts(ptr, m * n).to_vec()
    }
}

fn run_mps_matmul(
    device: &Device,
    a_data: &[bf16],
    b_data: &[bf16],
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) -> Vec<bf16> {
    autoreleasepool(|_| {
        let graph = Graph::new();

        let a_shape = [m as isize, k as isize];
        let b_shape_for_placeholder = if transpose_b {
            [n as isize, k as isize]
        } else {
            [k as isize, n as isize]
        };

        let a_placeholder = graph.placeholder(
            Some(&a_shape),
            mpsgraph::DataType::BFloat16,
            Some("A"),
        );
        let b_placeholder = graph.placeholder(
            Some(&b_shape_for_placeholder),
            mpsgraph::DataType::BFloat16,
            Some("B"),
        );

        let result = if transpose_b {
            let b_transposed =
                graph.transpose(&b_placeholder, &[1, 0], Some("B_T"));
            graph.matrix_multiplication(&a_placeholder, &b_transposed, None)
        } else {
            graph.matrix_multiplication(&a_placeholder, &b_placeholder, None)
        };

        let a_shaped_type = ShapedType::new_with_shape_data_type(
            Some(&a_shape),
            mpsgraph::DataType::BFloat16,
        );
        let b_shaped_type = ShapedType::new_with_shape_data_type(
            Some(&b_shape_for_placeholder),
            mpsgraph::DataType::BFloat16,
        );

        let feeds = HashMap::from([
            (&*a_placeholder, &*a_shaped_type),
            (&*b_placeholder, &*b_shaped_type),
        ]);

        let mps_device = MPSDevice::with_device(device);
        let executable =
            graph.compile(&mps_device, &feeds, &[&result], None, None);

        let a_buf = device.new_buffer_with_data(
            a_data.as_ptr() as *const _,
            (a_data.len() * core::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = device.new_buffer_with_data(
            b_data.as_ptr() as *const _,
            (b_data.len() * core::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let d_buf = device.new_buffer(
            (m * n * core::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let b_usize_shape: Vec<usize> = if transpose_b {
            vec![n, k]
        } else {
            vec![k, n]
        };

        let a_tensor = TensorData::new_with_mtl_buffer(
            &a_buf,
            &[m, k],
            mpsgraph::DataType::BFloat16,
            None,
        );
        let b_tensor = TensorData::new_with_mtl_buffer(
            &b_buf,
            &b_usize_shape,
            mpsgraph::DataType::BFloat16,
            None,
        );
        let d_tensor = TensorData::new_with_mtl_buffer(
            &d_buf,
            &[m, n],
            mpsgraph::DataType::BFloat16,
            None,
        );

        let inputs: &[&TensorData] = &[&a_tensor, &b_tensor];
        let outputs: &[&TensorData] = &[&d_tensor];

        let command_queue = device.new_command_queue();
        let exec_desc = ExecutableExecutionDescriptor::new();
        exec_desc.set_enable_commit_and_continue(true);

        let command_buffer = CommandBuffer::from_command_queue(&command_queue);
        let root_cb = command_buffer.root_command_buffer().to_owned();
        let _ = executable.encode_to_command_buffer(
            &command_buffer,
            inputs,
            Some(outputs),
            Some(&exec_desc),
        );
        root_cb.commit();
        root_cb.wait_until_completed();

        unsafe {
            let ptr = d_buf.contents() as *const bf16;
            std::slice::from_raw_parts(ptr, m * n).to_vec()
        }
    })
}

fn generate_test_data(
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) -> (Vec<bf16>, Vec<bf16>) {
    let a: Vec<bf16> = (0..(m * k))
        .map(|i| bf16::from_f32(((i % 13) as f32) * 0.01 - 0.06))
        .collect();
    let b_size = if transpose_b {
        n * k
    } else {
        k * n
    };
    let b: Vec<bf16> = (0..b_size)
        .map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.15))
        .collect();
    (a, b)
}

fn compare_results(
    metal: &[bf16],
    mps: &[bf16],
    tolerance: f32,
    test_name: &str,
) {
    assert_eq!(
        metal.len(),
        mps.len(),
        "{}: length mismatch {} vs {}",
        test_name,
        metal.len(),
        mps.len()
    );

    let mut max_diff: f32 = 0.0;
    let mut max_diff_idx = 0;
    let mut diff_count = 0;

    for (i, (&m_val, &p_val)) in metal.iter().zip(mps.iter()).enumerate() {
        let mf = m_val.to_f32();
        let pf = p_val.to_f32();
        let diff = (mf - pf).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        if diff > tolerance {
            diff_count += 1;
        }
    }

    assert!(
        max_diff <= tolerance,
        "{}: max diff {} at idx {} exceeds tolerance {} ({} elements differ)",
        test_name,
        max_diff,
        max_diff_idx,
        tolerance,
        diff_count
    );
}

#[test]
fn matmul_correctness_gemv_small() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (1, 2048, 2048);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.01, "GEMV small");
}

#[test]
fn matmul_correctness_gemv_medium() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (1, 4096, 4096);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.01, "GEMV medium");
}

#[test]
fn matmul_correctness_gemv_large() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (1, 8192, 8192);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.02, "GEMV large");
}

#[test]
fn matmul_correctness_small_batch() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (8, 2048, 2048);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.02, "Small batch");
}

#[test]
fn matmul_correctness_medium_batch() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (32, 2048, 2048);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.05, "Medium batch");
}

#[test]
fn matmul_correctness_large_batch() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (128, 2048, 2048);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.1, "Large batch");
}

#[test]
fn matmul_correctness_prefill_small() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (256, 2048, 2048);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.15, "Prefill small");
}

#[test]
fn matmul_correctness_prefill_medium() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (512, 2048, 2048);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.2, "Prefill medium");
}

#[test]
fn matmul_correctness_mlp_up() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (1, 2048, 8192);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.01, "MLP up");
}

#[test]
fn matmul_correctness_mlp_down() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (1, 8192, 2048);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.02, "MLP down");
}

#[test]
fn matmul_correctness_square_small() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (256, 256, 256);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.1, "Square small");
}

#[test]
fn matmul_correctness_square_medium() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (512, 512, 512);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.1, "Square medium");
}

#[test]
fn matmul_correctness_square_large() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (1024, 1024, 1024);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.1, "Square large");
}

#[test]
fn matmul_correctness_non_transposed() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (64, 128, 64);
    let (a, b) = generate_test_data(m, k, n, false);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, false);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, false);

    compare_results(&metal_result, &mps_result, 0.15, "Non-transposed");
}

#[test]
fn matmul_correctness_attention_qk() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (512, 64, 512);
    let (a, b) = generate_test_data(m, k, n, true);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, true);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, true);

    compare_results(&metal_result, &mps_result, 0.1, "Attention Q*K^T");
}

#[test]
fn matmul_correctness_attention_av() {
    let Some(ctx) = create_test_context() else {
        return;
    };
    let (m, k, n) = (512, 512, 64);
    let (a, b) = generate_test_data(m, k, n, false);

    let metal_result = run_metal_matmul(&ctx, &a, &b, m, k, n, false);
    let mps_result = run_mps_matmul(&ctx.device, &a, &b, m, k, n, false);

    compare_results(&metal_result, &mps_result, 0.1, "Attention A*V");
}
