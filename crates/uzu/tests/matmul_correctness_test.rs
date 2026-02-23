#![cfg(target_os = "macos")]

use bytemuck;
use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use half::bf16;
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use ndarray::Array2;
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context,
            kernel::matmul::{MatmulArguments, MatmulKernel},
        },
        metal::{Metal, MetalContext, choose_dispatch_descriptor},
    },
};

#[derive(Clone)]
struct TestShape {
    batch: usize,
    input_dim: usize,
    output_dim: usize,
}

impl std::fmt::Display for TestShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.batch, self.input_dim, self.output_dim)
    }
}

#[derive(Clone)]
struct DtypeCombo {
    a_dtype: DataType,
    b_dtype: DataType,
    output_dtype: DataType,
}

impl std::fmt::Display for DtypeCombo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}*{:?}->{:?}", self.a_dtype, self.b_dtype, self.output_dtype)
    }
}

struct TestResult {
    combo: DtypeCombo,
    shape: TestShape,
    passed: bool,
    max_diff: f64,
    tolerance: f64,
}

fn test_shapes() -> Vec<TestShape> {
    let mut shapes = Vec::new();

    for &batch in &[512, 1024, 2048] {
        for &output_dim in &[512, 1024, 2048] {
            for &input_dim in &[1, 2, 4, 8, 16, 32, 64] {
                shapes.push(TestShape { batch, input_dim, output_dim });
            }
        }
    }

    let model_dims: &[(usize, usize)] = &[
        (896, 896), (896, 4864), (4864, 896),
        (1024, 1024), (1024, 4096), (4096, 1024),
        (1152, 1152), (1152, 6912), (6912, 1152),
        (1536, 1536), (1536, 8960), (8960, 1536),
        (2048, 2048), (2048, 8192), (8192, 2048),
        (2560, 2560), (2560, 10240), (10240, 2560),
        (3072, 3072), (3072, 8192), (8192, 3072),
        (3584, 3584), (3584, 18944), (18944, 3584),
        (4096, 4096), (4096, 14336), (14336, 4096),
        (5120, 5120), (5120, 17408), (17408, 5120),
    ];
    for &(input_dim, output_dim) in model_dims {
        for &batch in &[1, 128] {
            shapes.push(TestShape { batch, input_dim, output_dim });
        }
    }

    shapes
}

fn tolerance_for(combo: &DtypeCombo, shape: &TestShape) -> f64 {
    match (combo.a_dtype, combo.b_dtype, combo.output_dtype) {
        (DataType::I8, DataType::I8, DataType::I32) => 0.0,
        (DataType::I8, DataType::BF16, DataType::BF16) => {
            0.05 * (shape.input_dim as f64 / 1024.0).sqrt()
        },
        (DataType::BF16, DataType::BF16, DataType::BF16) => {
            let base = 0.01 * (shape.input_dim as f64 / 1024.0).sqrt();
            base * (1.0 + (shape.batch as f64).ln() / std::f64::consts::LN_2 * 0.02)
        },
        _ => 0.01,
    }
}

fn generate_a_data(combo: &DtypeCombo, count: usize) -> Vec<u8> {
    match combo.a_dtype {
        DataType::I8 => {
            let data: Vec<i8> = (0..count).map(|i| ((i % 13) as i8) - 6).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        DataType::BF16 => {
            let data: Vec<bf16> = (0..count).map(|i| bf16::from_f32(((i % 13) as f32) * 0.01 - 0.06)).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        _ => panic!("Unsupported a_dtype: {:?}", combo.a_dtype),
    }
}

fn generate_b_data(combo: &DtypeCombo, count: usize) -> Vec<u8> {
    match combo.b_dtype {
        DataType::I8 => {
            let data: Vec<i8> = (0..count).map(|i| ((i % 17) as i8) - 8).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        DataType::BF16 => {
            let data: Vec<bf16> = (0..count).map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.15)).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        _ => panic!("Unsupported b_dtype: {:?}", combo.b_dtype),
    }
}

fn a_bytes_to_f64(combo: &DtypeCombo, bytes: &[u8]) -> Vec<f64> {
    match combo.a_dtype {
        DataType::I8 => bytemuck::cast_slice::<u8, i8>(bytes).iter().map(|&x| x as f64).collect(),
        DataType::BF16 => bytemuck::cast_slice::<u8, bf16>(bytes).iter().map(|x| x.to_f64()).collect(),
        _ => panic!("Unsupported a_dtype: {:?}", combo.a_dtype),
    }
}

fn b_bytes_to_f64(combo: &DtypeCombo, bytes: &[u8]) -> Vec<f64> {
    match combo.b_dtype {
        DataType::I8 => bytemuck::cast_slice::<u8, i8>(bytes).iter().map(|&x| x as f64).collect(),
        DataType::BF16 => bytemuck::cast_slice::<u8, bf16>(bytes).iter().map(|x| x.to_f64()).collect(),
        _ => panic!("Unsupported b_dtype: {:?}", combo.b_dtype),
    }
}

fn output_to_f64(combo: &DtypeCombo, buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f64> {
    unsafe {
        let ptr = buf.contents().as_ptr();
        match combo.output_dtype {
            DataType::I32 => {
                let slice = std::slice::from_raw_parts(ptr as *const i32, count);
                slice.iter().map(|&x| x as f64).collect()
            },
            DataType::BF16 => {
                let slice = std::slice::from_raw_parts(ptr as *const bf16, count);
                slice.iter().map(|x| x.to_f64()).collect()
            },
            _ => panic!("Unsupported output_dtype: {:?}", combo.output_dtype),
        }
    }
}

fn ndarray_reference(combo: &DtypeCombo, a_bytes: &[u8], b_bytes: &[u8], shape: &TestShape) -> Vec<f64> {
    let a_f64 = a_bytes_to_f64(combo, a_bytes);
    let b_f64 = b_bytes_to_f64(combo, b_bytes);

    let a_arr = Array2::from_shape_vec((shape.batch, shape.input_dim), a_f64).expect("A shape");
    let b_arr = Array2::from_shape_vec((shape.output_dim, shape.input_dim), b_f64).expect("B shape");
    let result = a_arr.dot(&b_arr.t());

    match combo.output_dtype {
        DataType::I32 => result.iter().map(|&x| (x as i32) as f64).collect(),
        DataType::BF16 => result.iter().map(|&x| bf16::from_f64(x).to_f64()).collect(),
        _ => result.iter().copied().collect(),
    }
}

fn run_metal_matmul(
    ctx: &MetalContext,
    combo: &DtypeCombo,
    a_bytes: &[u8],
    b_bytes: &[u8],
    shape: &TestShape,
) -> Vec<f64> {
    let a_buf = ctx
        .device
        .new_buffer_with_data(a_bytes, MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create A buffer");
    let b_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(b_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let mut d_buf = ctx
        .device
        .new_buffer(
            shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create D buffer");

    let mut kernel =
        MatmulKernel::<Metal>::new_mixed(combo.a_dtype, combo.b_dtype, combo.output_dtype).expect("kernel creation");

    let mut kernel = MatmulKernel::<Metal>::new(DataType::BF16).expect("kernel");

    let mut command_buffer = ctx.create_command_buffer().unwrap().start_encoding();
    let mut arguments = MatmulArguments {
        a: &a_buf,
        a_offset: 0,
        b: &b_buf,
        d: &mut d_buf,
        bias: None,
        batch: shape.batch as i32,
        input_dim: shape.input_dim as i32,
        output_dim: shape.output_dim as i32,
        lda: shape.input_dim as i32,
        ldb: shape.input_dim as i32,
        ldd: shape.output_dim as i32,
        batch_count: 1,
        transpose_b: true,
    };
    MatmulKernel::<Metal>::apply_batch_collapse(&mut arguments);
    let descriptor = choose_dispatch_descriptor(ctx, DataType::BF16, &arguments).expect("dispatch descriptor");
    kernel.encode_with_descriptor(ctx, arguments, &descriptor, &mut command_buffer).expect("encode");
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    output_to_f64(combo, &d_buf, shape.batch * shape.output_dim)
}

fn run_correctness_case(ctx: &MetalContext, combo: &DtypeCombo, shape: &TestShape) -> TestResult {
    let a_bytes = generate_a_data(combo, shape.batch * shape.input_dim);
    let b_bytes = generate_b_data(combo, shape.output_dim * shape.input_dim);

    let metal_result = run_metal_matmul(ctx, combo, &a_bytes, &b_bytes, shape);
    let reference = ndarray_reference(combo, &a_bytes, &b_bytes, shape);

    let tolerance = tolerance_for(combo, shape);
    let max_diff = metal_result
        .iter()
        .zip(reference.iter())
        .map(|(&m_val, &r_val)| (m_val - r_val).abs())
        .fold(0.0f64, f64::max);

    TestResult {
        combo: combo.clone(),
        shape: shape.clone(),
        passed: max_diff <= tolerance,
        max_diff,
        tolerance,
    }
}

fn print_results_table(results: &[TestResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Dtype Combo", "Shape (BxKxN)", "Status", "Max Diff", "Tolerance"]);

    for r in results {
        table.add_row(vec![
            format!("{}", r.combo),
            format!("{}", r.shape),
            if r.passed { "PASS".into() } else { "FAIL".into() },
            format!("{:.6}", r.max_diff),
            format!("{:.6}", r.tolerance),
        ]);
    }

    for col in [3, 4] {
        if let Some(column) = table.column_mut(col) {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }

    println!("{table}");
}

#[test]
#[ignore]
fn matmul_correctness() {
    let ctx = MetalContext::new().expect("Metal context required");

    let mut combos = vec![DtypeCombo {
        a_dtype: DataType::BF16,
        b_dtype: DataType::BF16,
        output_dtype: DataType::BF16,
    }];

    let mpp_combos = [
        DtypeCombo { a_dtype: DataType::I8, b_dtype: DataType::I8, output_dtype: DataType::I32 },
        DtypeCombo { a_dtype: DataType::I8, b_dtype: DataType::BF16, output_dtype: DataType::BF16 },
    ];
    if ctx.is_mpp_available() {
        combos.extend(mpp_combos);
    } else {
        eprintln!("MPP not available, skipping I8 combos");
    }

    let shapes = test_shapes();
    let total = combos.len() * shapes.len();

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} [{elapsed_precise}] {msg}")
            .expect("progress style"),
    );

    let mut results = Vec::with_capacity(total);

    for combo in &combos {
        for shape in &shapes {
            pb.set_message(format!("{} {}", combo, shape));
            let result = run_correctness_case(&ctx, combo, shape);
            results.push(result);
            pb.inc(1);
        }
    }

    pb.finish_with_message("done");
    print_results_table(&results);

    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failures.is_empty() {
        eprintln!("\n{} / {} cases failed:", failures.len(), results.len());
        for f in &failures {
            eprintln!("  {} {} max_diff={:.6} > tol={:.6}", f.combo, f.shape, f.max_diff, f.tolerance);
        }
        panic!("{} matmul correctness cases failed", failures.len());
    }
}
