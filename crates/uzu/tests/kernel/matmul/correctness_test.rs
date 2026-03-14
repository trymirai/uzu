use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use half::{bf16, f16};
use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use ndarray::Array2;
use objc2::runtime::ProtocolObject;
use serde::Serialize;
use uzu::{
    DataType,
    backends::{
        common::{
            CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending, Context,
            kernel::matmul::MatmulKernel,
        },
        metal::{MetalContext, kernel::matmul::MatmulMetalKernel},
    },
};

use crate::common::matmul::{DtypeCombo, MatmulVariant, TestShape, applicable_variants, make_full_precision_arguments};

const MODEL_DIMS: &[(usize, usize)] = &[
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

fn test_shapes() -> Vec<TestShape> {
    let small_dim_shapes = [512usize, 1024, 2048].iter().flat_map(|&batch| {
        [512, 1024, 2048].iter().flat_map(move |&output_dim| {
            [1, 2, 4, 8, 16, 32, 64].iter().map(move |&input_dim| TestShape {
                batch, input_dim, output_dim,
            })
        })
    });

    let model_shapes = MODEL_DIMS.iter().flat_map(|&(input_dim, output_dim)| {
        [1, 128].iter().map(move |&batch| TestShape {
            batch, input_dim, output_dim,
        })
    });

    small_dim_shapes.chain(model_shapes).collect()
}

// -- Reference computation ----------------------------------------------------

fn generate_typed_data(dtype: DataType, count: usize, modulus: usize, offset: i64) -> Vec<u8> {
    match dtype {
        DataType::I8 => {
            let data: Vec<i8> = (0..count).map(|i| ((i % modulus) as i8).wrapping_add(offset as i8)).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        DataType::BF16 => {
            let data: Vec<bf16> =
                (0..count).map(|i| bf16::from_f32(((i % modulus) as f32) * 0.01 + offset as f32 * 0.01)).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        DataType::F16 => {
            let data: Vec<f16> =
                (0..count).map(|i| f16::from_f32(((i % modulus) as f32) * 0.01 + offset as f32 * 0.01)).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        DataType::F32 => {
            let data: Vec<f32> = (0..count).map(|i| ((i % modulus) as f32) * 0.01 + offset as f32 * 0.01).collect();
            bytemuck::cast_slice(&data).to_vec()
        },
        other => panic!("Unsupported dtype for data generation: {other:?}"),
    }
}

fn bytes_to_f64(dtype: DataType, bytes: &[u8]) -> Vec<f64> {
    match dtype {
        DataType::I8 => bytemuck::cast_slice::<u8, i8>(bytes).iter().map(|&x| x as f64).collect(),
        DataType::BF16 => bytemuck::cast_slice::<u8, bf16>(bytes).iter().map(|x| x.to_f64()).collect(),
        DataType::F16 => bytemuck::cast_slice::<u8, f16>(bytes).iter().map(|x| x.to_f64()).collect(),
        DataType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).iter().map(|&x| x as f64).collect(),
        other => panic!("Unsupported dtype for conversion: {other:?}"),
    }
}

fn output_to_f64(output_dtype: DataType, buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f64> {
    unsafe {
        let pointer = buffer.contents().as_ptr();
        match output_dtype {
            DataType::I32 => std::slice::from_raw_parts(pointer as *const i32, count).iter().map(|&x| x as f64).collect(),
            DataType::F32 => std::slice::from_raw_parts(pointer as *const f32, count).iter().map(|&x| x as f64).collect(),
            DataType::F16 => std::slice::from_raw_parts(pointer as *const f16, count).iter().map(|x| x.to_f64()).collect(),
            DataType::BF16 => std::slice::from_raw_parts(pointer as *const bf16, count).iter().map(|x| x.to_f64()).collect(),
            other => panic!("Unsupported output_dtype: {other:?}"),
        }
    }
}

fn ndarray_reference(combo: &DtypeCombo, a_bytes: &[u8], b_bytes: &[u8], shape: &TestShape) -> Vec<f64> {
    let a_f64 = bytes_to_f64(combo.a_dtype, a_bytes);
    let b_f64 = bytes_to_f64(combo.b_dtype, b_bytes);

    let a_array = Array2::from_shape_vec((shape.batch, shape.input_dim), a_f64).expect("A shape");
    let b_array = Array2::from_shape_vec((shape.output_dim, shape.input_dim), b_f64).expect("B shape");
    let result = a_array.dot(&b_array.t());

    match combo.output_dtype {
        DataType::I32 => result.iter().map(|&x| (x as i32) as f64).collect(),
        DataType::F32 => result.iter().map(|&x| (x as f32) as f64).collect(),
        DataType::F16 => result.iter().map(|&x| f16::from_f64(x).to_f64()).collect(),
        DataType::BF16 => result.iter().map(|&x| bf16::from_f64(x).to_f64()).collect(),
        _ => result.iter().copied().collect(),
    }
}

fn tolerance_for(combo: &DtypeCombo, shape: &TestShape) -> f64 {
    match (combo.a_dtype, combo.b_dtype, combo.output_dtype) {
        (DataType::I8, DataType::I8, DataType::I32) => 0.0,
        (DataType::I8, DataType::BF16, DataType::BF16) => 0.05 * (shape.input_dim as f64 / 1024.0).sqrt(),
        (DataType::BF16, DataType::BF16, DataType::BF16) => {
            let base = 0.01 * (shape.input_dim as f64 / 1024.0).sqrt();
            base * (1.0 + (shape.batch as f64).ln() / std::f64::consts::LN_2 * 0.02)
        },
        _ => 0.01,
    }
}

// -- Test runner --------------------------------------------------------------

#[derive(Serialize)]
struct TestResult {
    combo: String,
    shape: String,
    dispatch_path: String,
    passed: bool,
    max_diff: f64,
    tolerance: f64,
}

fn run_correctness_case(
    context: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
    dispatch_path_name: &str,
    variant: MatmulVariant,
) -> TestResult {
    let a_bytes = generate_typed_data(combo.a_dtype, shape.batch * shape.input_dim, 13, -6);
    let b_bytes = generate_typed_data(combo.b_dtype, shape.output_dim * shape.input_dim, 17, -8);

    let metal_result = run_metal_matmul(context, combo, &a_bytes, &b_bytes, shape, variant);
    let reference = ndarray_reference(combo, &a_bytes, &b_bytes, shape);

    let tolerance = tolerance_for(combo, shape);
    let max_diff = metal_result
        .iter()
        .zip(reference.iter())
        .map(|(&metal_value, &reference_value)| (metal_value - reference_value).abs())
        .fold(0.0f64, f64::max);

    TestResult {
        combo: format!("{combo}"),
        shape: format!("{shape}"),
        dispatch_path: dispatch_path_name.to_owned(),
        passed: max_diff <= tolerance,
        max_diff,
        tolerance,
    }
}

fn run_metal_matmul(
    context: &MetalContext,
    combo: &DtypeCombo,
    a_bytes: &[u8],
    b_bytes: &[u8],
    shape: &TestShape,
    variant: MatmulVariant,
) -> Vec<f64> {
    let a_buffer = context.device.new_buffer_with_data(a_bytes, MTLResourceOptions::STORAGE_MODE_SHARED).expect("A buffer");
    let b_buffer = context.device.new_buffer_with_data(b_bytes, MTLResourceOptions::STORAGE_MODE_SHARED).expect("B buffer");
    let mut d_buffer = context.device
        .new_buffer(shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("D buffer");

    let mut kernel = MatmulMetalKernel::new(context, combo.output_dtype).expect("kernel creation");
    let mut command_buffer = context.create_command_buffer().unwrap().start_encoding();
    let arguments = make_full_precision_arguments(&a_buffer, &b_buffer, &mut d_buffer, shape);

    match variant {
        MatmulVariant::Gemv => kernel.encode_gemv(context, &mut command_buffer, arguments),
        MatmulVariant::GemmMpp => kernel.encode_gemm_mpp(context, &mut command_buffer, arguments),
        MatmulVariant::Gemm => kernel.encode_gemm(context, &mut command_buffer, arguments),
    }
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    output_to_f64(combo.output_dtype, &d_buffer, shape.batch * shape.output_dim)
}

// -- Output -------------------------------------------------------------------

fn print_results_table(results: &[TestResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Dtype Combo", "Shape (BxKxN)", "Dispatch", "Status", "Max Diff", "Tolerance"]);

    for result in results {
        table.add_row(vec![
            result.combo.clone(), result.shape.clone(), result.dispatch_path.clone(),
            if result.passed { "PASS".into() } else { "FAIL".into() },
            format!("{:.6}", result.max_diff), format!("{:.6}", result.tolerance),
        ]);
    }

    for col in [4, 5] {
        if let Some(column) = table.column_mut(col) {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }

    println!("{table}");
}

// -- Test entry point ---------------------------------------------------------

#[test]
#[ignore]
fn matmul_correctness() {
    let context = MetalContext::new().expect("Metal context required");

    let combos = vec![DtypeCombo {
        a_dtype: DataType::F16,
        b_dtype: DataType::F16,
        output_dtype: DataType::F16,
    }];
    let shapes = test_shapes();

    eprintln!(
        "Matmul correctness: {} combos x {} shapes, testing all applicable dispatch paths",
        combos.len(), shapes.len(),
    );

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::with_template("{spinner} {pos} tests [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::new();

    for combo in &combos {
        for shape in &shapes {
            for (path_name, variant) in &applicable_variants(&context, combo, shape) {
                progress_bar.set_message(format!("{} {} {}", combo, shape, path_name));
                results.push(run_correctness_case(&context, combo, shape, path_name, *variant));
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
