use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use objc2::runtime::ProtocolObject;
use serde::Serialize;
use thiserror::Error;
use uzu::{
    DataType,
    backends::{
        common::{
            CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial,
            CommandBufferPending, Context,
            kernel::matmul::MatmulKernel,
        },
        metal::{MetalContext, kernel::matmul::MatmulMetalKernel},
    },
};

use crate::common::matmul::{
    DtypeCombo, MatmulVariant, TestShape, applicable_variants, make_full_precision_arguments, test_combos,
    write_json_results,
};

// -- Shapes -------------------------------------------------------------------

const BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

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
    let grid_shapes = BATCH_SIZES.iter().flat_map(|&batch| {
        [512, 1024, 2048].iter().flat_map(move |&input_dim| {
            [512, 1024, 2048].iter().map(move |&output_dim| TestShape {
                batch, input_dim, output_dim,
            })
        })
    });

    let model_shapes = MODEL_DIMS.iter().flat_map(|&(input_dim, output_dim)| {
        BATCH_SIZES.iter().map(move |&batch| TestShape {
            batch, input_dim, output_dim,
        })
    });

    grid_shapes.chain(model_shapes).collect()
}

// -- Benchmark runner ---------------------------------------------------------

#[derive(Debug, Error)]
enum BenchError {
    #[error("failed to create command buffer")]
    CommandBuffer,
    #[error("GPU timestamps unavailable")]
    GpuTimestamps,
    #[error("kernel creation failed: {0}")]
    Kernel(String),
    #[error("buffer allocation failed")]
    BufferAllocation,
    #[error("warmup iteration {iteration}: {source}")]
    Warmup { iteration: usize, source: Box<BenchError> },
    #[error("benchmark iteration {iteration}: {source}")]
    Benchmark { iteration: usize, source: Box<BenchError> },
}

const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 10;

fn fill_buffer_random(buffer: &ProtocolObject<dyn MTLBuffer>, byte_count: usize) {
    let pointer = buffer.contents().as_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(pointer, byte_count) };
    for (i, byte) in slice.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }
}

fn encode_and_run(
    context: &MetalContext,
    kernel: &mut MatmulMetalKernel,
    shape: &TestShape,
    a_buffer: &objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>,
    b_buffer: &objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>,
    d_buffer: &mut objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>,
    variant: MatmulVariant,
) -> Result<f64, BenchError> {
    let mut command_buffer = context.create_command_buffer().map_err(|_| BenchError::CommandBuffer)?.start_encoding();

    let arguments = make_full_precision_arguments(a_buffer, b_buffer, d_buffer, shape);
    match variant {
        MatmulVariant::Gemv => kernel.encode_gemv(context, &mut command_buffer, arguments),
        MatmulVariant::GemmMpp => kernel.encode_gemm_mpp(context, &mut command_buffer, arguments),
        MatmulVariant::Gemm => kernel.encode_gemm(context, &mut command_buffer, arguments),
    }

    let completed =
        command_buffer.end_encoding().submit().wait_until_completed().map_err(|_| BenchError::CommandBuffer)?;

    completed.gpu_execution_time().map(|duration| duration.as_secs_f64() * 1000.0).ok_or(BenchError::GpuTimestamps)
}

fn run_benchmark(
    context: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
    variant: MatmulVariant,
) -> Result<f64, BenchError> {
    let mut kernel = MatmulMetalKernel::new(context, combo.output_dtype)
        .map_err(|e| BenchError::Kernel(e.to_string()))?;

    let a_byte_count = shape.batch * shape.input_dim * combo.a_dtype.size_in_bytes();
    let b_byte_count = shape.output_dim * shape.input_dim * combo.b_dtype.size_in_bytes();
    let d_byte_count = shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes();

    let (a_buffer, b_buffer, mut d_buffer) = match (
        context.device.new_buffer(a_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
        context.device.new_buffer(b_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
        context.device.new_buffer(d_byte_count, MTLResourceOptions::STORAGE_MODE_SHARED),
    ) {
        (Some(a), Some(b), Some(d)) => (a, b, d),
        _ => return Err(BenchError::BufferAllocation),
    };

    fill_buffer_random(&a_buffer, a_byte_count);
    fill_buffer_random(&b_buffer, b_byte_count);

    for iteration in 0..WARMUP_ITERATIONS {
        encode_and_run(context, &mut kernel, shape, &a_buffer, &b_buffer, &mut d_buffer, variant)
            .map_err(|source| BenchError::Warmup { iteration, source: Box::new(source) })?;
    }

    let mut gpu_time_total_ms = 0.0;
    for iteration in 0..BENCHMARK_ITERATIONS {
        gpu_time_total_ms += encode_and_run(context, &mut kernel, shape, &a_buffer, &b_buffer, &mut d_buffer, variant)
            .map_err(|source| BenchError::Benchmark { iteration, source: Box::new(source) })?;
    }

    Ok(gpu_time_total_ms / BENCHMARK_ITERATIONS as f64)
}

// -- Output -------------------------------------------------------------------

#[derive(Serialize)]
struct PerfResult {
    combo: String,
    shape: String,
    dispatch_path: String,
    duration_ms: f64,
    gflops: f64,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn benchmark_single(
    context: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
    path_name: &str,
    variant: MatmulVariant,
) -> PerfResult {
    match run_benchmark(context, combo, shape, variant) {
        Ok(duration_ms) => {
            let flops = 2.0 * shape.batch as f64 * shape.input_dim as f64 * shape.output_dim as f64;
            let gflops = flops / (duration_ms / 1000.0) / 1e9;
            PerfResult {
                combo: format!("{combo}"), shape: format!("{shape}"),
                dispatch_path: path_name.to_owned(), duration_ms, gflops,
                status: "ok".into(), error: None,
            }
        },
        Err(error) => PerfResult {
            combo: format!("{combo}"), shape: format!("{shape}"),
            dispatch_path: path_name.to_owned(), duration_ms: 0.0, gflops: 0.0,
            status: "error".into(), error: Some(error.to_string()),
        },
    }
}

fn print_results_table(results: &[PerfResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Dtype Combo", "Shape (MxKxN)", "Dispatch", "GFLOPS", "ms/iter", "Status"]);

    for r in results {
        let (gflops_str, ms_str, status_str) = if r.status == "ok" {
            (format!("{:.1}", r.gflops), format!("{:.3}", r.duration_ms), "ok".into())
        } else {
            ("-".into(), "-".into(), format!("ERR: {}", r.error.as_deref().unwrap_or("?")))
        };
        table.add_row(vec![r.combo.clone(), r.shape.clone(), r.dispatch_path.clone(), gflops_str, ms_str, status_str]);
    }

    for col in [3, 4] {
        if let Some(column) = table.column_mut(col) {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }

    println!("{table}");
}

// -- Test entry point ---------------------------------------------------------

#[test]
#[ignore]
fn matmul_perf() {
    let context = MetalContext::new().expect("Metal context required");

    let dtype_combos = test_combos();
    let shapes = test_shapes();

    eprintln!(
        "Matmul perf: {} combos x {} shapes, testing all applicable dispatch paths",
        dtype_combos.len(), shapes.len(),
    );

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::with_template("{spinner} {pos} benchmarks [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::new();

    for combo in &dtype_combos {
        for shape in &shapes {
            for (path_name, variant) in &applicable_variants(&context, combo, shape) {
                progress_bar.set_message(format!("{} {} {}", combo, shape, path_name));
                results.push(benchmark_single(&context, combo, shape, path_name, *variant));
                progress_bar.inc(1);
            }
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
