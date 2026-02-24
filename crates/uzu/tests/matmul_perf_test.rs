#![cfg(target_os = "macos")]

use std::time::Instant;

use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt, MTLResourceOptions};
use serde::Serialize;
use uzu::{
    DataType,
    backends::{
        common::{
            Context,
            kernel::matmul::{MatmulArguments, MatmulDispatchDescriptor, MatmulKernel},
        },
        metal::{Metal, MetalContext, choose_dispatch_descriptor},
    },
};

const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 10;

#[derive(Clone)]
struct TestShape {
    batch: usize,
    input_dim: usize,
    output_dim: usize,
}

impl std::fmt::Display for TestShape {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
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
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{:?}*{:?}->{:?}", self.a_dtype, self.b_dtype, self.output_dtype)
    }
}

#[derive(Serialize)]
struct PerfResult {
    combo: String,
    shape: String,
    dispatch_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    mpp_used: Option<bool>,
    duration_ms: f64,
    gflops: f64,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn dispatch_path_name(descriptor: &MatmulDispatchDescriptor) -> &'static str {
    match descriptor {
        MatmulDispatchDescriptor::Gemv(_) => "Gemv",
        MatmulDispatchDescriptor::SplitK(_) => "SplitK",
        MatmulDispatchDescriptor::Gemm(_) => "Gemm",
        MatmulDispatchDescriptor::GemmMpp(_) => "GemmMpp",
        MatmulDispatchDescriptor::GemmScalarInt(_) => "GemmScalarInt",
    }
}

fn write_json_results<T: Serialize>(
    test_name: &str,
    device: &str,
    mpp: bool,
    results: &[T],
) {
    if let Ok(dir) = std::env::var("UZU_TEST_RESULTS_DIR") {
        let path = std::path::Path::new(&dir);
        std::fs::create_dir_all(path).expect("create results dir");
        let file = path.join(format!("{test_name}.json"));
        let wrapper = serde_json::json!({ "device": device, "mpp_available": mpp, "results": results });
        let json = serde_json::to_string_pretty(&wrapper).expect("serialize");
        std::fs::write(&file, json).expect("write results");
        eprintln!("Results written to {}", file.display());
    }
}

fn test_shapes() -> Vec<TestShape> {
    let mut shapes = Vec::new();

    for &batch in &[512, 1024, 2048] {
        for &output_dim in &[512, 1024, 2048] {
            for &input_dim in &[1, 2, 4, 8, 16, 32, 64] {
                shapes.push(TestShape {
                    batch,
                    input_dim,
                    output_dim,
                });
            }
        }
    }

    let model_dims: &[(usize, usize)] = &[
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
    for &(input_dim, output_dim) in model_dims {
        for &batch in &[1, 128] {
            shapes.push(TestShape {
                batch,
                input_dim,
                output_dim,
            });
        }
    }

    shapes
}

fn fill_buffer_random(
    buf: &objc2::runtime::ProtocolObject<dyn MTLBuffer>,
    byte_count: usize,
) {
    let ptr = buf.contents().as_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, byte_count) };
    for (i, byte) in slice.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }
}

fn encode_and_run(
    ctx: &MetalContext,
    kernel: &mut MatmulKernel<Metal>,
    arguments: MatmulArguments<Metal>,
    descriptor: &MatmulDispatchDescriptor,
) -> Result<(), String> {
    let cb = ctx.command_queue.command_buffer().ok_or("Failed to create command buffer")?.to_owned();
    let enc = cb.new_compute_command_encoder().ok_or("Failed to create compute encoder")?;
    kernel.encode_with_descriptor(ctx, arguments, descriptor, &enc).map_err(|e| format!("encode: {e}"))?;
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(())
}

fn benchmark_single(
    ctx: &MetalContext,
    combo: &DtypeCombo,
    shape: &TestShape,
) -> PerfResult {
    let error_result = |msg: String| PerfResult {
        combo: format!("{}", combo),
        shape: format!("{}", shape),
        dispatch_path: String::new(),
        mpp_used: None,
        duration_ms: 0.0,
        gflops: 0.0,
        status: "error".into(),
        error: Some(msg),
    };

    let mut kernel = match MatmulKernel::<Metal>::new_mixed(combo.a_dtype, combo.b_dtype, combo.output_dtype) {
        Ok(k) => k,
        Err(e) => return error_result(format!("kernel: {e}")),
    };

    let a_bytes = shape.batch * shape.input_dim * combo.a_dtype.size_in_bytes();
    let b_bytes = shape.output_dim * shape.input_dim * combo.b_dtype.size_in_bytes();
    let d_bytes = shape.batch * shape.output_dim * combo.output_dtype.size_in_bytes();

    let (a_buf, b_buf, d_buf) = match (
        ctx.device.new_buffer(a_bytes, MTLResourceOptions::STORAGE_MODE_SHARED),
        ctx.device.new_buffer(b_bytes, MTLResourceOptions::STORAGE_MODE_SHARED),
        ctx.device.new_buffer(d_bytes, MTLResourceOptions::STORAGE_MODE_SHARED),
    ) {
        (Some(a), Some(b), Some(d)) => (a, b, d),
        _ => return error_result("buffer allocation failed".into()),
    };

    fill_buffer_random(&a_buf, a_bytes);
    fill_buffer_random(&b_buf, b_bytes);

    let mut arguments = MatmulArguments {
        a: &a_buf,
        a_offset: 0,
        b: &b_buf,
        d: &d_buf,
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

    let descriptor = match choose_dispatch_descriptor(ctx, combo.a_dtype, combo.b_dtype, combo.output_dtype, &arguments)
    {
        Ok(d) => d,
        Err(e) => return error_result(format!("dispatch: {e}")),
    };

    let path = dispatch_path_name(&descriptor).to_owned();
    let mpp_used = Some(matches!(descriptor, MatmulDispatchDescriptor::GemmMpp(_)));

    for i in 0..WARMUP_ITERATIONS {
        let args = MatmulArguments {
            a: &a_buf,
            a_offset: 0,
            b: &b_buf,
            d: &d_buf,
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
        if let Err(e) = encode_and_run(ctx, &mut kernel, args, &descriptor) {
            return error_result(format!("warmup {i}: {e}"));
        }
    }

    let start = Instant::now();
    for i in 0..BENCHMARK_ITERATIONS {
        let args = MatmulArguments {
            a: &a_buf,
            a_offset: 0,
            b: &b_buf,
            d: &d_buf,
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
        if let Err(e) = encode_and_run(ctx, &mut kernel, args, &descriptor) {
            return error_result(format!("bench {i}: {e}"));
        }
    }
    let elapsed = start.elapsed();

    let duration_ms = elapsed.as_secs_f64() * 1000.0 / BENCHMARK_ITERATIONS as f64;
    let flops = 2.0 * shape.batch as f64 * shape.input_dim as f64 * shape.output_dim as f64;
    let gflops = flops / (duration_ms / 1000.0) / 1e9;

    PerfResult {
        combo: format!("{}", combo),
        shape: format!("{}", shape),
        dispatch_path: path,
        mpp_used,
        duration_ms,
        gflops,
        status: "ok".into(),
        error: None,
    }
}

fn print_results_table(results: &[PerfResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Dtype Combo", "Shape (BxKxN)", "Dispatch", "MPP Used", "GFLOPS", "ms/iter", "Status"]);

    for r in results {
        let (gflops_str, ms_str, status_str) = if r.status == "ok" {
            (format!("{:.1}", r.gflops), format!("{:.3}", r.duration_ms), "ok".into())
        } else {
            ("-".into(), "-".into(), format!("ERR: {}", r.error.as_deref().unwrap_or("?")))
        };
        let mpp_used = match r.mpp_used {
            Some(true) => "yes",
            Some(false) => "no",
            None => "-",
        };
        table.add_row(vec![
            r.combo.clone(),
            r.shape.clone(),
            r.dispatch_path.clone(),
            mpp_used.into(),
            gflops_str,
            ms_str,
            status_str,
        ]);
    }

    for col in [4, 5] {
        if let Some(column) = table.column_mut(col) {
            column.set_cell_alignment(CellAlignment::Right);
        }
    }

    println!("{table}");
}

#[test]
#[ignore]
fn matmul_perf() {
    let ctx = MetalContext::new().expect("Metal context required");

    eprintln!("MPP available: {}", ctx.is_mpp_available());

    let combos = vec![
        DtypeCombo {
            a_dtype: DataType::BF16,
            b_dtype: DataType::BF16,
            output_dtype: DataType::BF16,
        },
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::I8,
            output_dtype: DataType::I32,
        },
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::BF16,
            output_dtype: DataType::BF16,
        },
    ];

    let shapes = test_shapes();
    let total = combos.len() * shapes.len();

    eprintln!(
        "Matmul perf: {} combos x {} shapes = {} cases, {} warmup + {} benchmark iterations each",
        combos.len(),
        shapes.len(),
        total,
        WARMUP_ITERATIONS,
        BENCHMARK_ITERATIONS,
    );

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::with_capacity(total);

    for combo in &combos {
        for shape in &shapes {
            pb.set_message(format!("{} {}", combo, shape));
            let result = benchmark_single(&ctx, combo, shape);
            results.push(result);
            pb.inc(1);
        }
    }

    pb.finish_with_message("done");
    print_results_table(&results);
    write_json_results("matmul_perf", &ctx.device.name(), ctx.is_mpp_available(), &results);

    let errors: Vec<_> = results.iter().filter(|r| r.status != "ok").collect();
    if !errors.is_empty() {
        eprintln!("{} / {} cases had errors", errors.len(), results.len());
    }
}
