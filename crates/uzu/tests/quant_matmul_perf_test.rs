#![cfg(target_os = "macos")]

use std::time::Instant;

use comfy_table::{CellAlignment, ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use serde::Serialize;
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending, Context,
            kernel::quant_matmul::{
                QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernelEncodable,
                QuantizedMatmulType,
            },
        },
        metal::Metal,
    },
    config::QuantizationMode,
};

const GROUP_SIZE: usize = 128;
const WARMUP_ITERATIONS: usize = 3;
const BENCHMARK_ITERATIONS: usize = 20;

#[derive(Clone)]
struct QuantConfig {
    quant_type: QuantizedMatmulType,
    bits: usize,
    data_type: DataType,
}

impl std::fmt::Display for QuantConfig {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let qt = match self.quant_type {
            QuantizedMatmulType::ZeroPoint => "ZP",
            QuantizedMatmulType::Mlx => "Mlx",
        };
        write!(f, "{qt}/{}-bit/{:?}", self.bits, self.data_type)
    }
}

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

#[derive(Serialize)]
struct PerfResult {
    config: String,
    shape: String,
    dispatch_path: String,
    duration_ms: f64,
    gflops: f64,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn quant_dispatch_path(
    batch: usize,
    output_dim: usize,
) -> &'static str {
    if batch < 32 || output_dim == 1 {
        "MatrixVector"
    } else {
        "MatrixMatrix"
    }
}

fn write_json_results<T: Serialize>(
    test_name: &str,
    device: &str,
    results: &[T],
) {
    if let Ok(dir) = std::env::var("UZU_TEST_RESULTS_DIR") {
        let path = std::path::Path::new(&dir);
        std::fs::create_dir_all(path).expect("create results dir");
        let file = path.join(format!("{test_name}.json"));
        let wrapper = serde_json::json!({ "device": device, "results": results });
        let json = serde_json::to_string_pretty(&wrapper).expect("serialize");
        std::fs::write(&file, json).expect("write results");
        eprintln!("Results written to {}", file.display());
    }
}

fn test_configs() -> Vec<QuantConfig> {
    vec![
        QuantConfig {
            quant_type: QuantizedMatmulType::Mlx,
            bits: 4,
            data_type: DataType::BF16,
        },
        QuantConfig {
            quant_type: QuantizedMatmulType::Mlx,
            bits: 8,
            data_type: DataType::BF16,
        },
        QuantConfig {
            quant_type: QuantizedMatmulType::ZeroPoint,
            bits: 4,
            data_type: DataType::F16,
        },
        QuantConfig {
            quant_type: QuantizedMatmulType::ZeroPoint,
            bits: 8,
            data_type: DataType::F16,
        },
    ]
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

fn shape_valid_for_config(shape: &TestShape) -> bool {
    shape.input_dim >= GROUP_SIZE
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

fn benchmark_single(
    ctx: &<Metal as Backend>::Context,
    config: &QuantConfig,
    shape: &TestShape,
) -> PerfResult {
    let error_result = |msg: String| PerfResult {
        config: format!("{}", config),
        shape: format!("{}", shape),
        dispatch_path: String::new(),
        duration_ms: 0.0,
        gflops: 0.0,
        status: "error".into(),
        error: Some(msg),
    };

    let quant_mode = match config.bits {
        4 => QuantizationMode::UInt4,
        8 => QuantizationMode::Int8,
        _ => return error_result(format!("unsupported bits: {}", config.bits)),
    };

    let kernel = match QuantizedMatmulKernelEncodable::<Metal>::new(
        ctx,
        QuantizedMatmulConfiguration {
            data_type: config.data_type,
            group_size: GROUP_SIZE,
            input_dim: shape.input_dim,
            output_dim: shape.output_dim,
            mode: quant_mode,
            quantization_type: config.quant_type,
            weights_transposed: true,
        },
    ) {
        Ok(k) => k,
        Err(e) => return error_result(format!("kernel: {e}")),
    };

    let weight_elem_count = shape.output_dim * shape.input_dim;
    let w_bytes = if config.bits == 4 {
        (weight_elem_count + 3) / 4 * 2
    } else {
        weight_elem_count
    };
    let num_groups = (shape.input_dim + GROUP_SIZE - 1) / GROUP_SIZE;
    let scales_count = shape.output_dim * num_groups;
    let biases_count = scales_count;

    let x_bytes = shape.batch * shape.input_dim * config.data_type.size_in_bytes();
    let y_bytes = shape.batch * shape.output_dim * config.data_type.size_in_bytes();
    let s_bytes = scales_count * config.data_type.size_in_bytes();
    let b_bytes = match config.quant_type {
        QuantizedMatmulType::ZeroPoint => {
            let zp_stride = if config.bits == 4 {
                ((num_groups + 1) / 2).max(1)
            } else {
                num_groups
            };
            shape.output_dim * zp_stride
        },
        QuantizedMatmulType::Mlx => biases_count * config.data_type.size_in_bytes(),
    };

    let alloc =
        |size| ctx.device.new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED).ok_or("buffer alloc failed");

    let (w_buf, x_buf, y_buf, s_buf, b_buf) =
        match (alloc(w_bytes), alloc(x_bytes), alloc(y_bytes), alloc(s_bytes), alloc(b_bytes)) {
            (Ok(w), Ok(x), Ok(y), Ok(s), Ok(b)) => (w, x, y, s, b),
            _ => return error_result("buffer alloc failed".into()),
        };

    fill_buffer_random(&w_buf, w_bytes);
    fill_buffer_random(&x_buf, x_bytes);
    fill_buffer_random(&s_buf, s_bytes);
    fill_buffer_random(&b_buf, b_bytes);

    let mut y_buf = y_buf;

    for _ in 0..WARMUP_ITERATIONS {
        let arguments = QuantizedMatmulArguments {
            a_buffer: &x_buf,
            a_offset: 0,
            b_buffer: &w_buf,
            scales_buffer: &s_buf,
            zero_points_or_biases_buffer: &b_buf,
            output_buffer: &mut y_buf,
            batch: shape.batch,
            input_dim: shape.input_dim,
            output_dim: shape.output_dim,
            quantization_type: config.quant_type,
        };
        let mut command_buffer = match ctx.create_command_buffer() {
            Ok(cb) => cb.start_encoding(),
            Err(e) => return error_result(format!("warmup cb: {e}")),
        };
        if let Err(e) = kernel.encode(&mut command_buffer, arguments) {
            return error_result(format!("warmup encode: {e}"));
        }
        if let Err(e) = command_buffer.end_encoding().submit().wait_until_completed() {
            return error_result(format!("warmup submit: {e}"));
        }
    }

    let start = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        let arguments = QuantizedMatmulArguments {
            a_buffer: &x_buf,
            a_offset: 0,
            b_buffer: &w_buf,
            scales_buffer: &s_buf,
            zero_points_or_biases_buffer: &b_buf,
            output_buffer: &mut y_buf,
            batch: shape.batch,
            input_dim: shape.input_dim,
            output_dim: shape.output_dim,
            quantization_type: config.quant_type,
        };
        let mut command_buffer = match ctx.create_command_buffer() {
            Ok(cb) => cb.start_encoding(),
            Err(e) => return error_result(format!("bench cb: {e}")),
        };
        if let Err(e) = kernel.encode(&mut command_buffer, arguments) {
            return error_result(format!("bench encode: {e}"));
        }
        if let Err(e) = command_buffer.end_encoding().submit().wait_until_completed() {
            return error_result(format!("bench submit: {e}"));
        }
    }
    let elapsed = start.elapsed();

    let duration_ms = elapsed.as_secs_f64() * 1000.0 / BENCHMARK_ITERATIONS as f64;
    let flops = 2.0 * shape.batch as f64 * shape.input_dim as f64 * shape.output_dim as f64;
    let gflops = flops / (duration_ms / 1000.0) / 1e9;

    PerfResult {
        config: format!("{}", config),
        shape: format!("{}", shape),
        dispatch_path: quant_dispatch_path(shape.batch, shape.output_dim).to_owned(),
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
        .set_header(vec!["Config", "Shape (BxKxN)", "Dispatch", "GFLOPS", "ms/iter", "Status"]);

    for r in results {
        let (gflops_str, ms_str, status_str) = if r.status == "ok" {
            (format!("{:.1}", r.gflops), format!("{:.3}", r.duration_ms), "ok".into())
        } else {
            ("-".into(), "-".into(), format!("ERR: {}", r.error.as_deref().unwrap_or("?")))
        };
        table.add_row(vec![r.config.clone(), r.shape.clone(), r.dispatch_path.clone(), gflops_str, ms_str, status_str]);
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
fn quant_matmul_perf() {
    let ctx = <Metal as Backend>::Context::new().expect("Metal context required");

    let configs = test_configs();
    let shapes = test_shapes();

    let valid_cases: Vec<_> =
        configs.iter().flat_map(|c| shapes.iter().filter(|s| shape_valid_for_config(s)).map(move |s| (c, s))).collect();

    let total = valid_cases.len();
    eprintln!(
        "Quant matmul perf: {} configs, {} valid cases, {} warmup + {} benchmark iterations each",
        configs.len(),
        total,
        WARMUP_ITERATIONS,
        BENCHMARK_ITERATIONS,
    );

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::with_capacity(total);

    for (config, shape) in &valid_cases {
        pb.set_message(format!("{} {}", config, shape));
        let result = benchmark_single(&ctx, config, shape);
        results.push(result);
        pb.inc(1);
    }

    pb.finish_with_message("done");
    print_results_table(&results);
    write_json_results("quant_matmul_perf", &ctx.device.name(), &results);

    let errors: Vec<_> = results.iter().filter(|r| r.status != "ok").collect();
    if !errors.is_empty() {
        eprintln!("{} / {} cases had errors", errors.len(), results.len());
    }
}
