#![cfg(target_os = "macos")]

use comfy_table::{ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::{ProgressBar, ProgressStyle};
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt, MTLResourceOptions};
use objc2::rc::Retained;
use serde::Serialize;
use uzu::{
    DataType,
    backends::{
        common::{
            Backend, Context,
            kernel::quant_matmul::{
                QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernelEncodable,
                QuantizedMatmulType,
            },
        },
        metal::Metal,
    },
    config::QuantizationMode,
};

const GROUP_SIZE: usize = 64;

#[derive(Clone)]
struct QuantConfig {
    quant_type: QuantizedMatmulType,
    bits: usize,
    weights_transposed: bool,
}

impl std::fmt::Display for QuantConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let qt = match self.quant_type {
            QuantizedMatmulType::ZeroPoint => "ZP",
            QuantizedMatmulType::Mlx => "Mlx",
        };
        let tr = if self.weights_transposed { "T" } else { "N" };
        write!(f, "{qt}/{}-bit/{tr}", self.bits)
    }
}

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

#[derive(Serialize)]
struct TestResult {
    config: String,
    shape: String,
    dispatch_path: String,
    passed: bool,
    mismatch_count: usize,
    total_outputs: usize,
}

fn quant_dispatch_path(batch: usize, output_dim: usize) -> &'static str {
    if batch < 32 || output_dim == 1 { "MatrixVector" } else { "MatrixMatrix" }
}

fn write_json_results<T: Serialize>(test_name: &str, device: &str, mpp: bool, results: &[T]) {
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

fn test_configs() -> Vec<QuantConfig> {
    let mut configs = Vec::new();
    for &quant_type in &[QuantizedMatmulType::ZeroPoint, QuantizedMatmulType::Mlx] {
        for &bits in &[4usize, 8] {
            for &weights_transposed in &[true, false] {
                configs.push(QuantConfig { quant_type, bits, weights_transposed });
            }
        }
    }
    configs
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

fn shape_valid_for_config(shape: &TestShape, config: &QuantConfig) -> bool {
    let grouped_dim = if config.weights_transposed { shape.input_dim } else { shape.output_dim };
    grouped_dim >= GROUP_SIZE
}

// --- weight generation ---

fn create_test_weights(output_dim: usize, input_dim: usize, weights_transposed: bool, bits: usize) -> Vec<u8> {
    let mut weights = Vec::with_capacity(output_dim * input_dim);
    if weights_transposed {
        for row in 0..output_dim {
            for _col in 0..input_dim {
                let v = if bits == 4 { ((row + 1) & 0x0F) as u8 } else { ((row + 1) & 0xFF) as u8 };
                weights.push(v);
            }
        }
    } else {
        for _row in 0..input_dim {
            for col in 0..output_dim {
                let v = if bits == 4 { ((col + 1) & 0x0F) as u8 } else { (col & 0xFF) as u8 };
                weights.push(v);
            }
        }
    }
    weights
}

fn pack_u4_weights(values: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((values.len() + 3) / 4 * 2);
    for chunk in values.chunks(4) {
        let w0 = (*chunk.first().unwrap_or(&0) as u16) & 0x0F;
        let w1 = ((*chunk.get(1).unwrap_or(&0) as u16) & 0x0F) << 4;
        let w2 = ((*chunk.get(2).unwrap_or(&0) as u16) & 0x0F) << 8;
        let w3 = ((*chunk.get(3).unwrap_or(&0) as u16) & 0x0F) << 12;
        let word: u16 = w0 | w1 | w2 | w3;
        out.push(word as u8);
        out.push((word >> 8) as u8);
    }
    out
}

// --- quantization params generation ---

fn get_zp_value(zero_points: &[u8], stride: usize, row_idx: usize, group_idx: usize, bits: usize) -> f32 {
    if bits == 4 {
        let byte_index = row_idx * stride + (group_idx >> 1);
        let byte = zero_points.get(byte_index).copied().unwrap_or(0);
        if (group_idx & 1) == 0 { (byte & 0x0F) as f32 } else { ((byte >> 4) & 0x0F) as f32 }
    } else {
        zero_points.get(row_idx * stride + group_idx).copied().unwrap_or(0) as f32
    }
}

struct QuantParams {
    scales_f32: Vec<f32>,
    biases_f32: Vec<f32>,
    zero_points: Vec<u8>,
    zero_points_stride: usize,
}

fn generate_quant_params(config: &QuantConfig, shape: &TestShape) -> QuantParams {
    let num_groups = if !config.weights_transposed {
        (shape.output_dim + GROUP_SIZE - 1) / GROUP_SIZE
    } else {
        (shape.input_dim + GROUP_SIZE - 1) / GROUP_SIZE
    };
    let primary_dim = if !config.weights_transposed { shape.input_dim } else { shape.output_dim };

    let len = primary_dim * num_groups;
    let scales = vec![1.0f32; len];
    let mut biases = vec![0.0f32; len];

    let zp_stride = if config.bits == 4 { ((num_groups + 1) / 2).max(1) } else { num_groups };
    let zp_len = if !config.weights_transposed {
        shape.input_dim * zp_stride
    } else {
        shape.output_dim * zp_stride
    };
    let mut zero_points = vec![0u8; zp_len];

    if config.quant_type == QuantizedMatmulType::ZeroPoint {
        if !config.weights_transposed {
            for k in 0..shape.input_dim {
                for g in 0..num_groups {
                    let k_eff = k / GROUP_SIZE;
                    let base_val = ((k_eff * 5 + g * 7) & 0xFF) as u8;
                    let zp_val_u8 = if config.bits == 4 { base_val & 0x0F } else { base_val };
                    if config.bits == 4 {
                        let byte_index = k * zp_stride + (g >> 1);
                        if (g & 1) == 0 {
                            zero_points[byte_index] = (zero_points[byte_index] & 0xF0) | (zp_val_u8 & 0x0F);
                        } else {
                            zero_points[byte_index] = (zero_points[byte_index] & 0x0F) | ((zp_val_u8 & 0x0F) << 4);
                        }
                    } else {
                        zero_points[k * zp_stride + g] = zp_val_u8;
                    }
                    let zp_val = if config.bits == 4 { (zp_val_u8 & 0x0F) as f32 } else { zp_val_u8 as f32 };
                    biases[k * num_groups + g] = -scales[k * num_groups + g] * zp_val;
                }
            }
        } else {
            for j in 0..shape.output_dim {
                for g in 0..num_groups {
                    let base_val = j + 3 * g;
                    let zp_val: u8 = if config.bits == 4 { (base_val as u8) & 0x0F } else { base_val as u8 };
                    if config.bits == 4 {
                        let byte_index = j * zp_stride + (g >> 1);
                        if (g & 1) == 0 {
                            zero_points[byte_index] = (zero_points[byte_index] & 0xF0) | (zp_val & 0x0F);
                        } else {
                            zero_points[byte_index] = (zero_points[byte_index] & 0x0F) | ((zp_val & 0x0F) << 4);
                        }
                    } else {
                        zero_points[j * zp_stride + g] = zp_val;
                    }
                    biases[j * num_groups + g] = -scales[j * num_groups + g] * (zp_val as f32);
                }
            }
        }
    }

    if config.quant_type == QuantizedMatmulType::Mlx {
        if !config.weights_transposed {
            for k in 0..shape.input_dim {
                for g in 0..num_groups {
                    biases[k * num_groups + g] = ((g * 3 % 19) as f32) * 0.125;
                }
            }
        } else {
            for j in 0..shape.output_dim {
                for g in 0..num_groups {
                    let base_val = j * 7 + g * 3;
                    biases[j * num_groups + g] = ((base_val % 19) as f32) * 0.125;
                }
            }
        }
    }

    QuantParams { scales_f32: scales, biases_f32: biases, zero_points, zero_points_stride: zp_stride }
}

// --- CPU reference ---

fn get_4bit_value(data: &[u8], index: usize) -> f32 {
    let word_idx = index / 4;
    let word_offset = index % 4;
    let byte_idx = word_idx * 2;
    let word = if byte_idx + 1 < data.len() {
        data[byte_idx] as u16 | ((data[byte_idx + 1] as u16) << 8)
    } else {
        0
    };
    ((word >> (word_offset * 4)) & 0x0F) as f32
}

fn cpu_reference(
    config: &QuantConfig,
    shape: &TestShape,
    x_f32: &[f32],
    b_quant: &[u8],
    params: &QuantParams,
) -> Vec<f32> {
    let num_groups = if !config.weights_transposed {
        (shape.output_dim + GROUP_SIZE - 1) / GROUP_SIZE
    } else {
        (shape.input_dim + GROUP_SIZE - 1) / GROUP_SIZE
    };

    let mut y = vec![0.0f32; shape.batch * shape.output_dim];
    for i in 0..shape.batch {
        for j in 0..shape.output_dim {
            let mut acc = 0.0f32;
            if !config.weights_transposed {
                let group_idx = j / GROUP_SIZE;
                for l in 0..shape.input_dim {
                    let weight_idx = l * shape.output_dim + j;
                    let val_q = if config.bits == 4 { get_4bit_value(b_quant, weight_idx) } else { b_quant[weight_idx] as f32 };
                    let val_a = x_f32[i * shape.input_dim + l];
                    let scale = params.scales_f32[l * num_groups + group_idx];
                    let bias = if config.quant_type == QuantizedMatmulType::ZeroPoint {
                        let zp = get_zp_value(&params.zero_points, params.zero_points_stride, l, group_idx, config.bits);
                        -scale * zp
                    } else {
                        params.biases_f32[l * num_groups + group_idx]
                    };
                    acc += val_a * (scale * val_q + bias);
                }
            } else {
                for g in 0..num_groups {
                    let scale = params.scales_f32[j * num_groups + g];
                    let bias = if config.quant_type == QuantizedMatmulType::ZeroPoint {
                        let zp = get_zp_value(&params.zero_points, params.zero_points_stride, j, g, config.bits);
                        -scale * zp
                    } else {
                        params.biases_f32[j * num_groups + g]
                    };
                    let l_start = g * GROUP_SIZE;
                    let l_end = (l_start + GROUP_SIZE).min(shape.input_dim);
                    let mut group_acc = 0.0f32;
                    let mut group_sum = 0.0f32;
                    for l in l_start..l_end {
                        let weight_idx = j * shape.input_dim + l;
                        let val_q = if config.bits == 4 { get_4bit_value(b_quant, weight_idx) } else { b_quant[weight_idx] as f32 };
                        let val_a = x_f32[i * shape.input_dim + l];
                        group_acc += val_a * val_q;
                        group_sum += val_a;
                    }
                    acc += scale * group_acc + bias * group_sum;
                }
            }
            y[i * shape.output_dim + j] = acc;
        }
    }
    y
}

// --- Metal execution ---

fn buffer_from_f32(
    ctx: &<Metal as Backend>::Context,
    values: &[f32],
) -> Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>> {
    ctx.device
        .new_buffer_with_data(bytemuck::cast_slice(values), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("buffer creation")
}

fn run_quant_matmul_case(
    ctx: &<Metal as Backend>::Context,
    config: &QuantConfig,
    shape: &TestShape,
) -> TestResult {
    let weights_quant = create_test_weights(shape.output_dim, shape.input_dim, config.weights_transposed, config.bits);
    let weights_packed = if config.bits == 4 { pack_u4_weights(&weights_quant) } else { weights_quant.clone() };
    let params = generate_quant_params(config, shape);

    let x_f32: Vec<f32> = (0..shape.batch * shape.input_dim)
        .map(|i| (i % shape.input_dim + 1) as f32 / shape.input_dim as f32)
        .collect();

    let w_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&weights_packed), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("w_buf");
    let s_buf = buffer_from_f32(ctx, &params.scales_f32);
    let zp_or_bias_buf = match config.quant_type {
        QuantizedMatmulType::ZeroPoint => ctx
            .device
            .new_buffer_with_data(&params.zero_points, MTLResourceOptions::STORAGE_MODE_SHARED)
            .expect("zp_buf"),
        QuantizedMatmulType::Mlx => buffer_from_f32(ctx, &params.biases_f32),
    };
    let x_buf = buffer_from_f32(ctx, &x_f32);
    let y_buf = ctx.create_buffer(shape.batch * shape.output_dim * DataType::F32.size_in_bytes()).expect("y_buf");

    let kernel = QuantizedMatmulKernelEncodable::<Metal>::new(
        ctx,
        QuantizedMatmulConfiguration {
            data_type: DataType::F32,
            group_size: GROUP_SIZE,
            input_dim: shape.input_dim,
            output_dim: shape.output_dim,
            mode: match config.bits {
                4 => QuantizationMode::UInt4,
                8 => QuantizationMode::Int8,
                _ => unreachable!(),
            },
            quantization_type: config.quant_type,
            weights_transposed: config.weights_transposed,
        },
    )
    .expect("kernel creation");

    let args = QuantizedMatmulArguments {
        a_buffer: &x_buf,
        a_offset: 0,
        b_buffer: &w_buf,
        scales_buffer: &s_buf,
        zero_points_or_biases_buffer: &zp_or_bias_buf,
        output_buffer: &y_buf,
        batch: shape.batch,
        input_dim: shape.input_dim,
        output_dim: shape.output_dim,
        quantization_type: config.quant_type,
    };
    let cb = ctx.command_queue.command_buffer().expect("cb").to_owned();
    let enc = cb.new_compute_command_encoder().expect("encoder");
    kernel.encode(&enc, args).expect("encode");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let y_expected = cpu_reference(config, shape, &x_f32, &weights_packed, &params);
    let y_out: Vec<f32> = unsafe {
        let ptr = y_buf.contents().as_ptr() as *const f32;
        std::slice::from_raw_parts(ptr, shape.batch * shape.output_dim).to_vec()
    };

    let rel_tol = 0.002f64;
    let abs_tol = 0.1f64;
    let mismatch_count = y_expected
        .iter()
        .zip(y_out.iter())
        .filter(|&(exp, got)| {
            let diff = (*exp as f64 - *got as f64).abs();
            let tol = abs_tol.max((*exp as f64).abs() * rel_tol);
            diff > tol
        })
        .count();

    TestResult {
        config: format!("{}", config),
        shape: format!("{}", shape),
        dispatch_path: quant_dispatch_path(shape.batch, shape.output_dim).to_owned(),
        passed: mismatch_count == 0,
        mismatch_count,
        total_outputs: shape.batch * shape.output_dim,
    }
}

fn print_results_table(results: &[TestResult]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Config", "Shape (BxKxN)", "Dispatch", "Status", "Mismatches"]);

    for r in results {
        table.add_row(vec![
            r.config.clone(),
            r.shape.clone(),
            r.dispatch_path.clone(),
            if r.passed { "PASS".into() } else { "FAIL".into() },
            format!("{}/{}", r.mismatch_count, r.total_outputs),
        ]);
    }

    println!("{table}");
}

#[test]
fn quant_matmul_correctness() {
    let ctx = <Metal as Backend>::Context::new().expect("Metal context required");

    let configs = test_configs();
    let shapes = test_shapes();

    let valid_cases: Vec<_> = configs
        .iter()
        .flat_map(|c| shapes.iter().filter(|s| shape_valid_for_config(s, c)).map(move |s| (c, s)))
        .collect();

    let total = valid_cases.len();
    eprintln!("Quant matmul correctness: {} configs x shapes (filtered) = {} cases", configs.len(), total);

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} [{elapsed_precise}] {msg}").expect("progress style"),
    );

    let mut results = Vec::with_capacity(total);

    for (config, shape) in &valid_cases {
        pb.set_message(format!("{} {}", config, shape));
        let result = run_quant_matmul_case(&ctx, config, shape);
        results.push(result);
        pb.inc(1);
    }

    pb.finish_with_message("done");
    print_results_table(&results);
    write_json_results("quant_matmul_correctness", &ctx.device.name(), ctx.is_mpp_available(), &results);

    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    if !failures.is_empty() {
        eprintln!("\n{} / {} cases failed:", failures.len(), results.len());
        for f in &failures {
            eprintln!("  {} {} [{}] mismatches={}/{}", f.config, f.shape, f.dispatch_path, f.mismatch_count, f.total_outputs);
        }
        panic!("{} quant matmul correctness cases failed", failures.len());
    }
}
