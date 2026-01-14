//! Matmul benchmark comparing MLX, UZU, and MPSGraph backends
//!
//! Run with: cargo run --release -p matmul-benchmark
use std::{
    collections::HashMap,
    fmt,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use half::{bf16, f16};
use matmul_benchmark::{
    benchmark_matmul_bf16 as mlx_bf16, benchmark_matmul_f16 as mlx_f16,
    benchmark_matmul_f32 as mlx_f32, matmul_bf16_with_output,
    matmul_f16_with_output, matmul_f32_with_output,
};
use metal::{Device, MTLResourceOptions};
use mpsgraph::{
    CommandBuffer, Device as MPSDevice, ExecutableExecutionDescriptor, Graph,
    ShapedType, TensorData,
};
use objc2::rc::autoreleasepool;
use serde::Deserialize;
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::{MatmulArguments, MatmulKernel},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DType {
    F16,
    F32,
    BF16,
}

impl fmt::Display for DType {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            DType::F16 => write!(f, " f16"),
            DType::F32 => write!(f, " f32"),
            DType::BF16 => write!(f, "bf16"),
        }
    }
}

impl DType {
    fn to_uzu(self) -> DataType {
        match self {
            DType::F16 => DataType::F16,
            DType::F32 => DataType::F32,
            DType::BF16 => DataType::BF16,
        }
    }

    fn to_mps(self) -> mpsgraph::DataType {
        match self {
            DType::F16 => mpsgraph::DataType::Float16,
            DType::F32 => mpsgraph::DataType::Float32,
            DType::BF16 => mpsgraph::DataType::BFloat16,
        }
    }

    fn element_size(self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    mlx_ms: f64,
    uzu_ms: f64,
    mps_ms: f64,
}

#[derive(Debug, Clone)]
struct AccuracyResult {
    max_abs_error: f32,
    mean_abs_error: f32,
    max_rel_error: f32,
}

#[derive(Debug, Clone)]
struct BenchCase {
    name: String,
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
}

impl BenchCase {
    fn b_bytes(&self) -> usize {
        self.n * self.k * self.dtype.element_size()
    }
}

// ============================================================================
// LLM SHAPES (derived from performance-benchmarks model.safetensors)
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct ModelConfigFile {
    repo: String,
    model_config: ModelConfigOuter,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelConfigOuter {
    model_config: ModelConfigInner,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelConfigInner {
    embedding_config: EmbeddingConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
enum EmbeddingConfig {
    TiedEmbeddingConfig {
        precision: String,
    },
    MLXQuantizedTiedEmbeddingConfig {
        activation_precision: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
struct SafetensorsTensorInfo {
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct PerfBenchRun {
    task: PerfBenchTask,
    tokens_count_input: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct PerfBenchTask {
    repo_id: String,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn model_code(model_dir_name: &str) -> String {
    if model_dir_name.starts_with("Llama-3.2-1B") {
        "L3.2-1B".to_string()
    } else if model_dir_name.starts_with("Qwen3-0.6B") {
        "Q3-0.6B".to_string()
    } else if model_dir_name.starts_with("Qwen3-4B-AWQ") {
        "Q3-4B-AWQ".to_string()
    } else if model_dir_name.starts_with("Qwen3-4B-MLX-4bit") {
        "Q3-4B-4b".to_string()
    } else {
        // Best-effort fallback: keep within table width (<= 9 chars)
        model_dir_name.chars().take(9).collect()
    }
}

fn dtype_from_precision_string(precision: &str) -> Option<DType> {
    match precision {
        "float16" => Some(DType::F16),
        "bfloat16" => Some(DType::BF16),
        "float32" => Some(DType::F32),
        _ => None,
    }
}

fn model_activation_dtype(cfg: &ModelConfigFile) -> Option<DType> {
    match &cfg.model_config.model_config.embedding_config {
        EmbeddingConfig::TiedEmbeddingConfig {
            precision,
        } => dtype_from_precision_string(precision),
        EmbeddingConfig::MLXQuantizedTiedEmbeddingConfig {
            activation_precision,
        } => dtype_from_precision_string(activation_precision),
        EmbeddingConfig::Unknown => None,
    }
}

fn read_safetensors_header(
    path: &Path
) -> Result<HashMap<String, SafetensorsTensorInfo>, Box<dyn std::error::Error>>
{
    let mut file = File::open(path)?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;

    let raw: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header_bytes)?;

    raw.into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, v)| {
            let info: SafetensorsTensorInfo = serde_json::from_value(v)?;
            Ok((k, info))
        })
        .collect()
}

fn weight_dims_nk(
    header: &HashMap<String, SafetensorsTensorInfo>,
    key: &str,
) -> Option<(usize, usize)> {
    let info = header.get(key)?;
    let (n, packed_k) = match info.shape.as_slice() {
        [n, k] => (*n, *k),
        _ => return None,
    };

    let k = match info.dtype.as_str() {
        // 4-bit packed (2 weights per byte)
        "U8" => packed_k.checked_mul(2)?,
        _ => packed_k,
    };

    Some((n, k))
}

fn collect_json_files(
    dir: &Path,
    out: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "json") {
            out.push(path);
        }
    }
    Ok(())
}

fn prefill_tokens_from_perf_results(
    json_files: &[PathBuf],
    repo_id: &str,
) -> Option<usize> {
    for path in json_files {
        let file = File::open(path).ok()?;
        let runs: Vec<PerfBenchRun> = serde_json::from_reader(file).ok()?;
        for run in runs {
            if run.task.repo_id == repo_id {
                return Some(run.tokens_count_input);
            }
        }
    }
    None
}

fn default_benchmark_cases() -> Vec<BenchCase> {
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("GEMV 4K", 1, 4096, 4096),
        ("GEMV 8K", 1, 8192, 8192),
        ("Batch 4", 4, 4096, 4096),
        ("Batch 16", 16, 4096, 4096),
        ("Prefill 128", 128, 4096, 4096),
        ("Prefill 512", 512, 4096, 4096),
        ("Square 1K", 1024, 1024, 1024),
        ("Square 2K", 2048, 2048, 2048),
    ];

    let dtypes = [DType::F16, DType::BF16, DType::F32];
    shapes
        .iter()
        .flat_map(|(name, m, k, n)| {
            dtypes.iter().map(|dtype| BenchCase {
                name: (*name).to_string(),
                m: *m,
                k: *k,
                n: *n,
                dtype: *dtype,
            })
        })
        .collect()
}

fn llm_benchmark_cases() -> Vec<BenchCase> {
    match try_llm_benchmark_cases() {
        Ok(cases) if !cases.is_empty() => cases,
        _ => default_benchmark_cases(),
    }
}

fn try_llm_benchmark_cases()
-> Result<Vec<BenchCase>, Box<dyn std::error::Error>> {
    let root = workspace_root();
    let models_dir =
        root.join("external/performance-benchmarks/workspace/models/0.1.7");
    if !models_dir.is_dir() {
        return Ok(Vec::new());
    }

    let results_dir =
        root.join("external/performance-benchmarks/workspace/results");
    let json_files = if results_dir.is_dir() {
        let mut files = Vec::new();
        collect_json_files(&results_dir, &mut files)?;
        files
    } else {
        Vec::new()
    };

    let mut model_dirs: Vec<PathBuf> = fs::read_dir(&models_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    model_dirs.sort();

    let ops: [(&str, &str); 4] = [
        ("QKV", "transformer.layers.0.mixer.qkv_projection.weights"),
        ("O", "transformer.layers.0.mixer.out_projection.weights"),
        ("MLPu", "transformer.layers.0.mlp.up_projection.weights"),
        ("MLPd", "transformer.layers.0.mlp.down_projection.weights"),
    ];

    let mut cases = Vec::new();
    for model_dir in model_dirs {
        let dir_name =
            model_dir.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        let code = model_code(dir_name);

        let cfg: ModelConfigFile = {
            let file = File::open(model_dir.join("config.json"))?;
            serde_json::from_reader(file)?
        };

        let dtype = model_activation_dtype(&cfg).unwrap_or(DType::BF16);

        let prefill_m =
            prefill_tokens_from_perf_results(&json_files, &cfg.repo)
                .unwrap_or(1024);
        let ms = if prefill_m == 1 {
            vec![1usize]
        } else {
            vec![1usize, prefill_m]
        };

        let header =
            read_safetensors_header(&model_dir.join("model.safetensors"))?;

        for (op_code, key) in ops {
            let Some((n, k)) = weight_dims_nk(&header, key) else {
                continue;
            };

            for m in &ms {
                cases.push(BenchCase {
                    name: format!("{code} {op_code}"),
                    m: *m,
                    k,
                    n,
                    dtype,
                });
            }
        }
    }

    Ok(cases)
}

fn compute_gflops(
    m: usize,
    k: usize,
    n: usize,
    time_ms: f64,
) -> f64 {
    let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
    flops / (time_ms * 1e6)
}

/// Generate deterministic test data
fn generate_test_data_f16(
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f16>, Vec<f16>) {
    let a: Vec<f16> =
        (0..(m * k)).map(|i| f16::from_f32(((i % 13) as f32) * 0.01)).collect();
    let b: Vec<f16> = (0..(n * k))
        .map(|i| f16::from_f32(((i % 17) as f32) * 0.02 - 0.1))
        .collect();
    (a, b)
}

fn generate_test_data_f32(
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i % 13) as f32) * 0.01).collect();
    let b: Vec<f32> =
        (0..(n * k)).map(|i| ((i % 17) as f32) * 0.02 - 0.1).collect();
    (a, b)
}

fn generate_test_data_bf16(
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let a: Vec<bf16> = (0..(m * k))
        .map(|i| bf16::from_f32(((i % 13) as f32) * 0.01))
        .collect();
    let b: Vec<bf16> = (0..(n * k))
        .map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.1))
        .collect();
    (a, b)
}

fn benchmark_uzu(
    ctx: &MTLContext,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> f64 {
    let elem_size = dtype.element_size();
    let a_bytes = m * k * elem_size;
    let b_bytes = n * k * elem_size;
    let d_bytes = m * n * elem_size;

    let a_buf = ctx
        .device
        .new_buffer(a_bytes as u64, MTLResourceOptions::StorageModeShared);
    let b_buf = ctx
        .device
        .new_buffer(b_bytes as u64, MTLResourceOptions::StorageModeShared);
    let d_buf = ctx
        .device
        .new_buffer(d_bytes as u64, MTLResourceOptions::StorageModeShared);

    // Initialize with test data
    match dtype {
        DType::F16 => {
            let (a, b) = generate_test_data_f16(m, k, n);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    a.as_ptr(),
                    a_buf.contents() as *mut f16,
                    a.len(),
                );
                std::ptr::copy_nonoverlapping(
                    b.as_ptr(),
                    b_buf.contents() as *mut f16,
                    b.len(),
                );
            }
        },
        DType::F32 => {
            let (a, b) = generate_test_data_f32(m, k, n);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    a.as_ptr(),
                    a_buf.contents() as *mut f32,
                    a.len(),
                );
                std::ptr::copy_nonoverlapping(
                    b.as_ptr(),
                    b_buf.contents() as *mut f32,
                    b.len(),
                );
            }
        },
        DType::BF16 => {
            let (a, b) = generate_test_data_bf16(m, k, n);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    a.as_ptr(),
                    a_buf.contents() as *mut bf16,
                    a.len(),
                );
                std::ptr::copy_nonoverlapping(
                    b.as_ptr(),
                    b_buf.contents() as *mut bf16,
                    b.len(),
                );
            }
        },
    }

    let mut kernel = MatmulKernel::new(ctx, dtype.to_uzu(), false, true)
        .expect("kernel new");

    for _ in 0..warmup_iters {
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
                    ldb: k as i32,
                    ldd: n as i32,
                    batch_count: 1,
                    alpha: 1.0,
                    beta: 0.0,
                },
                None,
            )
            .expect("encode");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let start = Instant::now();
    for _ in 0..bench_iters {
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
                    ldb: k as i32,
                    ldd: n as i32,
                    batch_count: 1,
                    alpha: 1.0,
                    beta: 0.0,
                },
                None,
            )
            .expect("encode");
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let elapsed = start.elapsed();
    (elapsed.as_secs_f64() * 1000.0) / bench_iters as f64
}

fn benchmark_mpsgraph(
    device: &metal::Device,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> f64 {
    autoreleasepool(|_| {
        let graph = Graph::new();
        let mps_dtype = dtype.to_mps();

        let a_shape = [m as isize, k as isize];
        let b_shape = [k as isize, n as isize];

        let a_placeholder =
            graph.placeholder(Some(&a_shape), mps_dtype, Some("a"));
        let b_placeholder =
            graph.placeholder(Some(&b_shape), mps_dtype, Some("b"));

        let result =
            graph.matrix_multiplication(&a_placeholder, &b_placeholder, None);

        let a_shaped_type =
            ShapedType::new_with_shape_data_type(Some(&a_shape), mps_dtype);
        let b_shaped_type =
            ShapedType::new_with_shape_data_type(Some(&b_shape), mps_dtype);

        let feeds = HashMap::from([
            (&*a_placeholder, &*a_shaped_type),
            (&*b_placeholder, &*b_shaped_type),
        ]);

        let mps_device = MPSDevice::with_device(device);
        let executable =
            graph.compile(&mps_device, &feeds, &[&result], None, None);

        let execution_descriptor = ExecutableExecutionDescriptor::new();

        let elem_size = dtype.element_size();
        let a_buf = device.new_buffer(
            (m * k * elem_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = device.new_buffer(
            (k * n * elem_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (m * n * elem_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_queue = device.new_command_queue();

        let a_tensor_data =
            TensorData::new_with_mtl_buffer(&a_buf, &[m, k], mps_dtype, None);
        let b_tensor_data =
            TensorData::new_with_mtl_buffer(&b_buf, &[k, n], mps_dtype, None);
        let output_tensor_data = TensorData::new_with_mtl_buffer(
            &output_buf,
            &[m, n],
            mps_dtype,
            None,
        );

        let inputs: &[&TensorData] = &[&a_tensor_data, &b_tensor_data];
        let outputs: &[&TensorData] = &[&output_tensor_data];

        execution_descriptor.set_enable_commit_and_continue(true);
        for _ in 0..warmup_iters {
            let command_buffer =
                CommandBuffer::from_command_queue(&command_queue);
            let root_cb = command_buffer.root_command_buffer().to_owned();
            let _ = executable.encode_to_command_buffer(
                &command_buffer,
                inputs,
                Some(outputs),
                Some(&execution_descriptor),
            );
            command_buffer.commit_and_continue();
            root_cb.wait_until_completed();
        }

        let start = Instant::now();
        for _ in 0..bench_iters {
            let command_buffer =
                CommandBuffer::from_command_queue(&command_queue);
            let root_cb = command_buffer.root_command_buffer().to_owned();
            let _ = executable.encode_to_command_buffer(
                &command_buffer,
                inputs,
                Some(outputs),
                Some(&execution_descriptor),
            );
            command_buffer.commit_and_continue();
            root_cb.wait_until_completed();
        }
        let elapsed = start.elapsed();
        (elapsed.as_secs_f64() * 1000.0) / bench_iters as f64
    })
}

fn benchmark_mlx(
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    warmup: i32,
    iters: i32,
) -> f64 {
    match dtype {
        DType::F16 => {
            mlx_f16(m as i32, n as i32, k as i32, warmup, iters).avg_time_ms
        },
        DType::F32 => {
            mlx_f32(m as i32, n as i32, k as i32, warmup, iters).avg_time_ms
        },
        DType::BF16 => {
            mlx_bf16(m as i32, n as i32, k as i32, warmup, iters).avg_time_ms
        },
    }
}

fn run_benchmark(
    ctx: &MTLContext,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iters: usize,
) -> BenchmarkResult {
    let mlx_ms = benchmark_mlx(dtype, m, k, n, warmup as i32, iters as i32);
    let uzu_ms = benchmark_uzu(ctx, dtype, m, k, n, warmup, iters);
    let mps_ms = benchmark_mpsgraph(&ctx.device, dtype, m, k, n, warmup, iters);

    BenchmarkResult {
        mlx_ms,
        uzu_ms,
        mps_ms,
    }
}

fn format_speedup(
    base: f64,
    other: f64,
) -> String {
    let ratio = other / base;
    format!("{:.2}x", ratio)
}

/// Run accuracy test for a given shape and dtype
fn run_accuracy_test(
    ctx: &MTLContext,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) -> AccuracyResult {
    match dtype {
        DType::F16 => run_accuracy_test_f16(ctx, m, k, n),
        DType::F32 => run_accuracy_test_f32(ctx, m, k, n),
        DType::BF16 => run_accuracy_test_bf16(ctx, m, k, n),
    }
}

fn run_accuracy_test_f16(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    n: usize,
) -> AccuracyResult {
    let (a, b) = generate_test_data_f16(m, k, n);

    // MLX
    let a_u16: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
    let b_u16: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();
    let mut mlx_out_u16 = vec![0u16; m * n];
    matmul_f16_with_output(
        &a_u16,
        &b_u16,
        &mut mlx_out_u16,
        m as i32,
        k as i32,
        n as i32,
    );
    let mlx_out: Vec<f16> =
        mlx_out_u16.iter().map(|&x| f16::from_bits(x)).collect();

    // UZU
    let a_buf = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx
        .device
        .new_buffer((m * n * 2) as u64, MTLResourceOptions::StorageModeShared);

    let mut kernel =
        MatmulKernel::new(ctx, DataType::F16, false, true).unwrap();
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
                ldb: k as i32,
                ldd: n as i32,
                batch_count: 1,
                alpha: 1.0,
                beta: 0.0,
            },
            None,
        )
        .unwrap();
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let uzu_out: Vec<f16> = unsafe {
        std::slice::from_raw_parts(d_buf.contents() as *const f16, m * n)
            .to_vec()
    };

    compare_outputs_f16(&mlx_out, &uzu_out)
}

fn run_accuracy_test_f32(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    n: usize,
) -> AccuracyResult {
    let (a, b) = generate_test_data_f32(m, k, n);

    // MLX
    let mut mlx_out = vec![0f32; m * n];
    matmul_f32_with_output(&a, &b, &mut mlx_out, m as i32, k as i32, n as i32);

    // UZU
    let a_buf = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx
        .device
        .new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

    let mut kernel =
        MatmulKernel::new(ctx, DataType::F32, false, true).unwrap();
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
                ldb: k as i32,
                ldd: n as i32,
                batch_count: 1,
                alpha: 1.0,
                beta: 0.0,
            },
            None,
        )
        .unwrap();
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let uzu_out: Vec<f32> = unsafe {
        std::slice::from_raw_parts(d_buf.contents() as *const f32, m * n)
            .to_vec()
    };

    compare_outputs_f32(&mlx_out, &uzu_out)
}

fn run_accuracy_test_bf16(
    ctx: &MTLContext,
    m: usize,
    k: usize,
    n: usize,
) -> AccuracyResult {
    let (a, b) = generate_test_data_bf16(m, k, n);

    // MLX
    let a_u16: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
    let b_u16: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();
    let mut mlx_out_u16 = vec![0u16; m * n];
    matmul_bf16_with_output(
        &a_u16,
        &b_u16,
        &mut mlx_out_u16,
        m as i32,
        k as i32,
        n as i32,
    );
    let mlx_out: Vec<bf16> =
        mlx_out_u16.iter().map(|&x| bf16::from_bits(x)).collect();

    // UZU
    let a_buf = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx
        .device
        .new_buffer((m * n * 2) as u64, MTLResourceOptions::StorageModeShared);

    let mut kernel =
        MatmulKernel::new(ctx, DataType::BF16, false, true).unwrap();
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
                ldb: k as i32,
                ldd: n as i32,
                batch_count: 1,
                alpha: 1.0,
                beta: 0.0,
            },
            None,
        )
        .unwrap();
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let uzu_out: Vec<bf16> = unsafe {
        std::slice::from_raw_parts(d_buf.contents() as *const bf16, m * n)
            .to_vec()
    };

    compare_outputs_bf16(&mlx_out, &uzu_out)
}

fn compare_outputs_f16(
    reference: &[f16],
    test: &[f16],
) -> AccuracyResult {
    let mut max_abs_error: f32 = 0.0;
    let mut sum_abs_error: f32 = 0.0;
    let mut max_rel_error: f32 = 0.0;

    for (r, t) in reference.iter().zip(test.iter()) {
        let r_f32 = r.to_f32();
        let t_f32 = t.to_f32();
        let abs_error = (r_f32 - t_f32).abs();
        sum_abs_error += abs_error;
        max_abs_error = max_abs_error.max(abs_error);
        if r_f32.abs() > 1e-6 {
            max_rel_error = max_rel_error.max(abs_error / r_f32.abs());
        }
    }

    AccuracyResult {
        max_abs_error,
        mean_abs_error: sum_abs_error / reference.len() as f32,
        max_rel_error,
    }
}

fn compare_outputs_f32(
    reference: &[f32],
    test: &[f32],
) -> AccuracyResult {
    let mut max_abs_error: f32 = 0.0;
    let mut sum_abs_error: f32 = 0.0;
    let mut max_rel_error: f32 = 0.0;

    for (r, t) in reference.iter().zip(test.iter()) {
        let abs_error = (r - t).abs();
        sum_abs_error += abs_error;
        max_abs_error = max_abs_error.max(abs_error);
        if r.abs() > 1e-6 {
            max_rel_error = max_rel_error.max(abs_error / r.abs());
        }
    }

    AccuracyResult {
        max_abs_error,
        mean_abs_error: sum_abs_error / reference.len() as f32,
        max_rel_error,
    }
}

fn compare_outputs_bf16(
    reference: &[bf16],
    test: &[bf16],
) -> AccuracyResult {
    let mut max_abs_error: f32 = 0.0;
    let mut sum_abs_error: f32 = 0.0;
    let mut max_rel_error: f32 = 0.0;

    for (r, t) in reference.iter().zip(test.iter()) {
        let r_f32 = r.to_f32();
        let t_f32 = t.to_f32();
        let abs_error = (r_f32 - t_f32).abs();
        sum_abs_error += abs_error;
        max_abs_error = max_abs_error.max(abs_error);
        if r_f32.abs() > 1e-6 {
            max_rel_error = max_rel_error.max(abs_error / r_f32.abs());
        }
    }

    AccuracyResult {
        max_abs_error,
        mean_abs_error: sum_abs_error / reference.len() as f32,
        max_rel_error,
    }
}

fn main() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let ctx = MTLContext::new(device.clone(), command_queue)
        .expect("Failed to create MTLContext");

    let cases = llm_benchmark_cases();

    // =========================================================================
    // ACCURACY TESTS
    // =========================================================================
    println!();
    println!(
        "╔═══════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                     ACCURACY TESTS: UZU vs MLX                                ║"
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════╣"
    );
    println!("║  Device: {:<69}║", device.name());
    println!(
        "╚═══════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();

    println!(
        "┌────────────────┬───────┬───────┬───────┬──────┬───────────┬───────────┬────────────┐"
    );
    println!(
        "│ Shape          │   M   │   K   │   N   │ Type │  Max Abs  │ Mean Abs  │  Max Rel   │"
    );
    println!(
        "├────────────────┼───────┼───────┼───────┼──────┼───────────┼───────────┼────────────┤"
    );

    let mut all_passed = true;
    let tolerance = 0.01;

    // Keep accuracy checks reasonably fast: only small(ish) B buffers and M=1
    let accuracy_cases: Vec<&BenchCase> = cases
        .iter()
        .filter(|c| c.m == 1 && c.b_bytes() <= 32 * 1024 * 1024)
        .collect();

    for case in &accuracy_cases {
        let result =
            run_accuracy_test(&ctx, case.dtype, case.m, case.k, case.n);
        let passed = result.max_rel_error < tolerance;
        if !passed {
            all_passed = false;
        }

        println!(
            "│ {:<14} │ {:>5} │ {:>5} │ {:>5} │ {:>4} │ {:>9.2e} │ {:>9.2e} │ {:>9.2e}  │",
            case.name,
            case.m,
            case.k,
            case.n,
            case.dtype,
            result.max_abs_error,
            result.mean_abs_error,
            result.max_rel_error
        );
    }

    println!(
        "└────────────────┴───────┴───────┴───────┴──────┴───────────┴───────────┴────────────┘"
    );
    println!();

    if all_passed {
        println!(
            "✓ All accuracy tests PASSED (max relative error < {:.0}%)",
            tolerance * 100.0
        );
    } else {
        println!(
            "✗ Some accuracy tests FAILED (max relative error >= {:.0}%)",
            tolerance * 100.0
        );
    }
    println!();

    // =========================================================================
    // PERFORMANCE BENCHMARKS
    // =========================================================================
    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║                              MATMUL BENCHMARK: MLX vs UZU vs MPSGraph                                            ║"
    );
    println!(
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!("║  Device: {:<103}║", device.name());
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();

    let warmup = 10;
    let iters = 100;

    println!(
        "┌────────────────┬───────┬───────┬───────┬──────┬────────────────────────────┬────────────────────────────┬─────────────────────┐"
    );
    println!(
        "│                │       │       │       │      │       Time (ms)            │       GFLOPS               │  UZU Speedup vs     │"
    );
    println!(
        "│ Shape          │   M   │   K   │   N   │ Type ├────────┬────────┬──────────┼────────┬────────┬──────────┼──────────┬──────────┤"
    );
    println!(
        "│                │       │       │       │      │   MLX  │   UZU  │   MPS    │   MLX  │   UZU  │   MPS    │   MLX    │   MPS    │"
    );
    println!(
        "├────────────────┼───────┼───────┼───────┼──────┼────────┼────────┼──────────┼────────┼────────┼──────────┼──────────┼──────────┤"
    );

    for case in &cases {
        let result = run_benchmark(
            &ctx, case.dtype, case.m, case.k, case.n, warmup, iters,
        );

        let mlx_gflops = compute_gflops(case.m, case.k, case.n, result.mlx_ms);
        let uzu_gflops = compute_gflops(case.m, case.k, case.n, result.uzu_ms);
        let mps_gflops = compute_gflops(case.m, case.k, case.n, result.mps_ms);

        let uzu_vs_mlx = format_speedup(result.uzu_ms, result.mlx_ms);
        let uzu_vs_mps = format_speedup(result.uzu_ms, result.mps_ms);

        println!(
            "│ {:<14} │ {:>5} │ {:>5} │ {:>5} │ {:>4} │ {:>6.3} │ {:>6.3} │ {:>8.3} │ {:>6.0} │ {:>6.0} │ {:>8.0} │ {:>8} │ {:>8} │",
            case.name,
            case.m,
            case.k,
            case.n,
            case.dtype,
            result.mlx_ms,
            result.uzu_ms,
            result.mps_ms,
            mlx_gflops,
            uzu_gflops,
            mps_gflops,
            uzu_vs_mlx,
            uzu_vs_mps
        );
    }

    println!(
        "└────────────────┴───────┴───────┴───────┴──────┴────────┴────────┴──────────┴────────┴────────┴──────────┴──────────┴──────────┘"
    );
    println!();
    println!("Legend:");
    println!("  - UZU Speedup > 1.0x means UZU is faster");
    println!("  - UZU Speedup < 1.0x means the other backend is faster");
    println!();
}
