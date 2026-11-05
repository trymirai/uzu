//! Autotune runner: forced GEMV specializations / GEMM plans x cells -> audit + timing as JSON.
//!
//! ```text
//! AT_JOB=/abs/job.json AT_OUT=/abs/result.json \
//!   cargo test --release -p uzu-engine autotune_runner -- --ignored --nocapture
//! ```
//!
//! Schema: `~/playground/autotune/autotune/runner_schema.md`. Driven by `at run`.
//!
//! P1/T1 additions (all inside this test harness, shipped uzu code is untouched): batch sizes up to `MAX_M` (GEMM cells at
//! batch 1024), `timing.protocols` (`pipelined` / `chain`, see `encode_slot`), a multi-threaded f32 CPU reference that checks
//! sampled or all rows (`audit_rows`), and a real-GPU wrong-answer fault (`"fault": "wrong_scale"`).
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use futures::executor::block_on;
use half::bf16;
use metal::{MTLDevice, MTLGPUFamily};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde_json::{Map, Value, json};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context,
            gpu_types::{
                gemm::{GemmBPrologueKind, GemmDTransform},
                quantization::QuantizationMode,
                quantization_method::QuantizationMethod,
            },
            kernel::{
                ActivationTransform,
                matmul::{
                    ActivationFormat, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput,
                    MatmulShape, QuantParams, QuantParamsLayout, packed_half_bytes,
                },
            },
        },
        metal::{
            Metal,
            command_buffer::{MetalCommandBufferEncoding, MetalCommandBufferPending},
            context::MetalContext,
            gpu_types::gemm::{GemmBLayout, GemmTiling},
            kernel::{
                GemvMetalKernel,
                matmul::{
                    MatmulMetalError, MatmulMetalKernel, MatmulOutputWork,
                    gemm::{
                        GemmEngine, GemmKernel, GemmPlan, GemmProblem,
                        small_batch_units::{Offsets, pack_small_batch_units},
                    },
                    gemv::{GemvKernel, GemvSpecialization, GemvTile},
                    matmul_dispatch::MatmulDispatch,
                },
            },
        },
    },
    data_type::DataType,
    tests::{
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        matmul::{QuantBuffers, QuantInput},
    },
};

/// Newest Apple GPU family the device supports (main no longer keeps it on `MetalContext`; `at report` groups chips by it).
fn newest_apple_gpu_family(context: &MetalContext) -> String {
    [MTLGPUFamily::Apple7, MTLGPUFamily::Apple8, MTLGPUFamily::Apple9, MTLGPUFamily::Apple10]
        .into_iter()
        .rev()
        .find(|family| context.device.supports_family(*family))
        .map_or_else(|| "?".to_string(), |family| format!("{family:?}"))
}

type Gb = <Metal as Backend>::GlobalGpuBuffer;
type Args<'a> = MatmulArguments<'a, Metal, &'a Gb, &'a Gb, &'a mut Gb, &'a Gb>;

const SENTINEL_BITS: u16 = 0x7FA5;
/// Max bf16 ULP error, measured at `ulp(max(|ref|, row_rms / 2))`, for a GEMV candidate to pass the strict audit.
/// A flipped last bit is 1.0; the floor keeps near-zero outputs of a deferred RHT (32 pre-rounded inputs) below it.
const ULP_PASS: f32 = 1.001;
/// Max |kernel - f32 CPU| / max(|cpu|, block_rms) for GEMV candidates (production GEMV measures 0.006-0.007).
const CPU_TOLERANCE: f32 = 0.012;
/// Same check for GEMM kernels, which round the dequantized weight to bf16 for the MMA (production MXU measures 0.015).
const CPU_TOLERANCE_MMA: f32 = 0.03;
/// Largest batch (rows of x) a cell may have. GEMV cells use 1..=8, GEMM cells 9..=MAX_M.
const MAX_M: u32 = 4096;
/// The legacy per-row GEMV reference (one command buffer per row) is only worth its cost up to this batch.
const LEGACY_REFERENCE_MAX_M: u32 = 64;
/// Batches above this default to a sampled CPU reference (all columns, `sample_rows` rows); `audit_rows: "full"` checks every row.
const SPOT_CHECK_MAX_M: u32 = 8;
const ALL_TILINGS: [GemmTiling; 14] = [
    GemmTiling::Tile8x32x64_Simdgroups1x1x4,
    GemmTiling::Tile8x32x128_Simdgroups1x2x4,
    GemmTiling::Tile16x32x32_Simdgroups2x1,
    GemmTiling::Tile8x32x32_Simdgroups1x1,
    GemmTiling::Tile64x32x32_Simdgroups2x2,
    GemmTiling::Tile64x64x32_Simdgroups2x2,
    GemmTiling::Tile32x32x32_Simdgroups2x2,
    GemmTiling::Tile16x32x256_Simdgroups1x1,
    GemmTiling::Tile16x32x256_Simdgroups1x1x4,
    GemmTiling::Tile16x128x256_Simdgroups1x4,
    GemmTiling::Tile32x64x256_Simdgroups2x2,
    GemmTiling::Tile64x32x256_Simdgroups4x1,
    GemmTiling::Tile64x64x256_Simdgroups2x2,
    GemmTiling::Tile128x128x256_Simdgroups4x4,
];

// ---------------------------------------------------------------- job parsing

struct Cell {
    n: u32,
    k: u32,
    m: u32,
    bits: u32,
    group: u32,
    zero_point: bool,
    rht: bool,
    signed_codes: bool,
    seed: u64,
    /// CPU reference coverage: "auto" (spot for m <= 8, sampled rows above), "spot", "sample" or "full".
    audit_rows: String,
}

impl Cell {
    fn parse(
        value: &Value,
        job: &Value,
    ) -> Result<Self, String> {
        let get = |key: &str| value.get(key).and_then(Value::as_u64).ok_or(format!("cell.{key} missing"));
        let dtype = value.get("dtype").and_then(Value::as_str).unwrap_or("bf16");
        if dtype != "bf16" {
            return Err(format!("cell.dtype {dtype} unsupported (bf16 only)"));
        }
        let rht = match value.get("rht").and_then(Value::as_str).unwrap_or("none") {
            "output" => true,
            "none" => false,
            other => return Err(format!("cell.rht {other} unsupported (output|none)")),
        };
        let bits = get("bits")? as u32;
        if !matches!(bits, 4 | 8) {
            return Err(format!("cell.bits {bits} unsupported"));
        }
        Ok(Self {
            n: get("n")? as u32,
            k: get("k")? as u32,
            m: get("m")? as u32,
            bits,
            group: get("group")? as u32,
            zero_point: value.get("zero_point").and_then(Value::as_bool).unwrap_or(bits == 4),
            rht,
            signed_codes: value.get("signed_codes").and_then(Value::as_bool).unwrap_or(false),
            seed: value.get("seed").and_then(Value::as_u64).unwrap_or(42),
            audit_rows: value
                .get("audit_rows")
                .or(job.get("audit_rows"))
                .and_then(Value::as_str)
                .unwrap_or("auto")
                .to_string(),
        })
    }

    fn method(&self) -> QuantizationMethod {
        match (self.bits, self.zero_point) {
            (8, _) => QuantizationMethod::ScaleSymmetric,
            (_, true) => QuantizationMethod::ScaleZeroPoint,
            (_, false) => QuantizationMethod::ScaleBias,
        }
    }

    fn id(&self) -> String {
        format!(
            "n{}_k{}_m{}_w{}g{}_{}",
            self.n,
            self.k,
            self.m,
            self.bits,
            self.group,
            if self.rht {
                "rht"
            } else {
                "nort"
            }
        )
    }

    fn to_json(&self) -> Value {
        json!({"cell_id": self.id(), "n": self.n, "k": self.k, "m": self.m, "bits": self.bits, "group": self.group,
            "zero_point": self.zero_point, "rht": if self.rht {"output"} else {"none"},
            "signed_codes": self.signed_codes, "dtype": "bf16", "seed": self.seed, "audit_rows": self.audit_rows})
    }

    /// Default lane count per quantization group of the legacy M=1 tile (`policy::lane_tile`).
    fn default_group_lanes(&self) -> u32 {
        self.group
            / (2 * if self.bits == 4 {
                8
            } else {
                4
            })
    }
}

#[derive(Clone)]
enum Kind {
    Production,
    Gemv {
        num_simdgroups: u32,
        output_row_tile: u32,
        input_row_tile: Option<u32>, // None = the cell's M
        reduction_lanes: u32,
        group_lanes: Option<u32>, // None = lane default
        mma: bool,                // register-MMA path
        k_split: u32,             // register-MMA path: K slices (simdgroups per row group)
    },
    Gemm {
        engine: GemmEngine,
        tiling: GemmTiling,
        /// None = what the production selector would choose for this tiling (`split_k: "auto"`).
        split_k: Option<u32>,
        /// P2l: weight layout arm (`_R<n>` tiling suffix, see `layout_arm`).
        layout: GemmBLayout,
    },
}

struct Candidate {
    name: String,
    kind: Kind,
    inject_fault: bool,
    /// Real GPU wrong answer: the audit dispatch reads a pool copy whose bf16 scales are all multiplied by 17/16.
    wrong_scale: bool,
    /// Resolve the specialization from a shape with this field flipped (real wrong-answer fault): `signed_codes`.
    misspec: Option<String>,
}

/// P2l layout arm (`_R1` tiling suffix): the packed-W4 GEMM reading `GemmBLayout::SmallBatch` weight units.
fn layout_arm(arm: &str) -> Result<GemmBLayout, String> {
    match arm {
        "1" => Ok(GemmBLayout::SmallBatch),
        other => Err(format!("unknown layout arm _R{other}")),
    }
}

fn u32_field(
    value: &Value,
    key: &str,
) -> Result<Option<u32>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text == "m" || text == "default" => Ok(None),
        Some(other) => other.as_u64().map(|v| Some(v as u32)).ok_or(format!("candidate.{key} must be a number")),
    }
}

impl Candidate {
    fn parse(value: &Value) -> Result<Self, String> {
        let name = value.get("name").and_then(Value::as_str).ok_or("candidate.name missing")?.to_string();
        let inject_fault = value.get("inject_fault").and_then(Value::as_bool).unwrap_or(false);
        let kind = value.get("kind").and_then(Value::as_str).unwrap_or(if name == "production" {
            "production"
        } else {
            "gemv"
        });
        let kind = match kind {
            "production" => Kind::Production,
            "gemv" if value.get("legacy").and_then(Value::as_bool) == Some(true) => Kind::Gemv {
                num_simdgroups: 8,
                output_row_tile: 32,
                input_row_tile: Some(1),
                reduction_lanes: 32,
                group_lanes: None,
                mma: false,
                k_split: 1,
            },
            "gemv" => Kind::Gemv {
                num_simdgroups: u32_field(value, "num_simdgroups")?.ok_or("candidate.num_simdgroups missing")?,
                output_row_tile: u32_field(value, "output_row_tile")?.ok_or("candidate.output_row_tile missing")?,
                input_row_tile: u32_field(value, "input_row_tile")?,
                reduction_lanes: u32_field(value, "reduction_lanes")?.unwrap_or(32),
                group_lanes: u32_field(value, "group_lanes")?,
                mma: value.get("mma").and_then(Value::as_bool).unwrap_or(false),
                k_split: u32_field(value, "k_split")?.unwrap_or(1),
            },
            "gemm" => {
                let engine = match value.get("engine").and_then(Value::as_str) {
                    Some("Simdgroup") => GemmEngine::Simdgroup,
                    Some("Mxu") => GemmEngine::Mxu,
                    other => return Err(format!("candidate.engine {other:?} (Simdgroup|Mxu)")),
                };
                let tiling_name = value.get("tiling").and_then(Value::as_str).ok_or("candidate.tiling missing")?;
                let (tiling_name, layout) = match tiling_name.rsplit_once("_R") {
                    Some((base, arm)) => (base, layout_arm(arm)?),
                    None => (tiling_name, GemmBLayout::RowMajor),
                };
                let tiling = ALL_TILINGS
                    .iter()
                    .copied()
                    .find(|tiling| format!("{tiling:?}") == tiling_name)
                    .ok_or(format!("unknown GemmTiling {tiling_name}"))?;
                Kind::Gemm {
                    engine,
                    tiling,
                    layout,
                    split_k: match value.get("split_k") {
                        Some(Value::String(text)) if text == "auto" => None,
                        _ => Some(u32_field(value, "split_k")?.unwrap_or(1)),
                    },
                }
            },
            other => return Err(format!("candidate.kind {other} (production|gemv|gemm)")),
        };
        Ok(Self {
            name,
            kind,
            inject_fault,
            wrong_scale: value.get("fault").and_then(Value::as_str) == Some("wrong_scale"),
            misspec: value.get("misspec").and_then(Value::as_str).map(str::to_string),
        })
    }
}

// ---------------------------------------------------------------- runners

enum Runner {
    Production(MatmulMetalKernel),
    Gemv {
        kernel: GemvKernel,
        work: MatmulOutputWork,
        with_rht: GemvSpecialization,
        plain: GemvSpecialization,
    },
    Gemm {
        kernel: GemmKernel,
        work: MatmulOutputWork,
        plan: GemmPlan,
    },
}

impl Runner {
    fn encode(
        &mut self,
        args: Args<'_>,
        rht: bool,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(), MatmulMetalError> {
        match self {
            Runner::Production(kernel) => kernel.encode(args, command_buffer),
            Runner::Gemv {
                kernel,
                work,
                with_rht,
                plain,
            } => kernel.encode(
                args,
                if rht {
                    *with_rht
                } else {
                    *plain
                },
                work,
                command_buffer,
            ),
            Runner::Gemm {
                kernel,
                work,
                plan,
            } => kernel.encode_plan(args, *plan, work, command_buffer),
        }
    }
}

struct Entry {
    name: String,
    runner: Option<Runner>,
    illegal: Option<String>,
    /// Resolved description: specialization / plan, function name and constants.
    resolved: Value,
    /// Whether the RHT is fused into the main kernel (None when the cell has no RHT or the entry is illegal).
    fused: Option<bool>,
    /// GEMV kernels must match the legacy GEMV to the last bit; GEMM kernels (bf16 MMA) are checked against f32 only.
    exact: bool,
    inject_fault: bool,
    wrong_scale: bool,
    /// The A operand is the PackedHalf image of the cell's activations (what a model passes the packed-W4 GEMM).
    a_packed: bool,
    layout: GemmBLayout,
    /// Specialization of the same tile without the RHT: input of the direct (resident / control) path.
    plain_spec: Option<GemvSpecialization>,
    direct: Option<Result<GemvMetalKernel, String>>,
}

impl Entry {
    fn illegal(
        name: &str,
        reason: String,
        inject_fault: bool,
        wrong_scale: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            runner: None,
            illegal: Some(reason),
            resolved: Value::Null,
            fused: None,
            exact: true,
            inject_fault,
            wrong_scale,
            a_packed: false,
            layout: GemmBLayout::RowMajor,
            plain_spec: None,
            direct: None,
        }
    }
}

/// Parse a derived-Debug struct (`Name { a: 1, b: Foo(X | Y) }`) into a JSON object.
fn debug_fields<T: std::fmt::Debug>(value: &T) -> Map<String, Value> {
    let text = format!("{value:?}");
    let mut map = Map::new();
    let Some(open) = text.find('{') else {
        map.insert("debug".into(), Value::String(text));
        return map;
    };
    let body = text[open + 1..].trim_end_matches('}').trim();
    let (mut depth, mut start, mut parts) = (0i32, 0usize, Vec::new());
    for (index, character) in body.char_indices() {
        match character {
            '(' | '{' | '[' => depth += 1,
            ')' | '}' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&body[start..index]);
                start = index + 1;
            },
            _ => {},
        }
    }
    parts.push(&body[start..]);
    for part in parts {
        if let Some((key, val)) = part.split_once(':') {
            let val = val.trim();
            let json_value = if let Ok(number) = val.parse::<u64>() {
                json!(number)
            } else if let Ok(flag) = val.parse::<bool>() {
                json!(flag)
            } else {
                json!(val)
            };
            map.insert(key.trim().to_string(), json_value);
        }
    }
    map
}

/// Mangled Metal entry-point name + function constants of a resolved GEMV specialization.
fn gemv_function(fields: &Map<String, Value>) -> Value {
    let text = |key: &str| match &fields[key] {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let variants = [
        "bfloat".to_string(),
        "bfloat".to_string(),
        "bfloat".to_string(),
        text("b_prologue"),
        text("group_size"),
        text("bits"),
        text("k_split"),
        text("input_aligned"),
        text("input_row_tile"),
        text("output_row_tile"),
        text("reduction_lanes"),
        text("group_lanes"),
        text("num_simdgroups"),
    ];
    let name = format!("_D4Gemv{}", variants.iter().map(|v| format!("S{}V{}", v.len(), v)).collect::<String>());
    json!({"name": name, "constants": {
        "output_transform": fields["output_transform"], "gathered": fields["gathered"],
        "signed_codes": fields["signed_codes"], "gemm_b_layout": fields["gemm_b_layout"],
        "full_tile": fields["full_tile"]}})
}

fn describe_gemv(spec: &GemvSpecialization) -> Value {
    let fields = debug_fields(spec);
    let function = gemv_function(&fields);
    json!({"kind": "gemv", "specialization": fields, "function": function})
}

fn describe_gemm(plan: &GemmPlan) -> Value {
    json!({"kind": "gemm", "engine": format!("{:?}", plan.engine), "tiling": format!("{:?}", plan.tiling),
        "split_k": plan.split_k})
}

/// Everything shared by the candidates of one cell.
struct CellData {
    /// PackedHalf image of H(x) (see `build_packed`), when a candidate needs it.
    packed: Option<Gb>,
    input: QuantInput<bf16>,
    pool: Vec<QuantBuffers<Metal, bf16>>,
    /// P2l: one pool per layout arm used by the candidates (weights relaid out, `weights_layout` set).
    alt_pools: Vec<(GemmBLayout, Vec<QuantBuffers<Metal, bf16>>)>,
    ys: Vec<Gb>,
    rht_factors: Vec<i32>,
    rht_buffer: Gb,
}

impl CellData {
    fn args<'a>(
        &'a self,
        x: &'a Gb,
        y: &'a mut Gb,
        copy: usize,
        rht: bool,
        m: u32,
        packed: bool,
        layout: GemmBLayout,
    ) -> Args<'a> {
        let pool = match self.alt_pools.iter().find(|(arm, _)| *arm == layout) {
            Some((_, pool)) => pool,
            None => &self.pool,
        };
        MatmulArguments {
            a: match (&self.packed, packed) {
                (Some(values), true) => MatmulA::PackedHalf {
                    values,
                    group_size: self.input.group_size,
                },
                _ => MatmulA::FullPrecision {
                    values: x,
                    offset: 0,
                },
            },
            b: pool[copy].matmul_b(&self.input),
            b_leading_dimension: None,
            b_transpose: true,
            output: MatmulOutput::new(
                y,
                MatmulDOps {
                    rht_factors: rht.then_some(&self.rht_buffer),
                    ..MatmulDOps::none()
                },
            ),
            gather_indices: None,
            m,
            n: self.input.n,
            k: self.input.k,
        }
    }

    fn shape(
        &self,
        context: &MetalContext,
        rht: bool,
        m: u32,
    ) -> MatmulShape {
        let mut y = create_buffer::<Metal, bf16>(context, 1);
        MatmulShape::from_arguments(&self.args(&self.pool[0].x, &mut y, 0, rht, m, false, GemmBLayout::RowMajor))
    }
}

/// P2l: lane-ordered weight units (`pack_small_batch_units`) built from the cell's production buffers.
fn relayout_units(
    context: &MetalContext,
    data: &CellData,
    cell: &Cell,
) -> Vec<u32> {
    let groups = (cell.k / cell.group) as u32;
    let params = QuantParams::new(QuantParamsLayout::GroupOutput, cell.n, groups);
    let source = &data.pool[0];
    let codes = buffer_to_vec::<Metal, u32>(context, &source.w);
    let scales = buffer_to_vec::<Metal, bf16>(context, &source.scales);
    let zero_points = source.zp.as_ref().map(|buffer| buffer_to_vec::<Metal, u8>(context, buffer));
    let biases = source.bias.as_ref().map(|buffer| buffer_to_vec::<Metal, bf16>(context, buffer));
    let offsets = match (&zero_points, &biases) {
        (Some(zero_points), _) => Offsets::ZeroPoints(zero_points, params.zero_point_strides(QuantizationMode::U4)),
        (_, Some(biases)) => Offsets::Biases(biases),
        _ => panic!("zero points or biases"),
    };
    pack_small_batch_units(
        cell.n as usize,
        cell.k as usize,
        cell.group as usize,
        &codes,
        &scales,
        params.scale_strides(),
        offsets,
    )
}

/// PackedHalf image of the cell's activations for the packed-W4 GEMM. The pack kernel applies H32 (all signs +1), so it is
/// fed H32(x): the GEMM then sees x (to half precision) and the CPU reference can keep using `input.x`.
fn build_packed(
    context: &MetalContext,
    data: &CellData,
    cell: &Cell,
) -> Result<Gb, String> {
    let (m, k, group) = (cell.m as usize, cell.k as usize, cell.group);
    let mut transformed = Vec::with_capacity(m * k);
    for block in data.input.x.chunks_exact(32) {
        let mut values = [0f32; 32];
        for (value, source) in values.iter_mut().zip(block) {
            *value = source.to_f32();
        }
        hadamard32(&mut values);
        transformed.extend(values.iter().map(|&value| bf16::from_f32(value)));
    }
    let source = create_buffer_with_data::<Metal, bf16>(context, &transformed);
    let signs = create_buffer_with_data::<Metal, i32>(context, &vec![1; k]);
    let mut packed =
        create_buffer::<Metal, u32>(context, packed_half_bytes(cell.k, group) / std::mem::size_of::<u32>());
    let mut command_buffer = context.create_command_buffer(Some("autotune_pack"), None).map_err(|e| e.to_string())?;
    ActivationTransform::<Metal>::quantize(
        context,
        DataType::BF16,
        ActivationFormat::PackedHalf {
            group_size: group,
        },
    )
    .map_err(|e| format!("pack kernel: {e}"))?
    .encode_pack_half(&source, &mut packed, &signs, cell.m, cell.k, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).map_err(|e| e.to_string())?;
    Ok(packed)
}

fn legacy_tile(cell: &Cell) -> GemvTile {
    GemvTile::quantized_output_tile(8, 32, 1, 32, cell.default_group_lanes())
}

fn resolve(
    candidate: &Candidate,
    cell: &Cell,
    context: &MetalContext,
    shape_rht: &MatmulShape,
    shape_plain: &MatmulShape,
) -> Entry {
    let dtype = DataType::BF16;
    // A deliberate mis-specification: the kernel is specialized for a shape the data does not have.
    let flipped = |shape: &MatmulShape| MatmulShape {
        signed_codes: !shape.signed_codes,
        ..*shape
    };
    let (shape_rht, shape_plain) = match candidate.misspec.as_deref() {
        Some("signed_codes") => (flipped(shape_rht), flipped(shape_plain)),
        _ => (*shape_rht, *shape_plain),
    };
    let (shape_rht, shape_plain) = (&shape_rht, &shape_plain);
    let name = candidate.name.as_str();
    let fail = |reason: String| Entry::illegal(name, reason, candidate.inject_fault, candidate.wrong_scale);
    let work = || MatmulOutputWork::new(context, dtype, dtype).map_err(|e| format!("MatmulOutputWork: {e}"));
    let rht_shape = if cell.rht {
        shape_rht
    } else {
        shape_plain
    };
    match &candidate.kind {
        Kind::Production => {
            let kernel = match MatmulMetalKernel::new(context, dtype, dtype, dtype) {
                Ok(kernel) => kernel,
                Err(error) => return fail(format!("MatmulMetalKernel::new: {error}")),
            };
            // The activation format a model linear uses: prepared once at load time, kept per batch size by the kernel.
            let model_format = |shape: &MatmulShape| {
                let prepared = kernel.activation_preparation(shape, GemmBLayout::RowMajor, context);
                kernel.select_activation_format(shape, GemmBLayout::RowMajor, prepared, context)
            };
            let format = model_format(rht_shape);
            let a_packed = matches!(format, ActivationFormat::PackedHalf { .. });
            let (shape_rht, shape_plain) = (
                &MatmulShape {
                    a: format,
                    ..*shape_rht
                },
                &MatmulShape {
                    a: format,
                    ..*shape_plain
                },
            );
            let rht_shape = if cell.rht {
                shape_rht
            } else {
                shape_plain
            };
            let dispatch = MatmulMetalKernel::choose_dispatch(
                rht_shape,
                GemmBLayout::RowMajor,
                context.gpu_core_count,
                context.supports_mxu,
                dtype,
                dtype,
                dtype,
            );
            let plain_spec = match MatmulMetalKernel::choose_dispatch(
                shape_plain,
                GemmBLayout::RowMajor,
                context.gpu_core_count,
                context.supports_mxu,
                dtype,
                dtype,
                dtype,
            ) {
                MatmulDispatch::Gemv(spec) => Some(spec),
                MatmulDispatch::Gemm(_) => None,
            };
            let (resolved, fused, exact) = match &dispatch {
                MatmulDispatch::Gemv(spec) => (describe_gemv(spec), spec.fuses_rht(), true),
                MatmulDispatch::Gemm(plan) => (describe_gemm(plan), plan.split_k == 1, false),
            };
            Entry {
                name: name.to_string(),
                runner: Some(Runner::Production(kernel)),
                illegal: None,
                resolved,
                fused: cell.rht.then_some(fused),
                exact,
                inject_fault: candidate.inject_fault,
                wrong_scale: candidate.wrong_scale,
                a_packed,
                layout: GemmBLayout::RowMajor,
                plain_spec,
                direct: None,
            }
        },
        Kind::Gemv {
            num_simdgroups,
            output_row_tile,
            input_row_tile,
            reduction_lanes,
            group_lanes,
            mma,
            k_split,
        } => {
            let input_row_tile = input_row_tile.unwrap_or(cell.m);
            let group_lanes = group_lanes.unwrap_or(cell.default_group_lanes());
            if *num_simdgroups == 0 || !output_row_tile.is_multiple_of(*num_simdgroups) {
                return fail("output_row_tile not a multiple of num_simdgroups".into());
            }
            if *mma || *k_split != 1 {
                // Port note: origin/main's GemvTile has no register-MMA path (the old QMV kernels were not ported) and no K-split GEMV tile.
                return fail("register-MMA / k_split GEMV tiles do not exist in this build".into());
            }
            let tile = {
                if !matches!(reduction_lanes, 8 | 16 | 32) || !reduction_lanes.is_multiple_of(group_lanes.max(1)) {
                    return fail("reduction_lanes must be 8|16|32 and a multiple of group_lanes".into());
                }
                if !(output_row_tile / num_simdgroups).is_multiple_of(32 / reduction_lanes) {
                    return fail("rows per simdgroup not a multiple of 32 / reduction_lanes".into());
                }
                GemvTile::quantized_output_tile(
                    *num_simdgroups,
                    *output_row_tile,
                    input_row_tile,
                    *reduction_lanes,
                    group_lanes,
                )
            };
            let select = |shape: &MatmulShape| GemvSpecialization::select_tile(shape, GemmBLayout::RowMajor, tile);
            let (Some(with_rht), Some(plain)) = (select(shape_rht), select(shape_plain)) else {
                return fail("GemvSpecialization::select_tile rejected the tile".into());
            };
            let primary = if cell.rht {
                with_rht
            } else {
                plain
            };
            let work = match work() {
                Ok(work) => work,
                Err(reason) => return fail(reason),
            };
            Entry {
                name: name.to_string(),
                runner: Some(Runner::Gemv {
                    kernel: GemvKernel::new(dtype, dtype, dtype),
                    work,
                    with_rht,
                    plain,
                }),
                illegal: None,
                resolved: describe_gemv(&primary),
                fused: cell.rht.then(|| primary.fuses_rht()),
                exact: true,
                inject_fault: candidate.inject_fault,
                wrong_scale: candidate.wrong_scale,
                a_packed: false,
                layout: GemmBLayout::RowMajor,
                plain_spec: Some(plain),
                direct: None,
            }
        },
        Kind::Gemm {
            engine,
            tiling,
            split_k,
            layout,
        } => {
            let plan = match split_k {
                Some(split_k) => GemmPlan {
                    engine: *engine,
                    tiling: *tiling,
                    split_k: *split_k,
                },
                None => GemmProblem::new(
                    MatmulShape {
                        a: ActivationFormat::PackedHalf {
                            group_size: cell.group,
                        },
                        ..*rht_shape
                    },
                    dtype,
                    dtype,
                    false,
                )
                .simdgroup_plan(*tiling),
            };
            let work = match work() {
                Ok(work) => work,
                Err(reason) => return fail(reason),
            };
            Entry {
                name: name.to_string(),
                runner: Some(Runner::Gemm {
                    kernel: GemmKernel::new(dtype, dtype, dtype),
                    work,
                    plan,
                }),
                illegal: None,
                resolved: describe_gemm(&plan),
                fused: cell.rht.then_some(plan.split_k == 1),
                exact: false,
                inject_fault: candidate.inject_fault,
                wrong_scale: candidate.wrong_scale,
                a_packed: tiling.is_packed_w4(),
                layout: *layout,
                plain_spec: None,
                direct: None,
            }
        },
    }
}

// ---------------------------------------------------------------- audit

#[derive(Default)]
struct UlpStats {
    max_ulp: f32,
    /// Buckets: 0, (0,1], (1,2], (2,4], (4,8], >8 (and NaN / unwritten).
    histogram: [u64; 6],
    bad_rows: u32,
    unwritten: u64,
    nan: u64,
}

impl UlpStats {
    fn passes(&self) -> bool {
        self.max_ulp <= ULP_PASS && self.unwritten == 0 && self.nan == 0
    }

    fn to_json(&self) -> Value {
        json!({"max_ulp": self.max_ulp, "histogram": {"0": self.histogram[0], "le1": self.histogram[1],
            "le2": self.histogram[2], "le4": self.histogram[3], "le8": self.histogram[4], "gt8": self.histogram[5]},
            "bad_rows": self.bad_rows, "unwritten_sentinels": self.unwritten, "nan": self.nan})
    }
}

fn bf16_ulp(value: f32) -> f32 {
    let exponent = ((value.to_bits() >> 23) & 0xFF) as i32 - 127;
    2f32.powi(exponent - 7)
}

/// Per-row ULP error of `actual` vs `reference` (both `rows x n`), measured at `ulp(max(|ref|, row_rms / 2))`.
fn ulp_compare(
    actual: &[bf16],
    reference: &[bf16],
    n: usize,
) -> UlpStats {
    let mut stats = UlpStats::default();
    for (actual_row, reference_row) in actual.chunks(n).zip(reference.chunks(n)) {
        let rms = (reference_row.iter().map(|v| v.to_f32().powi(2)).sum::<f32>() / n as f32).sqrt();
        let floor = rms * 0.5;
        let mut row_bad = false;
        for (a, r) in actual_row.iter().zip(reference_row) {
            let error = if a.to_bits() == SENTINEL_BITS {
                stats.unwritten += 1;
                f32::INFINITY
            } else if a.is_nan() || r.is_nan() {
                stats.nan += 1;
                f32::INFINITY
            } else {
                let r = r.to_f32();
                (a.to_f32() - r).abs() / bf16_ulp(r.abs().max(floor).max(f32::MIN_POSITIVE))
            };
            let bucket = match error {
                e if e == 0.0 => 0,
                e if e <= 1.0 => 1,
                e if e <= 2.0 => 2,
                e if e <= 4.0 => 3,
                e if e <= 8.0 => 4,
                _ => 5,
            };
            stats.histogram[bucket] += 1;
            if error.is_finite() {
                stats.max_ulp = stats.max_ulp.max(error);
            } else {
                stats.max_ulp = f32::INFINITY;
            }
            row_bad |= error > ULP_PASS;
        }
        stats.bad_rows += u32::from(row_bad);
    }
    stats
}

fn hadamard32(values: &mut [f32; 32]) {
    let mut stride = 1;
    while stride < 32 {
        for lane in 0..32 {
            if lane & stride == 0 {
                let (a, b) = (values[lane], values[lane | stride]);
                values[lane] = a + b;
                values[lane | stride] = a - b;
            }
        }
        stride <<= 1;
    }
    values.iter_mut().for_each(|v| *v *= 1.0 / 32f32.sqrt());
}

/// Summary of one f32 CPU reference check (`cpu_audit`).
struct CpuAudit {
    max_err: f32,
    /// Elements whose error exceeds the class tolerance.
    over: u64,
    elements: u64,
    rows: usize,
    columns: usize,
    /// (row, column) of the largest error.
    worst: (usize, usize),
    mode: &'static str,
    seconds: f64,
}

impl CpuAudit {
    fn to_json(&self) -> Value {
        json!({"mode": self.mode, "rows_checked": self.rows, "columns_checked": self.columns, "elements": self.elements,
            "over_tolerance": self.over, "worst_row": self.worst.0, "worst_col": self.worst.1,
            "cpu_seconds": self.seconds})
    }
}

/// Rows and whether to cover every column block, per the cell's `audit_rows`: `spot` = rows {0, m-1} x three 32-column blocks
/// (the original GEMV check), `sample` = ~25 rows (0, 1, m/2, m-1 and either side of every power of two, which straddles the
/// 8/16/32/64/128-row tile edges, plus 8 seeded random rows) x all columns, `full` = every row x all columns.
fn audit_plan(cell: &Cell) -> (&'static str, Vec<usize>, bool) {
    let m = cell.m as usize;
    let mode = match cell.audit_rows.as_str() {
        "spot" => "spot",
        "sample" => "sample",
        "full" => "full",
        _ if cell.m <= SPOT_CHECK_MAX_M => "spot",
        _ => "sample",
    };
    let rows: Vec<usize> = match mode {
        "full" => (0..m).collect(),
        "sample" => {
            let mut rows = vec![0, 1, m / 2, m - 1, m.saturating_sub(2)];
            let mut power = 2;
            while power < m {
                rows.extend([power - 1, power]);
                power *= 2;
            }
            let mut rng = SmallRng::seed_from_u64(cell.seed ^ 0xA0D17);
            rows.extend((0..8).map(|_| rng.random_range(0..m)));
            rows.retain(|&row| row < m);
            rows.sort_unstable();
            rows.dedup();
            rows
        },
        _ => {
            let mut rows = vec![0, m - 1];
            rows.dedup();
            rows
        },
    };
    (mode, rows, mode != "spot")
}

/// f32 dot product with 16 independent accumulators (auto-vectorizes; the order is irrelevant at the 0.03 tolerance).
fn dot_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    let mut acc = [0f32; 16];
    let (a_chunks, b_chunks) = (a.chunks_exact(16), b.chunks_exact(16));
    let tail: f32 = a_chunks.remainder().iter().zip(b_chunks.remainder()).map(|(x, y)| x * y).sum();
    for (ca, cb) in a_chunks.zip(b_chunks) {
        for lane in 0..16 {
            acc[lane] += ca[lane] * cb[lane];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Dequantize one output column (`k` weights) to f32 exactly as the format defines it (no bf16 rounding of the product).
fn dequant_column(
    cell: &Cell,
    input: &QuantInput<bf16>,
    column: usize,
    out: &mut [f32],
) {
    let k = cell.k as usize;
    let group_size = cell.group as usize;
    let groups = k.div_ceil(group_size);
    let pack = (32 / cell.bits) as usize;
    let mask = (1u32 << cell.bits) - 1;
    let zero_point_stride = if cell.bits == 4 {
        2 * groups.div_ceil(2)
    } else {
        groups
    };
    for group in 0..groups {
        let scale = input.scales[column * groups + group].to_f32();
        let (zero, bias) = match cell.method() {
            QuantizationMethod::ScaleZeroPoint => {
                let index = column * zero_point_stride + group;
                let byte = input.zero_points.as_ref().unwrap()[index / 2];
                (
                    if index % 2 == 0 {
                        byte & 0x0F
                    } else {
                        byte >> 4
                    } as f32,
                    0.0,
                )
            },
            QuantizationMethod::ScaleBias => (0.0, input.biases.as_ref().unwrap()[column * groups + group].to_f32()),
            QuantizationMethod::ScaleSymmetric => (128.0, 0.0),
        };
        for kk in group * group_size..((group + 1) * group_size).min(k) {
            let linear = column * k + kk;
            let code = ((input.w_packed[linear / pack] >> ((linear % pack) as u32 * cell.bits)) & mask) as f32;
            out[kk] = scale * (code - zero) + bias;
        }
    }
}

/// f32 CPU reference: dequantize + matmul (+ output RHT) on the rows/columns `audit_plan` picks, multi-threaded over 32-column
/// blocks (the RHT block). Error per element = `|got - expected| / max(|expected|, block_rms)` (block = 32 columns of one row).
/// Cost: (rows + 1) x N x K flops; the full 1024 x 5120 x 17408 check takes seconds on a laptop-class CPU.
fn cpu_audit(
    cell: &Cell,
    data: &CellData,
    actual: &[bf16],
    tolerance: f32,
) -> CpuAudit {
    let started = Instant::now();
    let input = &data.input;
    let (n, k) = (cell.n as usize, cell.k as usize);
    let (mode, rows, all_columns) = audit_plan(cell);
    let blocks = n.div_ceil(32);
    let block_ids: Vec<usize> = if all_columns {
        (0..blocks).collect()
    } else {
        let mut ids = vec![0, blocks / 2, blocks - 1];
        ids.dedup();
        ids
    };
    let xs: Vec<Vec<f32>> =
        rows.iter().map(|&row| input.x[row * k..(row + 1) * k].iter().map(|v| v.to_f32()).collect()).collect();
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |count| count.get()).min(block_ids.len());
    let partials: Vec<(f32, u64, u64, (usize, usize))> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let (mut worst, mut over, mut elements, mut at) = (0f32, 0u64, 0u64, (0usize, 0usize));
                    let mut weights = vec![0f32; 8 * k];
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&block) = block_ids.get(index) else {
                            break;
                        };
                        let lanes = (n - block * 32).min(32);
                        let mut values = vec![[0f32; 32]; rows.len()];
                        for first in (0..lanes).step_by(8) {
                            let count = (lanes - first).min(8);
                            for c in 0..count {
                                dequant_column(cell, input, block * 32 + first + c, &mut weights[c * k..(c + 1) * k]);
                            }
                            for (r, x) in xs.iter().enumerate() {
                                for c in 0..count {
                                    values[r][first + c] = dot_f32(x, &weights[c * k..(c + 1) * k]);
                                }
                            }
                        }
                        for (r, &row) in rows.iter().enumerate() {
                            let expected = &mut values[r];
                            if cell.rht {
                                hadamard32(expected);
                                for (lane, value) in expected.iter_mut().enumerate() {
                                    *value *= data.rht_factors[block * 32 + lane] as f32;
                                }
                            }
                            let rms = (expected[..lanes].iter().map(|v| v * v).sum::<f32>() / lanes as f32).sqrt();
                            for lane in 0..lanes {
                                let got = actual[row * n + block * 32 + lane].to_f32();
                                let error = if got.is_nan() {
                                    f32::INFINITY
                                } else {
                                    (got - expected[lane]).abs() / expected[lane].abs().max(rms).max(f32::MIN_POSITIVE)
                                };
                                elements += 1;
                                over += u64::from(error > tolerance);
                                if error > worst {
                                    (worst, at) = (error, (row, block * 32 + lane));
                                }
                            }
                        }
                    }
                    (worst, over, elements, at)
                })
            })
            .collect();
        workers.into_iter().map(|worker| worker.join().expect("cpu audit worker")).collect()
    });
    let mut audit = CpuAudit {
        max_err: 0.0,
        over: 0,
        elements: 0,
        rows: rows.len(),
        columns: if all_columns {
            n
        } else {
            block_ids.len() * 32
        },
        worst: (0, 0),
        mode,
        seconds: 0.0,
    };
    for (worst, over, elements, at) in partials {
        audit.over += over;
        audit.elements += elements;
        if worst > audit.max_err {
            (audit.max_err, audit.worst) = (worst, at);
        }
    }
    audit.seconds = started.elapsed().as_secs_f64();
    audit
}

fn run_once(
    context: &MetalContext,
    runner: &mut Runner,
    data: &mut CellData,
    cell: &Cell,
    copy: usize,
    packed: bool,
    layout: GemmBLayout,
) -> Result<Vec<bf16>, String> {
    let sentinel = vec![bf16::from_bits(SENTINEL_BITS); (cell.m * cell.n) as usize];
    data.ys[copy] = create_buffer_with_data::<Metal, bf16>(context, &sentinel);
    let mut command_buffer = context.create_command_buffer(Some("autotune_audit"), None).map_err(|e| e.to_string())?;
    {
        let mut y = std::mem::replace(&mut data.ys[copy], create_buffer::<Metal, bf16>(context, 1));
        let result = runner.encode(
            data.args(&data.pool[copy].x, &mut y, copy, cell.rht, cell.m, packed, layout),
            cell.rht,
            &mut command_buffer,
        );
        data.ys[copy] = y;
        result.map_err(|e| e.to_string())?;
    }
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).map_err(|e| e.to_string())?;
    Ok(buffer_to_vec::<Metal, bf16>(context, &data.ys[copy]))
}

/// Legacy M=1 GEMV once per input row: fused-RHT / deferred-RHT / plain references, `m x n` each.
fn legacy_references(
    context: &MetalContext,
    data: &mut CellData,
    cell: &Cell,
) -> Result<(Vec<bf16>, Vec<bf16>), String> {
    let dtype = DataType::BF16;
    let tile = legacy_tile(cell);
    let mut kernel = GemvKernel::new(dtype, dtype, dtype);
    let work = MatmulOutputWork::new(context, dtype, dtype).map_err(|e| e.to_string())?;
    let (mut fused, mut deferred) = (Vec::new(), Vec::new());
    let (k, n) = (cell.k as usize, cell.n as usize);
    for row in 0..cell.m as usize {
        let x = create_buffer_with_data::<Metal, bf16>(context, &data.input.x[row * k..(row + 1) * k]);
        for deferred_mode in [false, true] {
            if deferred_mode && !cell.rht {
                continue;
            }
            let use_rht = cell.rht && !deferred_mode;
            let mut y = create_buffer::<Metal, bf16>(context, n);
            let mut command_buffer =
                context.create_command_buffer(Some("autotune_ref"), None).map_err(|e| e.to_string())?;
            let args = data.args(&x, &mut y, 0, use_rht, 1, false, GemmBLayout::RowMajor);
            let shape = MatmulShape::from_arguments(&args);
            let spec = GemvSpecialization::select_tile(&shape, GemmBLayout::RowMajor, tile)
                .ok_or("legacy tile rejected by select_tile")?;
            assert_eq!(spec.fuses_rht(), use_rht, "legacy reference must fuse RHT iff requested");
            kernel.encode(args, spec, &work, &mut command_buffer).map_err(|e| format!("legacy ref: {e}"))?;
            if deferred_mode {
                work.apply(&mut y, &data.rht_buffer, None, 1, cell.n, &mut command_buffer);
            }
            block_on(command_buffer.end_encoding().submit().wait_until_completed()).map_err(|e| e.to_string())?;
            let values = buffer_to_vec::<Metal, bf16>(context, &y);
            if deferred_mode {
                deferred.extend(values);
            } else {
                fused.extend(values);
            }
        }
    }
    Ok((fused, deferred))
}

// ---------------------------------------------------------------- timing

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

/// Host monotonic clock in ns (`CLOCK_UPTIME_RAW`, the unit of `mach_absolute_time` converted to ns).
fn uptime_ns() -> u64 {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: u32) -> u64;
    }
    const CLOCK_UPTIME_RAW: u32 = 8;
    unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
}

/// 1-minute host load average (other processes competing for CPU and memory bandwidth on a shared machine).
fn load1() -> f64 {
    unsafe extern "C" {
        fn getloadavg(
            loadavg: *mut f64,
            nelem: i32,
        ) -> i32;
    }
    let mut load = [0f64; 3];
    unsafe { getloadavg(load.as_mut_ptr(), 3) };
    load[0]
}

/// Host CPU state: 1-minute load, instantaneous CPU busy % (`top`, second 1 s sample) and the top CPU processes.
fn host_snapshot() -> Value {
    use std::process::Command;
    let text = |program: &str, args: &[&str]| {
        Command::new(program).args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
    };
    let idle = text("top", &["-l", "2", "-s", "1", "-n", "0"]).ok().and_then(|out| {
        let line = out.lines().filter(|l| l.starts_with("CPU usage:")).next_back()?.to_string();
        let part = line.split(',').find(|part| part.trim_end().ends_with("idle"))?.to_string();
        part.split_whitespace().next()?.trim_end_matches('%').parse::<f64>().ok()
    });
    let top: Vec<String> = text("ps", &["-Aceo", "pcpu,pid,comm", "-r"])
        .map(|out| out.lines().skip(1).take(5).map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect())
        .unwrap_or_default();
    json!({"load1": load1(), "cpu_busy_pct": idle.map(|idle| 100.0 - idle), "top": top})
}

/// Waits (up to `checks` x `wait_s`) for the host CPU to go quiet; returns the log and whether it never did.
fn host_guard(guard: &Value) -> (Value, bool) {
    let max_busy = guard["max_cpu_busy_pct"].as_f64().unwrap_or(10.0);
    let checks = guard["checks"].as_u64().unwrap_or(12);
    let wait_s = guard["wait_s"].as_u64().unwrap_or(15);
    let started = Instant::now();
    let mut log = Vec::new();
    let mut quiet = false;
    for check in 0..checks.max(1) {
        let snapshot = host_snapshot();
        quiet = snapshot["cpu_busy_pct"].as_f64().is_none_or(|busy| busy <= max_busy);
        log.push(snapshot);
        if quiet || guard["enabled"].as_bool() == Some(false) || check + 1 == checks.max(1) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(wait_s));
    }
    let noisy = !quiet && guard["enabled"].as_bool() != Some(false);
    (json!({"max_cpu_busy_pct": max_busy, "checks": log, "waited_s": started.elapsed().as_secs_f64()}), noisy)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotKind {
    /// The whole linear exactly as `MatmulKernel::encode` runs it (includes a deferred RHT pass).
    Whole,
    /// The same tile without the cell's RHT.
    KernelOnly,
    /// The tile without RHT through the raw pipeline, out_vec_size = n (control for the resident slot).
    DirectNormal,
    /// Same raw pipeline, but every threadgroup clamps to the same weight rows (see `encode_direct`).
    Resident,
}

impl SlotKind {
    fn suffix(self) -> &'static str {
        match self {
            SlotKind::Whole => "",
            SlotKind::KernelOnly => "~kernel_only",
            SlotKind::DirectNormal => "~direct",
            SlotKind::Resident => "~resident",
        }
    }

    fn key(self) -> &'static str {
        match self {
            SlotKind::Whole => "timing",
            SlotKind::KernelOnly => "kernel_only",
            SlotKind::DirectNormal => "direct_normal",
            SlotKind::Resident => "resident",
        }
    }
}

struct Slot {
    entry: usize,
    kind: SlotKind,
    /// Timing protocol: false = `pipelined` (independent dispatches overlap), true = `chain` (see `encode_slot`).
    chain: bool,
    /// Index into the job's protocol list; protocol 0 fills `timing` / `kernel_only`, later ones `timing_<name>`.
    protocol: usize,
    samples: Vec<f64>,
    records: Vec<Value>,
    hold: Option<Value>,
}

struct InFlight {
    pending: MetalCommandBufferPending,
    commit_ns: u64,
    enc_us: f64,
    label: String,
}

fn build_direct(
    context: &MetalContext,
    spec: &GemvSpecialization,
) -> Result<GemvMetalKernel, String> {
    let fields = debug_fields(spec);
    let number = |key: &str| fields[key].as_u64().ok_or(format!("spec.{key}")).map(|v| v as u32);
    let flag = |key: &str| fields[key].as_bool().ok_or(format!("spec.{key}"));
    let prologue = match fields["b_prologue"].as_str().unwrap_or_default() {
        "FullPrecision" => GemmBPrologueKind::FullPrecision,
        "ScaleBiasDequant" => GemmBPrologueKind::ScaleBiasDequant,
        "ScaleZeroPointDequant" => GemmBPrologueKind::ScaleZeroPointDequant,
        "ScaleSymmetricDequant" => GemmBPrologueKind::ScaleSymmetricDequant,
        other => return Err(format!("unknown prologue {other}")),
    };
    let dtype = DataType::BF16;
    GemvMetalKernel::new(
        context,
        dtype,
        dtype,
        dtype,
        prologue,
        number("group_size")?,
        number("bits")?,
        number("k_split")?,
        flag("input_aligned")?,
        number("input_row_tile")?,
        number("output_row_tile")?,
        number("reduction_lanes")?,
        number("group_lanes")?,
        number("num_simdgroups")?,
        GemmDTransform::empty(),
        false,
        flag("signed_codes")?,
        GemmBLayout::RowMajor,
        false,
    )
    .map_err(|e| e.to_string())
}

impl CellData {
    /// Raw GEMV dispatch. `resident` shrinks `out_vec_size` to `tile_rows - 1` while the grid keeps
    /// `ceil(n / tile_rows)` threadgroups: with `full_tile = false` the kernel clamps every row index to the last valid
    /// row, so all threadgroups run the identical instruction stream on the same few weight rows (L1 resident), and
    /// only threadgroup 0 stores.
    fn encode_direct(
        &self,
        kernel: &GemvMetalKernel,
        tile_rows: u32,
        resident: bool,
        copy: usize,
        y: &mut Gb,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) {
        let MatmulB::Quantized(quantized) = self.pool[copy].matmul_b(&self.input) else {
            unreachable!("quantized cells only")
        };
        let (scales, zero_points) = (quantized.params.scale_strides(), quantized.zero_point_strides());
        kernel.encode(
            quantized.codes,
            Some(quantized.scales),
            quantized.zero_points(),
            quantized.biases(),
            &self.pool[copy].x,
            y,
            None::<&Gb>,
            None::<&Gb>,
            None::<&Gb>,
            self.input.k,
            if resident {
                tile_rows - 1
            } else {
                self.input.n
            },
            self.input.m,
            1.0,
            self.input.n.div_ceil(tile_rows),
            scales.output_stride,
            scales.group_stride,
            zero_points.output_stride,
            zero_points.group_stride,
            None,
            command_buffer,
        );
    }
}

/// Encode `dispatches` back-to-back dispatches over the rotating cold copies into one command buffer.
///
/// `chain = false` (protocol `pipelined`): every dispatch writes its own output buffer (`ys[copy]`), so nothing orders them and
/// consecutive dispatches overlap on the GPU (what a throughput microbenchmark measures).
///
/// `chain = true` (protocol `chain`, the model-like one): every dispatch writes the SAME output buffer (`ys[start_copy]`). The
/// generated kernel wrappers report their buffer accesses to `MetalCommandBufferEncoding::access`, which inserts a device-visibility
/// barrier before a dispatch whose write overlaps an earlier write: the write-after-write hazard here produces exactly the barrier
/// a dependent op chain has (the old `chain_barrier` hook did the same by hand). The weights still rotate over the cold pool.
/// No shipped code is involved: this is plain buffer aliasing in the harness.
fn encode_slot(
    entry: &mut Entry,
    kind: SlotKind,
    chain: bool,
    data: &CellData,
    cell: &Cell,
    ys: &mut [Gb],
    dispatches: usize,
    start_copy: usize,
    command_buffer: &mut MetalCommandBufferEncoding,
) -> Result<(), String> {
    let copies = data.pool.len();
    let tile_rows = entry.plain_spec.map_or(0, |spec| spec.output_row_tile());
    for dispatch in 0..dispatches {
        let copy = if kind == SlotKind::Resident {
            0
        } else {
            (start_copy + dispatch) % copies
        };
        let out = if chain {
            start_copy % copies
        } else {
            copy
        };
        match kind {
            SlotKind::Whole | SlotKind::KernelOnly => {
                let rht = kind == SlotKind::Whole && cell.rht;
                let packed = entry.a_packed;
                entry
                    .runner
                    .as_mut()
                    .unwrap()
                    .encode(
                        data.args(&data.pool[copy].x, &mut ys[out], copy, rht, cell.m, packed, entry.layout),
                        rht,
                        command_buffer,
                    )
                    .map_err(|e| e.to_string())?;
            },
            SlotKind::DirectNormal | SlotKind::Resident => {
                let kernel = entry.direct.as_ref().unwrap().as_ref().map_err(|e| e.clone())?;
                data.encode_direct(kernel, tile_rows, kind == SlotKind::Resident, copy, &mut ys[out], command_buffer);
            },
        }
    }
    Ok(())
}

/// Encode and commit one command buffer of `dispatches` back-to-back dispatches over rotating cold copies.
fn submit_slot(
    context: &MetalContext,
    entry: &mut Entry,
    kind: SlotKind,
    chain: bool,
    data: &mut CellData,
    cell: &Cell,
    dispatches: usize,
    start_copy: usize,
    label: String,
) -> Result<InFlight, String> {
    let started = Instant::now();
    let mut command_buffer = context.create_command_buffer(Some(&label), None).map_err(|e| e.to_string())?;
    let mut ys = std::mem::take(&mut data.ys);
    let encoded = encode_slot(entry, kind, chain, data, cell, &mut ys, dispatches, start_copy, &mut command_buffer);
    data.ys = ys;
    encoded?;
    let enc_us = started.elapsed().as_secs_f64() * 1e6;
    let commit_ns = uptime_ns();
    Ok(InFlight {
        pending: command_buffer.end_encoding().submit(),
        commit_ns,
        enc_us,
        label,
    })
}

struct Finished {
    /// GPU time per dispatch from the commit feedback.
    us: f64,
    /// Host completion time.
    done_ns: u64,
    commit_ns: u64,
    enc_us: f64,
    label: String,
}

fn finish_slot(
    in_flight: InFlight,
    dispatches: usize,
) -> Result<Finished, String> {
    let completed = block_on(in_flight.pending.wait_until_completed()).map_err(|e| e.to_string())?;
    Ok(Finished {
        us: completed.gpu_execution_time().ok_or("no GPU execution time on the command buffer")?.as_secs_f64() * 1e6
            / dispatches as f64,
        done_ns: uptime_ns(),
        commit_ns: in_flight.commit_ns,
        enc_us: in_flight.enc_us,
        label: in_flight.label,
    })
}

fn window_json(finished: Finished) -> Value {
    json!({"label": finished.label, "commit_ns": finished.commit_ns, "done_ns": finished.done_ns, "us": finished.us})
}

/// Round driver settings and cross-round state.
struct Driver {
    dispatches: usize,
    /// Commit every command buffer of a round before waiting for any (GPU never idles between them).
    queue: bool,
    /// Walk every round forwards (default: odd rounds backwards, A B C / C B A).
    forward: bool,
    /// Always start the cold-pool rotation at copy 0.
    fixed_copy: bool,
    last_done_ns: u64,
    previous_slot: Option<usize>,
}

impl Driver {
    fn round(
        &mut self,
        context: &MetalContext,
        entries: &mut [Entry],
        data: &mut CellData,
        cell: &Cell,
        slots: &mut [Slot],
        round: usize,
        tag: &str,
        record: bool,
    ) -> Result<(), String> {
        let count = slots.len();
        let mut in_flight: Vec<(usize, usize, usize, InFlight)> = Vec::new();
        for step in 0..count {
            let position = if self.forward || round % 2 == 0 {
                step
            } else {
                count - 1 - step
            };
            let slot_index = (position + round) % count;
            let start_copy = if self.fixed_copy {
                0
            } else {
                round * 11 + step
            };
            let slot = &slots[slot_index];
            let label = format!(
                "at|{}|{}{}{}|{}",
                cell.id(),
                entries[slot.entry].name,
                slot.kind.suffix(),
                if slot.chain {
                    "+chain"
                } else {
                    ""
                },
                tag
            );
            let submitted = submit_slot(
                context,
                &mut entries[slot.entry],
                slot.kind,
                slot.chain,
                data,
                cell,
                self.dispatches,
                start_copy,
                label,
            )?;
            if self.queue {
                in_flight.push((step, slot_index, start_copy, submitted));
            } else {
                let gap_us = submitted.commit_ns.saturating_sub(self.last_done_ns) as f64 / 1e3;
                let finished = finish_slot(submitted, self.dispatches)?;
                self.last_done_ns = finished.done_ns;
                self.record(slots, record, round, step, slot_index, start_copy, Some(gap_us), finished);
            }
        }
        for (step, slot_index, start_copy, submitted) in in_flight {
            let finished = finish_slot(submitted, self.dispatches)?;
            self.last_done_ns = finished.done_ns;
            self.record(slots, record, round, step, slot_index, start_copy, None, finished);
        }
        Ok(())
    }

    fn record(
        &mut self,
        slots: &mut [Slot],
        record: bool,
        round: usize,
        position: usize,
        slot_index: usize,
        start_copy: usize,
        gap_us: Option<f64>,
        finished: Finished,
    ) {
        if record {
            slots[slot_index].samples.push(finished.us);
            slots[slot_index].records.push(json!({"round": round, "pos": position, "slot": slot_index,
                "prev": self.previous_slot, "copy0": start_copy, "us": finished.us, "enc_us": finished.enc_us,
                "gap_us": gap_us, "commit_ns": finished.commit_ns, "done_ns": finished.done_ns,
                "load1": load1(), "label": finished.label}));
        }
        self.previous_slot = Some(slot_index);
    }
}

/// Repeat one slot back to back (two command buffers deep) for at least `hold_ms` of GPU time.
fn hold_slot(
    context: &MetalContext,
    entries: &mut [Entry],
    data: &mut CellData,
    cell: &Cell,
    slot: &Slot,
    dispatches: usize,
    hold_ms: u64,
) -> Result<Value, String> {
    let mut pending = std::collections::VecDeque::new();
    let (mut windows, mut gpu_ms, mut index) = (Vec::new(), 0.0f64, 0usize);
    loop {
        let label = format!("at|{}|{}{}|h{}", cell.id(), entries[slot.entry].name, slot.kind.suffix(), index);
        pending.push_back(submit_slot(
            context,
            &mut entries[slot.entry],
            slot.kind,
            slot.chain,
            data,
            cell,
            dispatches,
            index * 5,
            label,
        )?);
        index += 1;
        let done = gpu_ms >= hold_ms as f64;
        if pending.len() >= 2 || done {
            let finished = finish_slot(pending.pop_front().unwrap(), dispatches)?;
            gpu_ms += finished.us * dispatches as f64 / 1e3;
            windows.push(window_json(finished));
        }
        if done {
            break;
        }
    }
    while let Some(submitted) = pending.pop_front() {
        let finished = finish_slot(submitted, dispatches)?;
        gpu_ms += finished.us * dispatches as f64 / 1e3;
        windows.push(window_json(finished));
    }
    Ok(json!({"command_buffers": windows.len(), "gpu_ms": gpu_ms, "windows": windows}))
}

fn timing_json(slot: &Slot) -> Value {
    let rounds_us = &slot.samples;
    let half = rounds_us.len() / 2;
    let drift = if half >= 1 {
        (median(&rounds_us[half..]) / median(&rounds_us[..half]) - 1.0) * 100.0
    } else {
        0.0
    };
    let mut sorted = rounds_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let quantile = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    json!({"rounds_us": rounds_us, "median_us": median(rounds_us), "q25_us": quantile(0.25), "p90_us": quantile(0.9),
        "min_us": sorted[0], "drift_pct": drift,
        "samples": slot.records, "profile_hold": slot.hold})
}

// ---------------------------------------------------------------- driver

/// Timing protocols of the job: `timing.protocols` (`pipelined` | `chain`, first one fills `timing`), else `["chain"]` when
/// `AT_CHAIN` is set (the old env switch: the whole job is model-like), else `["pipelined"]`. Returns names and chain flags.
fn protocol_list(timing: &Value) -> (Vec<&'static str>, Vec<bool>) {
    let names: Vec<&'static str> = match timing["protocols"].as_array() {
        Some(list) if !list.is_empty() => list
            .iter()
            .map(|name| match name.as_str() {
                Some("pipelined") => "pipelined",
                Some("chain") => "chain",
                other => panic!("timing.protocols: {other:?} (pipelined|chain)"),
            })
            .collect(),
        _ if std::env::var_os("AT_CHAIN").is_some() => vec!["chain"],
        _ => vec!["pipelined"],
    };
    let chain = names.iter().map(|name| *name == "chain").collect();
    (names, chain)
}

fn run_cell(
    context: &MetalContext,
    cell: &Cell,
    candidates: &[Candidate],
    job: &Value,
) -> Result<Value, String> {
    let timing = &job["timing"];
    let (protocol_names, protocols) = protocol_list(timing);
    let dispatches = timing["dispatches"].as_u64().unwrap_or(32) as usize;
    let rounds = timing["rounds"].as_u64().unwrap_or(5) as usize;
    let cold_mb = timing["cold_mb"].as_u64().unwrap_or(1024) as usize;
    let warmup_ms = timing["warmup_ms"].as_u64().unwrap_or(500);
    let kernel_only = timing["kernel_only"].as_bool().unwrap_or(true) && cell.rht;
    let audit_enabled = job["audit"].as_bool().unwrap_or(true);
    if cell.rht && !cell.n.is_multiple_of(32) {
        return Err("rht cells need n % 32 == 0".into());
    }
    if cell.m == 0 || cell.m > MAX_M {
        return Err(format!("cell.m must be 1..={MAX_M}"));
    }

    let mut input = QuantInput::<bf16>::new(cell.m, cell.k, cell.n, cell.group, cell.bits, cell.method(), cell.seed)
        .with_group_output();
    if cell.signed_codes {
        input = input.with_signed_weight_codes();
    }
    let mut rng = SmallRng::seed_from_u64(cell.seed ^ 0x5EED);
    let rht_factors: Vec<i32> = (0..cell.n)
        .map(|_| {
            if rng.random_range(0u8..2) == 0 {
                1
            } else {
                -1
            }
        })
        .collect();
    let bytes = input.weight_buffer_bytes();
    let copies = (cold_mb << 20).div_ceil(bytes).max(2);
    let pool: Vec<_> = (0..copies).map(|_| QuantBuffers::<Metal, bf16>::allocate(context, &input)).collect();
    let ys = (0..copies).map(|_| create_buffer::<Metal, bf16>(context, (cell.m * cell.n) as usize)).collect();
    let rht_buffer = create_buffer_with_data::<Metal, i32>(context, &rht_factors);
    let mut data = CellData {
        packed: None,
        input,
        pool,
        alt_pools: Vec::new(),
        ys,
        rht_factors,
        rht_buffer,
    };

    let shape_rht = data.shape(context, true, cell.m);
    let shape_plain = data.shape(context, false, cell.m);
    let mut entries: Vec<Entry> =
        candidates.iter().map(|candidate| resolve(candidate, cell, context, &shape_rht, &shape_plain)).collect();

    if entries.iter().any(|entry| entry.a_packed) {
        data.packed = Some(build_packed(context, &data, cell)?);
    }
    for entry in &entries {
        if entry.runner.is_some()
            && entry.layout != GemmBLayout::RowMajor
            && !data.alt_pools.iter().any(|(arm, _)| *arm == entry.layout)
        {
            let units = relayout_units(context, &data, cell);
            let pool = (0..data.pool.len())
                .map(|_| {
                    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(context, &data.input);
                    buffers.w = create_buffer_with_data::<Metal, u32>(context, &units);
                    buffers.weights_layout = entry.layout;
                    buffers
                })
                .collect();
            data.alt_pools.push((entry.layout, pool));
        }
    }

    // The strict per-row ULP check against the legacy M=1 GEMV only applies to GEMV candidates, and costs one command buffer per row.
    let legacy_needed = entries.iter().any(|entry| entry.runner.is_some() && entry.exact);
    let legacy_enabled = audit_enabled && legacy_needed && cell.m <= LEGACY_REFERENCE_MAX_M;
    let (reference_fused, reference_deferred) = if legacy_enabled {
        legacy_references(context, &mut data, cell)?
    } else {
        (Vec::new(), Vec::new())
    };
    // Cells without RHT use the plain legacy output (stored in `reference_fused`).

    // Audit every legal entry on sentinel-filled outputs.
    let mut audits: Vec<Option<Value>> = vec![None; entries.len()];
    let mut audit_pass = vec![false; entries.len()];
    // P2l: outputs of every later entry against the first legal entry (bit-exactness of relayouts).
    let mut first_actual: Option<Vec<bf16>> = None;
    for (index, entry) in entries.iter_mut().enumerate() {
        let Some(runner) = entry.runner.as_mut() else {
            continue;
        };
        if !audit_enabled {
            audit_pass[index] = true;
            continue;
        }
        let corrupt_index = data.pool.len();
        if entry.wrong_scale {
            // Real wrong answer on the GPU: an extra pool copy whose scales are all x17/16 (the CPU reference keeps the true ones).
            let mut bad = QuantBuffers::<Metal, bf16>::allocate(context, &data.input);
            let scales: Vec<bf16> = buffer_to_vec::<Metal, bf16>(context, &bad.scales)
                .into_iter()
                .map(|v| bf16::from_f32(v.to_f32() * 1.0625))
                .collect();
            bad.scales = create_buffer_with_data::<Metal, bf16>(context, &scales);
            data.pool.push(bad);
            data.ys.push(create_buffer::<Metal, bf16>(context, (cell.m * cell.n) as usize));
        }
        let audit_copy = if entry.wrong_scale {
            corrupt_index
        } else {
            0
        };
        let result = run_once(context, runner, &mut data, cell, audit_copy, entry.a_packed, entry.layout);
        if entry.wrong_scale {
            data.pool.pop();
            data.ys.pop();
        }
        let mut actual = match result {
            Ok(actual) => actual,
            Err(reason) => {
                entry.illegal = Some(format!("encode/dispatch failed: {reason}"));
                entry.runner = None;
                continue;
            },
        };
        if entry.inject_fault {
            // Corrupt the last row by x17/16 (>= 8 bf16 ULP) after the kernel, as a stand-in for a wrong tile.
            let n = cell.n as usize;
            let row = (cell.m as usize - 1) * n;
            actual[row..row + n].iter_mut().for_each(|v| *v = bf16::from_f32(v.to_f32() * 1.0625));
        }
        let (against, reference) = match entry.fused {
            Some(false) => ("deferred_legacy", &reference_deferred),
            Some(true) => ("fused_legacy", &reference_fused),
            None => ("plain_legacy", &reference_fused),
        };
        let tolerance = if entry.exact {
            CPU_TOLERANCE
        } else {
            CPU_TOLERANCE_MMA
        };
        let stats = if legacy_enabled {
            ulp_compare(&actual, reference, cell.n as usize)
        } else {
            // No legacy reference (batch > LEGACY_REFERENCE_MAX_M or a GEMM-only job): still reject unwritten / NaN outputs.
            let mut stats = UlpStats::default();
            for value in &actual {
                if value.to_bits() == SENTINEL_BITS {
                    stats.unwritten += 1;
                } else if value.is_nan() {
                    stats.nan += 1;
                }
            }
            stats
        };
        let vs_first = match &first_actual {
            None => {
                first_actual = Some(actual.clone());
                Value::Null
            },
            Some(first) => {
                let differing = actual.iter().zip(first).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                let stats = ulp_compare(&actual, first, cell.n as usize);
                json!({"differing": differing, "total": actual.len(), "max_ulp": stats.max_ulp})
            },
        };
        let cpu = cpu_audit(cell, &data, &actual, tolerance);
        let cpu_error = cpu.max_err;
        let pass = if entry.exact && legacy_enabled {
            stats.passes() && cpu_error <= tolerance
        } else {
            stats.unwritten == 0 && stats.nan == 0 && cpu_error <= tolerance
        };
        audit_pass[index] = pass;
        let mut audit = stats.to_json();
        audit["vs_first"] = vs_first;
        audit["pass"] = json!(pass);
        audit["reference"] = json!(if legacy_enabled {
            against
        } else {
            "f32_cpu_only"
        });
        audit["ulp_threshold"] = json!(ULP_PASS);
        audit["cpu_max_err"] = json!(if cpu_error.is_finite() {
            cpu_error
        } else {
            1.0e9
        });
        audit["cpu"] = cpu.to_json();
        audit["class"] = json!(if !entry.exact {
            "mma"
        } else if legacy_enabled {
            "exact"
        } else {
            "exact_cpu_only"
        });
        audit["cpu_tolerance"] = json!(tolerance);
        audit["fault_injected"] = json!(entry.inject_fault || entry.wrong_scale);
        audit["fault_kind"] = json!(if entry.wrong_scale {
            "wrong_scale"
        } else if entry.inject_fault {
            "host_last_row"
        } else {
            "none"
        });
        audits[index] = Some(audit);
    }

    // Timing: interleaved round-robin over slots (whole linear, kernel-only when the cell has an RHT, and in resident
    // mode the raw-pipeline control and resident slots).
    let resident_mode = job["weights"].as_str() == Some("resident");
    let mut slots: Vec<Slot> = Vec::new();
    let mut resident_notes: Vec<Option<String>> = vec![None; entries.len()];
    for index in 0..entries.len() {
        let mut kinds = Vec::new();
        if entries[index].runner.is_some() && audit_pass[index] {
            kinds.push(SlotKind::Whole);
            if kernel_only {
                kinds.push(SlotKind::KernelOnly);
            }
        }
        if resident_mode && entries[index].runner.is_some() {
            match entries[index].plain_spec {
                Some(spec) if spec.output_row_tile() >= 2 => {
                    let direct = build_direct(context, &spec);
                    resident_notes[index] = direct.as_ref().err().cloned();
                    if direct.is_ok() {
                        kinds.extend([SlotKind::DirectNormal, SlotKind::Resident]);
                    }
                    entries[index].direct = Some(direct);
                },
                _ => resident_notes[index] = Some("not a GEMV".into()),
            }
        }
        for (protocol, &chain) in protocols.iter().enumerate() {
            for &kind in &kinds {
                slots.push(Slot {
                    entry: index,
                    kind,
                    chain,
                    protocol,
                    samples: Vec::new(),
                    records: Vec::new(),
                    hold: None,
                });
            }
        }
    }
    let (mut guard_log, noisy_start) = host_guard(&job["guard"]);
    let mut driver = Driver {
        dispatches,
        queue: timing["pipeline"].as_str() == Some("queue"),
        forward: timing["order"].as_str() == Some("forward"),
        fixed_copy: timing["copy_mode"].as_str() == Some("fixed"),
        last_done_ns: 0,
        previous_slot: None,
    };
    let mut timing_error: Option<String> = None;
    if !slots.is_empty() && rounds > 0 {
        let started = Instant::now();
        let mut warm = 0usize;
        // Discarded warm-up rounds (every slot, so PSOs and clocks are hot), until `warmup_ms` of GPU work is issued.
        loop {
            let tag = format!("w{warm}");
            if let Err(error) = driver.round(context, &mut entries, &mut data, cell, &mut slots, warm, &tag, false) {
                timing_error = Some(error);
            }
            warm += 1;
            if timing_error.is_some() || started.elapsed().as_millis() as u64 >= warmup_ms {
                break;
            }
        }
        // Rounds rotate the start slot and walk odd rounds backwards (A B C / C B A: ABBA pairs) unless `order` is
        // `forward`, so position and clock-ramp bias cancel in the medians.
        for round in 0..rounds {
            if timing_error.is_some() {
                break;
            }
            let tag = format!("r{round}");
            if let Err(error) = driver.round(context, &mut entries, &mut data, cell, &mut slots, round, &tag, true) {
                timing_error = Some(error);
            }
        }
        if let Some(hold_ms) = job["profile_hold_ms"].as_u64().filter(|&ms| ms > 0) {
            for slot_index in 0..slots.len() {
                if timing_error.is_some() {
                    break;
                }
                match hold_slot(context, &mut entries, &mut data, cell, &slots[slot_index], dispatches, hold_ms) {
                    Ok(hold) => slots[slot_index].hold = Some(hold),
                    Err(error) => timing_error = Some(error),
                }
            }
        }
    }

    let end_snapshot = host_snapshot();
    let max_busy = guard_log["max_cpu_busy_pct"].as_f64().unwrap_or(10.0);
    let noisy_end = end_snapshot["cpu_busy_pct"].as_f64().is_some_and(|busy| busy > max_busy);
    guard_log["end"] = end_snapshot;
    let noisy_host = (noisy_start || noisy_end) && job["guard"]["enabled"].as_bool() != Some(false);

    let candidates_json: Vec<Value> = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let mut out = json!({"name": entry.name, "legal": entry.illegal.is_none(),
                "reason": entry.illegal, "resolved": entry.resolved,
                "rht": match (cell.rht, entry.fused) {(false, _) => "none", (true, Some(true)) => "fused",
                    (true, Some(false)) => "deferred", (true, None) => "unknown"},
                "audit": audits[index]});
            for slot in slots.iter().filter(|slot| slot.entry == index && !slot.samples.is_empty()) {
                let key = match slot.protocol {
                    0 => slot.kind.key().to_string(),
                    p => format!("{}_{}", slot.kind.key(), protocol_names[p]),
                };
                out[key] = timing_json(slot);
            }
            if resident_mode {
                out["resident_audit"] = json!("n/a: resident mode reads the same weight rows in every threadgroup");
                if let Some(note) = &resident_notes[index] {
                    out["resident_note"] = json!(note);
                }
            }
            if entry.runner.is_some() && !audit_pass[index] {
                out["timing"] = json!("skipped: audit failed");
            }
            out
        })
        .collect();

    Ok(json!({"cell": cell.to_json(), "cold_pool": {"copies": copies, "bytes_per_copy": bytes},
        "timing_error": timing_error, "noisy_host": noisy_host, "host_guard": guard_log,
        "candidates": candidates_json}))
}

#[uzu_test]
#[ignore]
fn autotune_runner() {
    let Ok(job_path) = std::env::var("AT_JOB") else {
        eprintln!("AT_JOB not set; nothing to do");
        return;
    };
    let job: Value = serde_json::from_slice(&std::fs::read(&job_path).expect("read AT_JOB")).expect("parse AT_JOB");
    let context = block_on(MetalContext::new()).expect("Metal context");
    let candidates: Vec<Candidate> = job["candidates"]
        .as_array()
        .expect("job.candidates")
        .iter()
        .map(|candidate| Candidate::parse(candidate).unwrap_or_else(|e| panic!("bad candidate: {e}")))
        .collect();
    let started = Instant::now();
    let load_start = load1();
    let results: Vec<Value> = job["cells"]
        .as_array()
        .expect("job.cells")
        .iter()
        .map(|cell_json| match Cell::parse(cell_json, &job) {
            Ok(cell) => run_cell(&context, &cell, &candidates, &job)
                .unwrap_or_else(|error| json!({"cell": cell.to_json(), "error": error})),
            Err(error) => json!({"cell": cell_json, "error": error}),
        })
        .collect();
    let output = json!({
        "schema": 1,
        "device": {"name": context.device_name, "gpu_family": newest_apple_gpu_family(&context),
            "gpu_cores": context.gpu_core_count, "supports_mxu": context.supports_mxu},
        "host_load1": {"start": load_start, "end": load1()},
        "noisy_host": results.iter().any(|cell| cell["noisy_host"].as_bool() == Some(true)),
        "timing": job["timing"], "protocols": protocol_list(&job["timing"]).0, "wall_s": started.elapsed().as_secs_f64(),
        "cells": results,
    });
    let text = serde_json::to_string_pretty(&output).unwrap();
    match std::env::var("AT_OUT") {
        Ok(path) => std::fs::write(&path, text).expect("write AT_OUT"),
        Err(_) => println!("{text}"),
    }
}
