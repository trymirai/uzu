#![cfg(metal_backend)]

//! int8-activation (a8w8) matmul against a bf16-activation baseline (a16w8) reading
//! the *same* 8-bit quantized weights, so only the activation dtype differs.
//!
//! Two pipelines are measured so the dynamic-quantization cost reads off directly:
//!  * `*_gemm_*` encodes the GEMM alone, against activations prepared on the CPU up front;
//!  * `*_full_*` encodes the real pipeline - a8 pays the fused RHT+quantize kernel, bf16
//!    pays the in-place Hadamard. Prepare cost is `full - gemm`.
//!
//! The prepare step is independent of the weight quantization scheme, so the full pipeline
//! is measured on symmetric weights only; the delta applies to bias and zero-point equally.
//!
//! Measurement protocol (Apple GPUs both ramp clocks up and throttle down, so both
//! directions have to be handled):
//!  * warm-up is at least 500 ms per benchmark, to ramp clocks;
//!  * an idle cool-off separates every benchmark function (`UZU_A8W_COOLOFF_MS`);
//!  * a fixed canary shape (a8 sym m32 k2048 n3072, GEMM-only) is measured three times per
//!    group - start, middle, end - so every table can carry its drift as an error bar;
//!  * within a shape the a8 variants run sym, bias, zp, sym again, so the two sym readings
//!    bound drift where the variant-to-variant comparison actually lives.
//!
//! One criterion group is one shape, which is the unit long sweeps should be chunked
//! across processes with, e.g. `cargo bench -p backend-uzu --lib "A8W/w8/k2048_n3072"`.

use std::{env, thread, time::Duration};

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use half::bf16;
use proc_macros::{uzu_bench, uzu_test};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    backends::{
        common::{
            Allocation, Backend, Encoder,
            gpu_types::{
                HADAMARD_TRANSFORM_BLOCK_SIZE, HadamardTransformOrder, QuantizationMethod, QuantizationMode,
                gemm::GemmTiling,
            },
            kernel::{
                HadamardTransformKernel, Kernels, RHTQuantizeActivationsKernel,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{GemmDispatchPath, Metal, MetalContext, select_mxu_quant_tiling, select_split_k},
    },
    data_type::DataType,
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
        matmul::{QuantInput, iter_encode_loop_named},
        util::{shared_metal_context, type_short_name},
    },
};

type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;
type MetalPrepare = <<Metal as Backend>::Kernels as Kernels>::RHTQuantizeActivationsKernel;
type MetalHadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel;

/// 4-bit weights are not natively supported yet, so this sweep is 8-bit only.
const BITS: u32 = 8;

/// Weight quantization group size. 64 keeps every MXU tiling eligible
/// (`Tile128x128x256_Simdgroups4x4` caps at 64) and gives two 32-wide activation
/// chunks per weight group, which is what the correction path has to amortize over.
const WEIGHT_GROUP_SIZE: u32 = 64;

/// Five distinct aspect ratios, drawn from the Qwen3 layer shapes.
const SHAPES: &[(&str, usize, usize)] = &[
    ("k2048_n3072", 2048, 3072),
    ("k1024_n7168", 1024, 7168),
    ("k2048_n12288", 2048, 12288),
    ("k3584_n1024", 3584, 1024),
    ("k9216_n2560", 9216, 2560),
];

/// 16-64 is the speculative-decode band, 128-256 the transition, 512-2048 real prefill.
/// m < 16 is the GEMV regime; it should not be reaching GEMM at all and needs a separate
/// strategy, so it is deliberately absent.
const MS: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 2048];

/// Cheap fixed reference shape, re-measured three times per group to bound drift.
const CANARY_M: usize = 32;
const CANARY_K: usize = 2048;
const CANARY_N: usize = 3072;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Act {
    A8,
    Bf16,
}

/// Which weights the GEMM reads. `Dense` is the no-quantization-anywhere baseline: the
/// decision-relevant comparison for a8w8 is against *this*, not against bf16 activations
/// that are themselves paying to dequantize int8 weights.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Weights {
    Quantized,
    Dense,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pipeline {
    /// GEMM alone, against activations prepared on the CPU.
    GemmOnly,
    /// Activation prepare (a8: RHT+quantize, bf16: Hadamard) followed by the GEMM.
    Full,
}

#[derive(Clone, Copy)]
struct Variant {
    act: Act,
    weights: Weights,
    pipeline: Pipeline,
    method: QuantizationMethod,
    /// Distinguishes the trailing A/B/A repeat from the leading measurement.
    repeat: bool,
}

impl Variant {
    fn name(self) -> String {
        let act = match self.act {
            Act::A8 => "a8",
            Act::Bf16 => "bf16",
        };
        let pipeline = match self.pipeline {
            Pipeline::GemmOnly => "gemm",
            Pipeline::Full => "full",
        };
        let method = match self.method {
            QuantizationMethod::ScaleSymmetric => "sym",
            QuantizationMethod::ScaleBias => "bias",
            QuantizationMethod::ScaleZeroPoint => "zp",
        };
        let repeat = if self.repeat {
            "2"
        } else {
            ""
        };
        if self.weights == Weights::Dense {
            return format!("bf16w16_{pipeline}");
        }
        format!("{act}_{pipeline}_{method}{repeat}")
    }
}

const fn gemm(
    act: Act,
    method: QuantizationMethod,
) -> Variant {
    Variant {
        act,
        weights: Weights::Quantized,
        pipeline: Pipeline::GemmOnly,
        method,
        repeat: false,
    }
}

const fn dense(pipeline: Pipeline) -> Variant {
    Variant {
        act: Act::Bf16,
        weights: Weights::Dense,
        pipeline,
        method: QuantizationMethod::ScaleSymmetric,
        repeat: false,
    }
}

/// Order matters: the two a8 sym readings bracket bias and zp so their sym-to-sym delta
/// bounds drift across exactly the comparisons being made.
const VARIANTS: &[Variant] = &[
    gemm(Act::A8, QuantizationMethod::ScaleSymmetric),
    gemm(Act::A8, QuantizationMethod::ScaleBias),
    gemm(Act::A8, QuantizationMethod::ScaleZeroPoint),
    gemm(Act::Bf16, QuantizationMethod::ScaleSymmetric),
    gemm(Act::Bf16, QuantizationMethod::ScaleBias),
    gemm(Act::Bf16, QuantizationMethod::ScaleZeroPoint),
    // Prepare cost is scheme-independent, so the full pipeline runs on sym only.
    Variant {
        act: Act::A8,
        weights: Weights::Quantized,
        pipeline: Pipeline::Full,
        method: QuantizationMethod::ScaleSymmetric,
        repeat: false,
    },
    Variant {
        act: Act::Bf16,
        weights: Weights::Quantized,
        pipeline: Pipeline::Full,
        method: QuantizationMethod::ScaleSymmetric,
        repeat: false,
    },
    // Full pipeline for the asymmetric weight schemes too. Prep is scheme-independent so
    // these could be derived, but the A8W8-vs-Abf16W8 comparison is the one that drives
    // decisions and it should be measured rather than reconstructed.
    Variant {
        act: Act::A8,
        weights: Weights::Quantized,
        pipeline: Pipeline::Full,
        method: QuantizationMethod::ScaleZeroPoint,
        repeat: false,
    },
    Variant {
        act: Act::Bf16,
        weights: Weights::Quantized,
        pipeline: Pipeline::Full,
        method: QuantizationMethod::ScaleZeroPoint,
        repeat: false,
    },
    Variant {
        act: Act::A8,
        weights: Weights::Quantized,
        pipeline: Pipeline::Full,
        method: QuantizationMethod::ScaleBias,
        repeat: false,
    },
    Variant {
        act: Act::Bf16,
        weights: Weights::Quantized,
        pipeline: Pipeline::Full,
        method: QuantizationMethod::ScaleBias,
        repeat: false,
    },
    // No quantization anywhere: bf16 activations against bf16 weights.
    dense(Pipeline::GemmOnly),
    dense(Pipeline::Full),
    Variant {
        act: Act::A8,
        weights: Weights::Quantized,
        pipeline: Pipeline::GemmOnly,
        method: QuantizationMethod::ScaleSymmetric,
        repeat: true,
    },
];

const CANARY_VARIANT: Variant = gemm(Act::A8, QuantizationMethod::ScaleSymmetric);

fn cool_off() {
    let millis = env::var("UZU_A8W_COOLOFF_MS").ok().and_then(|value| value.parse::<u64>().ok()).unwrap_or(3_000);
    if millis > 0 {
        thread::sleep(Duration::from_millis(millis));
    }
}

/// Criterion only applies its filter at `bench_function` time, so a filtered chunk would
/// still pay full setup - weight generation plus a CPU quantization pass - for every shape
/// in the sweep. Pre-checking the filter against the ids this group would emit keeps a
/// single-group chunk from constructing all of them.
fn group_selected(group_name: &str) -> bool {
    let filters: Vec<String> = env::args().skip(1).filter(|arg| !arg.starts_with('-')).collect();
    if filters.is_empty() {
        return true;
    }
    let mut names: Vec<String> = VARIANTS.iter().map(|variant| variant.name()).collect();
    names.push(format!("{}_canary", CANARY_VARIANT.name()));
    filters.iter().any(|filter| {
        filter.contains(group_name)
            || names.iter().any(|name| {
                let id = format!("{group_name}/{name}");
                id.contains(filter.as_str()) || filter.contains(id.as_str())
            })
    })
}

/// Weights, activations and output for one (k, n), allocated once at the largest m and
/// reused for every smaller m. Row-major layouts mean a smaller m simply reads the leading
/// rows, so every variant sees byte-identical inputs.
struct ShapeData {
    weights: Allocation<Metal>,
    /// Same numerical codes as `weights`, stored as native two's-complement int8.
    signed_weights: Allocation<Metal>,
    scales: Allocation<Metal>,
    biases: Allocation<Metal>,
    zero_points: Allocation<Metal>,
    rht_factors: Allocation<Metal>,
    /// Pristine bf16 activations, used by the GEMM-only bf16 path.
    a_bf16: Allocation<Metal>,
    /// Scratch the full bf16 pipeline applies its in-place Hadamard to.
    a_working: Allocation<Metal>,
    /// CPU-prepared int8 activations; the full a8 pipeline overwrites these with the
    /// kernel's own output, which is numerically equivalent and timing-irrelevant.
    a_int8: Allocation<Metal>,
    a_scales: Allocation<Metal>,
    a_group_sums: Allocation<Metal>,
    weights_bf16: Allocation<Metal>,
    output: Allocation<Metal>,
    k: u32,
    n: u32,
    mode: QuantizationMode,
}

impl ShapeData {
    fn new(
        context: &MetalContext,
        max_m: usize,
        k: usize,
        n: usize,
        seed: u64,
    ) -> Self {
        // Symmetric weights are generated once; the bias and zero-point tables are added
        // alongside so all three B prologues read the *same* packed weights.
        let input =
            QuantInput::<bf16>::new(max_m, k, n, WEIGHT_GROUP_SIZE, BITS, QuantizationMethod::ScaleSymmetric, seed)
                .with_prepared_a();
        let groups = k.div_ceil(WEIGHT_GROUP_SIZE as usize);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x5EED_A800);
        let biases: Vec<bf16> = (0..n * groups).map(|_| bf16::from_f32(rng.random_range(-0.03f32..0.03f32))).collect();
        let zero_points: Vec<u8> = (0..n * groups).map(|_| rng.random_range(0u8..u8::MAX)).collect();
        let rht: Vec<i32> = (0..k)
            .map(|index| {
                if index.is_multiple_of(3) {
                    -1
                } else {
                    1
                }
            })
            .collect();
        let prepared = input.prepared_a.as_ref().expect("prepared int8 activations");
        let signed_weights: Vec<u32> = input.w_packed.iter().map(|word| word ^ 0x8080_8080).collect();

        Self {
            weights: alloc_allocation_with_data::<Metal, u32>(context, &input.w_packed),
            signed_weights: alloc_allocation_with_data::<Metal, u32>(context, &signed_weights),
            scales: alloc_allocation_with_data::<Metal, bf16>(context, &input.scales),
            biases: alloc_allocation_with_data::<Metal, bf16>(context, &biases),
            zero_points: alloc_allocation_with_data::<Metal, u8>(context, &zero_points),
            rht_factors: alloc_allocation_with_data::<Metal, i32>(context, &rht),
            a_bf16: alloc_allocation_with_data::<Metal, bf16>(context, &input.x),
            a_working: alloc_allocation_with_data::<Metal, bf16>(context, &input.x),
            a_int8: alloc_allocation_with_data::<Metal, i8>(context, &prepared.values),
            a_scales: alloc_allocation_with_data::<Metal, f32>(context, &prepared.scales),
            a_group_sums: alloc_allocation_with_data::<Metal, i32>(context, &prepared.group_sums),
            weights_bf16: alloc_allocation_with_data::<Metal, bf16>(
                context,
                &(0..n * k).map(|_| bf16::from_f32(rng.random_range(-0.1f32..0.1f32))).collect::<Vec<_>>(),
            ),
            output: alloc_allocation::<Metal, bf16>(context, max_m * n),
            k: k as u32,
            n: n as u32,
            mode: if BITS == 4 {
                QuantizationMode::U4
            } else {
                QuantizationMode::U8
            },
        }
    }

    fn arguments(
        &mut self,
        variant: Variant,
        m: u32,
    ) -> MatmulArguments<'_, '_, '_, Metal, &Allocation<Metal>> {
        self.arguments_with_weight_storage(variant, m, false)
    }

    fn arguments_with_weight_storage(
        &mut self,
        variant: Variant,
        m: u32,
        signed_w8_storage: bool,
    ) -> MatmulArguments<'_, '_, '_, Metal, &Allocation<Metal>> {
        if variant.weights == Weights::Dense {
            let a = match variant.pipeline {
                Pipeline::GemmOnly => &self.a_bf16,
                Pipeline::Full => &self.a_working,
            };
            return MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: a,
                    offset: 0,
                },
                b: MatmulB::FullPrecision {
                    b: &self.weights_bf16,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut self.output,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                m,
                n: self.n,
                k: self.k,
            };
        }
        let weights = if signed_w8_storage {
            &self.signed_weights
        } else {
            &self.weights
        };
        let b = match variant.method {
            QuantizationMethod::ScaleSymmetric => MatmulB::ScaleSymmetricDequant {
                b: weights,
                scales: &self.scales,
                mode: if signed_w8_storage {
                    QuantizationMode::I8
                } else {
                    self.mode
                },
                group_size: WEIGHT_GROUP_SIZE,
            },
            QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
                b: weights,
                scales: &self.scales,
                biases: &self.biases,
                mode: if signed_w8_storage {
                    QuantizationMode::I8
                } else {
                    self.mode
                },
                group_size: WEIGHT_GROUP_SIZE,
            },
            QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
                b: weights,
                scales: &self.scales,
                zero_points: &self.zero_points,
                mode: if signed_w8_storage {
                    QuantizationMode::I8
                } else {
                    self.mode
                },
                group_size: WEIGHT_GROUP_SIZE,
            },
        };
        let a = match (variant.act, variant.pipeline) {
            (Act::A8, _) => MatmulA::Int8Symmetric {
                values: &self.a_int8,
                scales: &self.a_scales,
                // Symmetric weights need no correction, so they never carry row sums.
                group_sums: (variant.method != QuantizationMethod::ScaleSymmetric).then_some(&self.a_group_sums),
            },
            (Act::Bf16, Pipeline::GemmOnly) => MatmulA::FullPrecision {
                values: &self.a_bf16,
                offset: 0,
            },
            (Act::Bf16, Pipeline::Full) => MatmulA::FullPrecision {
                values: &self.a_working,
                offset: 0,
            },
        };
        MatmulArguments {
            a,
            b,
            b_leading_dimension: None,
            b_transpose: true,
            d: &mut self.output,
            d_transform: MatmulDOps::none(),
            gather_indices: None,
            m,
            n: self.n,
            k: self.k,
        }
    }
}

struct BenchKernels {
    matmul: MetalMatmul,
    /// One per specialization: symmetric weights skip the row-sum reduction entirely.
    prepare: MetalPrepare,
    prepare_with_group_sums: MetalPrepare,
    hadamard: MetalHadamard,
}

fn encode_step(
    data: &mut ShapeData,
    kernels: &mut BenchKernels,
    variant: Variant,
    m: u32,
    encoder: &mut Encoder<Metal>,
) {
    if variant.pipeline == Pipeline::Full {
        match variant.act {
            Act::A8 => {
                let needs_group_sums = variant.method != QuantizationMethod::ScaleSymmetric;
                let prepare = if needs_group_sums {
                    &kernels.prepare_with_group_sums
                } else {
                    &kernels.prepare
                };
                prepare.encode(
                    &data.a_bf16,
                    &mut data.a_int8,
                    &mut data.a_scales,
                    needs_group_sums.then_some(&mut data.a_group_sums),
                    &data.rht_factors,
                    m,
                    data.k,
                    HADAMARD_TRANSFORM_BLOCK_SIZE as u32,
                    encoder,
                )
            },
            // In-place, matching a real bf16 pipeline. No activation copy: that is pure
            // overhead a production path would not pay.
            Act::Bf16 => {
                kernels.hadamard.encode(&mut data.a_working, &data.rht_factors, data.k, m, encoder);
            },
        }
    }
    let arguments = data.arguments(variant, m);
    kernels.matmul.gemm.encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder).expect("a8w gemm encode");
}

#[allow(clippy::too_many_arguments)]
fn bench_one(
    group: &mut BenchmarkGroup<'_, WallTime>,
    context: &MetalContext,
    data: &mut ShapeData,
    kernels: &mut BenchKernels,
    group_name: &str,
    function_name: &str,
    parameter: &str,
    variant: Variant,
    m: u32,
) {
    cool_off();
    group.bench_function(BenchmarkId::new(function_name, parameter), |bench| {
        let benchmark_path = format!("{group_name}/{function_name}/{parameter}");
        iter_encode_loop_named::<Metal, _>(context, bench, &benchmark_path, |encoder| {
            encode_step(data, kernels, variant, m, encoder);
        });
    });
}

fn bench_shape(
    c: &mut Criterion,
    context: &MetalContext,
    kernels: &mut BenchKernels,
    canary: &mut ShapeData,
    shape_label: &str,
    k: usize,
    n: usize,
) {
    let group_name = format!("{}/A8W/w{BITS}/{shape_label}", type_short_name::<Metal>());
    if !group_selected(&group_name) {
        return;
    }

    let max_m = *MS.iter().max().expect("non-empty m sweep");
    let mut data = ShapeData::new(context, max_m, k, n, 0xA8_00 ^ u64::from(BITS) ^ (k as u64) ^ (n as u64));

    let mut group = c.benchmark_group(group_name.as_str());
    group.sample_size(10);
    // The protocol's floor: a shorter warm-up measures a cold, low-clock GPU.
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(800));

    let canary_name = format!("{}_canary", CANARY_VARIANT.name());
    let run_canary = |group: &mut BenchmarkGroup<'_, WallTime>,
                      canary: &mut ShapeData,
                      kernels: &mut BenchKernels,
                      position: &str| {
        bench_one(
            group,
            context,
            canary,
            kernels,
            &group_name,
            &canary_name,
            position,
            CANARY_VARIANT,
            CANARY_M as u32,
        );
    };

    run_canary(&mut group, canary, kernels, "start");

    for (index, &m) in MS.iter().enumerate() {
        let parameter = format!("m{m}");
        for &variant in VARIANTS {
            bench_one(
                &mut group,
                context,
                &mut data,
                kernels,
                &group_name,
                &variant.name(),
                &parameter,
                variant,
                m as u32,
            );
        }
        if index == MS.len() / 2 {
            run_canary(&mut group, canary, kernels, "mid");
        }
    }

    run_canary(&mut group, canary, kernels, "end");
    group.finish();
}

#[uzu_bench]
fn bench_a8w(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }

    let mut kernels = BenchKernels {
        matmul: <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
            .expect("matmul kernel"),
        prepare: <MetalPrepare as RHTQuantizeActivationsKernel>::new(&context, DataType::BF16, false)
            .expect("prepare kernel"),
        prepare_with_group_sums: <MetalPrepare as RHTQuantizeActivationsKernel>::new(&context, DataType::BF16, true)
            .expect("prepare kernel with row sums"),
        hadamard: <MetalHadamard as HadamardTransformKernel>::new(
            &context,
            DataType::BF16,
            HadamardTransformOrder::Input,
        )
        .expect("hadamard kernel"),
    };

    let selected: Vec<_> = SHAPES
        .iter()
        .filter(|(shape_label, _, _)| {
            group_selected(&format!("{}/A8W/w{BITS}/{shape_label}", type_short_name::<Metal>()))
        })
        .collect();
    if selected.is_empty() {
        return;
    }
    let mut canary = ShapeData::new(&context, CANARY_M, CANARY_K, CANARY_N, 0xCA_11A2);

    for &(shape_label, k, n) in selected {
        bench_shape(c, &context, &mut kernels, &mut canary, shape_label, k, n);
    }
}

// ---------------------------------------------------------------------------
// Dispatch-knob sweeps.
//
// a8 currently inherits the bf16-tuned tiling table (`select_mxu_tiling`) and a
// split-k target 4x lower than the bf16 path's, neither of which was chosen with
// int8 activations in mind. These sweeps pin each knob via the test-only overrides
// so the heuristics can be rebuilt from measurements.
//
// One criterion group is one (shape, knob value), which keeps a chunk to 19
// benchmarks and lets process isolation sit between knob values.
// ---------------------------------------------------------------------------

/// Knob values worth sweeping. Only sym and zp are measured: they bracket the
/// register pressure, and bias sits between them.
const SWEEP_METHODS: &[QuantizationMethod] = &[QuantizationMethod::ScaleSymmetric, QuantizationMethod::ScaleZeroPoint];

const MXU_TILINGS: &[(GemmTiling, &str)] = &[
    (GemmTiling::Tile16x32x256_Simdgroups1x1, "t16x32_1x1"),
    (GemmTiling::Tile16x128x256_Simdgroups1x4, "t16x128_1x4"),
    (GemmTiling::Tile32x64x256_Simdgroups2x2, "t32x64_2x2"),
    (GemmTiling::Tile64x32x256_Simdgroups4x1, "t64x32_4x1"),
    (GemmTiling::Tile64x64x256_Simdgroups2x2, "t64x64_2x2"),
    (GemmTiling::Tile128x128x256_Simdgroups4x4, "t128x128_4x4"),
];

/// 128 is the current int8 value, 512 the bf16 one.
const SPLIT_K_TARGETS: &[u32] = &[128, 256, 512, 1024];

#[derive(Clone, Copy)]
enum Knob {
    Tiling(GemmTiling),
    SplitKTarget(u32),
}

impl Knob {
    fn label(self) -> String {
        match self {
            Knob::Tiling(tiling) => MXU_TILINGS
                .iter()
                .find(|(candidate, _)| *candidate == tiling)
                .map(|(_, name)| (*name).to_owned())
                .expect("tiling has a sweep label"),
            Knob::SplitKTarget(target) => format!("target{target}"),
        }
    }

    /// Combinations the dispatch path would reject are skipped rather than measured.
    fn valid(self) -> bool {
        match self {
            Knob::Tiling(tiling) => tiling.fits_quant_group_size(WEIGHT_GROUP_SIZE),
            Knob::SplitKTarget(_) => true,
        }
    }

    fn apply(
        self,
        kernels: &mut BenchKernels,
    ) {
        match self {
            Knob::Tiling(tiling) => kernels.matmul.gemm.tiling_override = Some(tiling),
            Knob::SplitKTarget(target) => kernels.matmul.gemm.split_k_target_override = Some(target),
        }
    }

    fn clear(kernels: &mut BenchKernels) {
        kernels.matmul.gemm.tiling_override = None;
        kernels.matmul.gemm.split_k_target_override = None;
    }
}

fn knob_group_name(
    family: &str,
    shape_label: &str,
) -> String {
    format!("{}/{family}/w{BITS}/{shape_label}", type_short_name::<Metal>())
}

fn method_suffix(method: QuantizationMethod) -> &'static str {
    match method {
        QuantizationMethod::ScaleSymmetric => "sym",
        QuantizationMethod::ScaleBias => "bias",
        QuantizationMethod::ScaleZeroPoint => "zp",
    }
}

fn bench_knob_shape(
    c: &mut Criterion,
    context: &MetalContext,
    kernels: &mut BenchKernels,
    canary: &mut ShapeData,
    family: &str,
    shape_label: &str,
    k: usize,
    n: usize,
    knobs: &[Knob],
) {
    let group_name = knob_group_name(family, shape_label);
    if !group_selected(&group_name) {
        return;
    }
    let selected: Vec<Knob> = knobs
        .iter()
        .copied()
        .filter(|knob| {
            let ok = knob.valid();
            if !ok {
                println!("skipping {} on {shape_label}: rejected by dispatch constraints", knob.label());
            }
            ok
        })
        .collect();
    if selected.is_empty() {
        return;
    }

    let max_m = *MS.iter().max().expect("non-empty m sweep");
    let mut data = ShapeData::new(context, max_m, k, n, 0xA8_00 ^ u64::from(BITS) ^ (k as u64) ^ (n as u64));

    let mut group = c.benchmark_group(group_name.as_str());
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(800));

    let canary_name = format!("{}_canary", CANARY_VARIANT.name());
    let run_canary = |group: &mut BenchmarkGroup<'_, WallTime>,
                      canary: &mut ShapeData,
                      kernels: &mut BenchKernels,
                      position: &str| {
        Knob::clear(kernels);
        bench_one(
            group,
            context,
            canary,
            kernels,
            &group_name,
            &canary_name,
            position,
            CANARY_VARIANT,
            CANARY_M as u32,
        );
    };

    run_canary(&mut group, canary, kernels, "start");

    // Every candidate for one (m, method) is measured back to back in a single process,
    // bracketed by the default dispatch. Candidates measured in *separate* processes proved
    // incomparable: transients of 20-70% hit individual benchmarks without moving the
    // canary, so only same-process adjacency makes the ordering trustworthy. The two
    // `default` readings bound drift across the candidates between them.
    for (index, &m) in MS.iter().enumerate() {
        let parameter = format!("m{m}");
        for &method in SWEEP_METHODS {
            let variant = gemm(Act::A8, method);
            let suffix = method_suffix(method);
            for (name, knob) in std::iter::once((format!("default_{suffix}"), None))
                .chain(selected.iter().map(|knob| (format!("{}_{suffix}", knob.label()), Some(*knob))))
                .chain(std::iter::once((format!("default2_{suffix}"), None)))
            {
                match knob {
                    Some(knob) => knob.apply(kernels),
                    None => Knob::clear(kernels),
                }
                bench_one(&mut group, context, &mut data, kernels, &group_name, &name, &parameter, variant, m as u32);
                Knob::clear(kernels);
            }
        }
        if index == MS.len() / 2 {
            run_canary(&mut group, canary, kernels, "mid");
        }
    }

    run_canary(&mut group, canary, kernels, "end");
    group.finish();
}

fn bench_knobs(
    c: &mut Criterion,
    family: &str,
    knobs: Vec<Knob>,
) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }
    let any_selected = SHAPES.iter().any(|(shape_label, _, _)| group_selected(&knob_group_name(family, shape_label)));
    if !any_selected {
        return;
    }

    let mut kernels = BenchKernels {
        matmul: <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
            .expect("matmul kernel"),
        prepare: <MetalPrepare as RHTQuantizeActivationsKernel>::new(&context, DataType::BF16, false)
            .expect("prepare kernel"),
        prepare_with_group_sums: <MetalPrepare as RHTQuantizeActivationsKernel>::new(&context, DataType::BF16, true)
            .expect("prepare kernel with row sums"),
        hadamard: <MetalHadamard as HadamardTransformKernel>::new(
            &context,
            DataType::BF16,
            HadamardTransformOrder::Input,
        )
        .expect("hadamard kernel"),
    };
    let mut canary = ShapeData::new(&context, CANARY_M, CANARY_K, CANARY_N, 0xCA_11A2);

    for &(shape_label, k, n) in SHAPES {
        bench_knob_shape(c, &context, &mut kernels, &mut canary, family, shape_label, k, n, &knobs);
    }
}

#[uzu_bench]
fn bench_a8w_tiling(c: &mut Criterion) {
    bench_knobs(c, "A8WTile", MXU_TILINGS.iter().map(|(tiling, _)| Knob::Tiling(*tiling)).collect());
}

#[uzu_bench]
fn bench_a8w_split_k(c: &mut Criterion) {
    bench_knobs(c, "A8WSplitK", SPLIT_K_TARGETS.iter().map(|target| Knob::SplitKTarget(*target)).collect());
}

// ---------------------------------------------------------------------------
// Native signed-W8 storage ablation.
//
// The existing format stores offset-binary bytes and XORs every loaded W8 code with
// 0x80 in the hot loop. The candidate stores those same codes as two's-complement int8
// once, so the fragment load is directly usable by the MXU. Keep this deliberately tiny:
// one representative shape, the trace's m=32 point, and a compute-heavy m=2048 point.
// ---------------------------------------------------------------------------

const SIGNED_W8_MS: &[usize] = &[32, 2048];
const SIGNED_W8_METHODS: &[QuantizationMethod] =
    &[QuantizationMethod::ScaleSymmetric, QuantizationMethod::ScaleBias, QuantizationMethod::ScaleZeroPoint];

#[uzu_bench]
fn bench_a8w_signed_w8(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }
    let group_name = format!("{}/A8WSignedW8/w{BITS}/k2048_n3072", type_short_name::<Metal>());
    if !group_selected(&group_name) {
        return;
    }

    let mut kernels = BenchKernels {
        matmul: <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
            .expect("matmul kernel"),
        prepare: <MetalPrepare as RHTQuantizeActivationsKernel>::new(&context, DataType::BF16, false)
            .expect("prepare kernel"),
        prepare_with_group_sums: <MetalPrepare as RHTQuantizeActivationsKernel>::new(&context, DataType::BF16, true)
            .expect("prepare kernel with row sums"),
        hadamard: <MetalHadamard as HadamardTransformKernel>::new(
            &context,
            DataType::BF16,
            HadamardTransformOrder::Input,
        )
        .expect("hadamard kernel"),
    };
    let mut data = ShapeData::new(&context, 2048, 2048, 3072, 0x51_6E_ED);
    let mut group = c.benchmark_group(group_name.as_str());
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(800));

    for &m in SIGNED_W8_MS {
        for &method in SIGNED_W8_METHODS {
            let variant = gemm(Act::A8, method);
            let suffix = method_suffix(method);
            for (name, signed, fused_epilogue) in [
                ("unsigned", false, false),
                ("signed", true, false),
                ("fused", true, true),
                ("signed2", true, false),
                ("unsigned2", false, false),
            ] {
                cool_off();
                group.bench_function(BenchmarkId::new(format!("{name}_{suffix}"), format!("m{m}")), |bench| {
                    let path = format!("{group_name}/{name}_{suffix}/m{m}");
                    iter_encode_loop_named::<Metal, _>(&context, bench, &path, |encoder| {
                        kernels.matmul.gemm.fused_a8_epilogue_override = Some(fused_epilogue);
                        let arguments = data.arguments_with_weight_storage(variant, m as u32, signed);
                        kernels
                            .matmul
                            .gemm
                            .encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder)
                            .expect("signed-W8 ablation encode");
                        kernels.matmul.gemm.fused_a8_epilogue_override = None;
                    });
                });
            }
        }
    }
    group.finish();
}

#[uzu_bench]
fn bench_signed_w8_other_consumers(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }
    let group_name = format!("{}/SignedW8Consumers/w{BITS}/k2048_n3072", type_short_name::<Metal>());
    if !group_selected(&group_name) {
        return;
    }

    let mut matmul = <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let mut data = ShapeData::new(&context, 32, 2048, 3072, 0x51_6E_ED);
    let mut group = c.benchmark_group(group_name.as_str());
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(400));
    group.measurement_time(Duration::from_millis(500));

    for (consumer, m) in [("gemv", 1), ("a16_mxu", 32)] {
        for method in [QuantizationMethod::ScaleSymmetric, QuantizationMethod::ScaleBias] {
            let variant = gemm(Act::Bf16, method);
            let suffix = method_suffix(method);
            for (name, signed) in [("unsigned", false), ("signed", true), ("unsigned2", false)] {
                cool_off();
                group.bench_function(
                    BenchmarkId::new(format!("{consumer}_{name}_{suffix}"), format!("m{m}")),
                    |bench| {
                        let path = format!("{group_name}/{consumer}_{name}_{suffix}/m{m}");
                        iter_encode_loop_named::<Metal, _>(&context, bench, &path, |encoder| {
                            let arguments = data.arguments_with_weight_storage(variant, m, signed);
                            if m == 1 {
                                matmul.encode(arguments, encoder).expect("signed-W8 GEMV encode");
                            } else {
                                matmul
                                    .gemm
                                    .encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder)
                                    .expect("signed-W8 A16 MXU encode");
                            }
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

#[uzu_test]
fn signed_w8_storage_is_bit_exact() {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }
    let mut matmul = <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let mut data = ShapeData::new(&context, 32, 256, 64, 0x51_6E_ED);
    let tiling = select_mxu_quant_tiling(32, 64, WEIGHT_GROUP_SIZE, true);

    assert_eq!(select_split_k(32, 64, 256, tiling, true, WEIGHT_GROUP_SIZE, false, false, 1), 1);
    assert!(select_split_k(32, 64, 256, tiling, true, WEIGHT_GROUP_SIZE, false, false, 256) > 1);

    for (split_mode, split_k_target) in [("non-split", 1), ("split", 256)] {
        matmul.gemm.split_k_target_override = Some(split_k_target);
        for &method in SIGNED_W8_METHODS {
            let variant = gemm(Act::A8, method);
            let mut run = |signed: bool, fused_epilogue: bool| {
                matmul.gemm.fused_a8_epilogue_override = Some(fused_epilogue);
                let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                let arguments = data.arguments_with_weight_storage(variant, 32, signed);
                matmul
                    .gemm
                    .encode_dispatch_path(arguments, GemmDispatchPath::Mxu, &mut encoder)
                    .expect("signed-W8 correctness encode");
                encoder.end_encoding().submit().wait_until_completed().expect("signed-W8 correctness submit");
                allocation_to_vec::<Metal, bf16>(&data.output)
            };

            let unsigned = run(false, false);
            let signed = run(true, false);
            let fused = run(true, true);
            assert_eq!(unsigned, signed, "signed-W8 mismatch for {method:?} in {split_mode} mode");
            assert_eq!(signed, fused, "fused A8 epilogue mismatch for {method:?} in {split_mode} mode");
        }
    }

    matmul.gemm.fused_a8_epilogue_override = None;
    matmul.gemm.split_k_target_override = None;
}

#[uzu_bench]
fn bench_a8w_fused_epilogue_cutoff(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }
    let group_name = format!("{}/A8WFusedCutoff/w{BITS}/k2048_n3072", type_short_name::<Metal>());
    if !group_selected(&group_name) {
        return;
    }

    let mut matmul = <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let mut data = ShapeData::new(&context, 512, 2048, 3072, 0xF0_5E_D0);
    let variant = gemm(Act::A8, QuantizationMethod::ScaleSymmetric);
    let mut group = c.benchmark_group(group_name.as_str());
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(800));

    for m in [16, 64, 128, 256, 512] {
        for (name, fused_epilogue) in [("signed", false), ("fused", true), ("signed2", false)] {
            cool_off();
            group.bench_function(BenchmarkId::new(name, format!("m{m}")), |bench| {
                let path = format!("{group_name}/{name}/m{m}");
                iter_encode_loop_named::<Metal, _>(&context, bench, &path, |encoder| {
                    matmul.gemm.fused_a8_epilogue_override = Some(fused_epilogue);
                    let arguments = data.arguments_with_weight_storage(variant, m, true);
                    matmul
                        .gemm
                        .encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder)
                        .expect("fused-epilogue cutoff encode");
                    matmul.gemm.fused_a8_epilogue_override = None;
                });
            });
        }
    }
    group.finish();
}

#[uzu_bench]
fn bench_a8w_optimization_confirmation(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }

    for (shape_label, k, n) in [("k2048_n3072", 2048, 3072), ("k3584_n1024", 3584, 1024)] {
        let group_name = format!("{}/A8WConfirm/w{BITS}/{shape_label}", type_short_name::<Metal>());
        if !group_selected(&group_name) {
            continue;
        }
        let mut matmul = <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
            .expect("matmul kernel");
        let mut data = ShapeData::new(&context, 2048, k, n, 0xC0_6F_12 ^ k as u64 ^ n as u64);
        let mut group = c.benchmark_group(group_name.as_str());
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(400));
        group.measurement_time(Duration::from_millis(500));

        for m in [16, 32, 128, 2048] {
            if env::var("UZU_A8W_CONFIRM_M")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .is_some_and(|only| only != m)
            {
                continue;
            }
            for &method in SIGNED_W8_METHODS {
                let suffix = method_suffix(method);
                if env::var("UZU_A8W_CONFIRM_METHOD").is_ok_and(|only| only != suffix) {
                    continue;
                }
                let variant = gemm(Act::A8, method);
                let candidates: &[(&str, bool, bool)] = if m <= 32 {
                    &[
                        ("unsigned", false, false),
                        ("signed", true, false),
                        ("fused", true, true),
                        ("signed2", true, false),
                        ("unsigned2", false, false),
                    ]
                } else {
                    &[("unsigned", false, false), ("signed", true, false), ("unsigned2", false, false)]
                };
                for &(name, signed, fused_epilogue) in candidates {
                    cool_off();
                    group.bench_function(BenchmarkId::new(format!("{name}_{suffix}"), format!("m{m}")), |bench| {
                        let path = format!("{group_name}/{name}_{suffix}/m{m}");
                        iter_encode_loop_named::<Metal, _>(&context, bench, &path, |encoder| {
                            matmul.gemm.fused_a8_epilogue_override = Some(fused_epilogue);
                            let arguments = data.arguments_with_weight_storage(variant, m, signed);
                            matmul
                                .gemm
                                .encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder)
                                .expect("A8 optimization confirmation encode");
                            matmul.gemm.fused_a8_epilogue_override = None;
                        });
                    });
                }
            }
        }
        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Compute roofline.
//
// A large square GEMM is unambiguously compute-bound, so achieved TOP/s here is the
// empirical ceiling to judge the shaped results against. Three variants separate the
// two things that get conflated when comparing a8w8 to a16w8:
//   * a8_sym    - int8 MMA, dequant applied per output element in the epilogue;
//   * bf16_w8   - bf16 MMA, weights dequantized into threadgroup memory first;
//   * bf16_w16  - bf16 MMA on bf16 weights, no dequantization anywhere.
// The last is the raw MXU bf16 rate, so bf16_w16 vs a8_sym is the dtype ratio proper,
// while bf16_w8 vs a8_sym is what the shaped benchmarks actually measure.
// ---------------------------------------------------------------------------

const ROOFLINE_DIM: usize = 4096;

#[uzu_bench]
fn bench_a8w_roofline(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu() {
        return;
    }
    let group_name = format!("{}/A8WRoofline/w{BITS}", type_short_name::<Metal>());
    if !group_selected(&group_name) {
        return;
    }

    let dim = ROOFLINE_DIM;
    let mut matmul = <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16)
        .expect("matmul kernel");
    let mut data = ShapeData::new(&context, dim, dim, dim, 0x0000_4001);
    let mut rng = SmallRng::seed_from_u64(0x5EED_400F);
    let dense: Vec<bf16> = (0..dim * dim).map(|_| bf16::from_f32(rng.random_range(-0.1f32..0.1f32))).collect();
    let weights_bf16 = alloc_allocation_with_data::<Metal, bf16>(&context, &dense);

    let mut group = c.benchmark_group(group_name.as_str());
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(1500));

    for name in ["a8_sym", "bf16_w8", "bf16_w16"] {
        cool_off();
        group.bench_function(BenchmarkId::new(name, format!("m{dim}")), |bench| {
            let path = format!("{group_name}/{name}/m{dim}");
            iter_encode_loop_named::<Metal, _>(&context, bench, &path, |encoder| {
                if name == "bf16_w16" {
                    let arguments: MatmulArguments<'_, '_, '_, Metal, &Allocation<Metal>> = MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &data.a_bf16,
                            offset: 0,
                        },
                        b: MatmulB::FullPrecision {
                            b: &weights_bf16,
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut data.output,
                        d_transform: MatmulDOps::none(),
                        gather_indices: None,
                        m: dim as u32,
                        n: dim as u32,
                        k: dim as u32,
                    };
                    matmul
                        .gemm
                        .encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder)
                        .expect("roofline bf16 encode");
                } else {
                    let act = if name == "a8_sym" {
                        Act::A8
                    } else {
                        Act::Bf16
                    };
                    let variant = gemm(act, QuantizationMethod::ScaleSymmetric);
                    let arguments = data.arguments(variant, dim as u32);
                    matmul
                        .gemm
                        .encode_dispatch_path(arguments, GemmDispatchPath::Mxu, encoder)
                        .expect("roofline encode");
                }
            });
        });
    }
    group.finish();
}
