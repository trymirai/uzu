#![cfg(backend = "metal")]

//! `Metal/Kernel/TrellisGemm` and `Metal/Kernel/TrellisGemv` — the trellis
//! weight format against the INT4 route it would replace, at prefill and at
//! decode batch widths.
//!
//! WHAT IS BEING COMPARED. The INT4 cell is the SHIPPED production dispatch —
//! g32 `ScaleZeroPoint` weights through `MatmulKernel::encode`, i.e. exactly
//! what a model runs today at these batch widths — and `trellis` is
//! `MatmulB::Trellis` through the same entry point. `bf16` is the dense
//! reference at the same shape. All cells of one `(shape, M)` are adjacent in
//! time and every conclusion is a within-`(shape, M)` ratio.
//!
//! THE GEMM CELLS PAY THE ACTIVATION QUANTIZE, the GEMV cells do not, because
//! that is what production does: `select_activation_format` keeps `M <= 8` on
//! bf16 and quantizes above it. `MatmulKernel::encode` does not own that pass,
//! so each GEMM cell encodes `ActivationTransform::encode_quantize` and then the
//! GEMM. The two GEMM cells differ in one place the format forces:
//! `ScaleZeroPoint` weights carry a zero point, so their quantize also emits
//! activation row sums; a trellis tape is symmetric and does not. The RHT is
//! excluded from both — trellis weights are fitted on rotated weights, so the
//! rotation is an upstream dispatch both formats need equally.
//!
//! PROTOCOL, because this machine punishes anything looser:
//!
//! * the caller leaves the GPU quiet (`ps -Ao comm= | grep -E 'uzu_engine-|gpudebug'`
//!   empty for >= 150 s) before every pass, and samples it continuously during;
//! * [`SETTLE`] of untimed dense GPU work before the first timed cell, so the
//!   clocks are off the post-idle boost. 45 s was not enough: round 7's first
//!   pass measured its start anchor 4.1% slower than its end anchor ten minutes
//!   later, i.e. the machine was still speeding up while the first cells ran;
//! * every weight buffer rotates through a [`ColdPool`], so no tape, weight
//!   block or dense matrix is cache-resident when a cell reads it;
//! * a [`CELL_COOLDOWN`] idle after every cell;
//! * a bf16 dense THERMAL ANCHOR first and last, named `anchor_bf16_start` /
//!   `anchor_bf16_end` so that a criterion filter listing the variants picks
//!   them up. If the two medians differ by more than 3% the pass drifted and is
//!   discarded, not averaged;
//! * every cell registered through [`cell`], which names its benchmark path so
//!   `UZU_CAPTURE_BENCH` can capture it.
//!
//! ```text
//! cargo bench -p uzu-engine --lib -- "Metal/Kernel/Trellis"
//! ```

use std::{
    sync::Arc,
    thread::sleep,
    time::{Duration, Instant},
};

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use half::bf16;
use uzu_engine_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Encoder,
            gpu_types::QuantizationMethod,
            kernel::{
                ActivationTransform, Kernels,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{
                    MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulShape,
                    trellis_format::{TrellisConfig, TrellisTape},
                },
            },
        },
        metal::{Metal, MetalContext, kernel::matmul::MatmulDispatch},
    },
    data_type::DataType,
    tests::{
        cold_pool::ColdPool,
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::{QuantBuffers, QuantInput, iter_encode_loop_named, quant_arguments},
        util::{shared_metal_context, type_short_name},
    },
};

// ---------------------------------------------------------------------------
// protocol
// ---------------------------------------------------------------------------

type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;
/// A weight buffer that rotates through enough copies to stay out of the
/// system level cache.
type Pool = ColdPool<Allocation<Metal>, Box<dyn FnMut() -> Allocation<Metal>>>;

const SETTLE: Duration = Duration::from_secs(120);
const CELL_COOLDOWN: Duration = Duration::from_secs(2);

fn cold_pool<T: ArrayElement>(
    context: &Arc<MetalContext>,
    data: Vec<T>,
) -> Pool {
    let bytes = size_of_val(data.as_slice());
    let pool_context = context.clone();
    ColdPool::new(bytes, Box::new(move || alloc_allocation_with_data::<Metal, T>(&pool_context, &data)))
}

/// Register one timed cell and cool down after it. `name` becomes both the
/// criterion parameter and the tail of the capture path.
fn cell<F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    context: &MetalContext,
    group_path: &str,
    name: &str,
    mut body: F,
) where
    F: FnMut(&mut Encoder<Metal>),
{
    group.bench_function(BenchmarkId::from_parameter(name), |bencher| {
        let benchmark_path = format!("{group_path}/{name}");
        iter_encode_loop_named::<Metal, _>(context, bencher, &benchmark_path, |encoder| body(encoder));
    });
    sleep(CELL_COOLDOWN);
}

/// Print, and assert, the dispatch a cell actually takes.
///
/// A plan table DERIVED from the same policy code the kernel runs is not
/// evidence. This reads the plan back through `MatmulKernel::select_dispatch`
/// for the exact arguments the cell encodes, so the tile, the engine and the
/// split-K in any report are the ones the GPU ran, and a cell that silently
/// routes to the other path fails here instead of being reported under the
/// wrong heading.
fn report_plan(
    context: &MetalContext,
    matmul: &MetalMatmul,
    label: &str,
    shape: &MatmulShape,
    expect_gemm: bool,
) {
    let dispatch = matmul.select_dispatch(shape, context);
    println!("PLAN {label}: {dispatch:?}");
    assert_eq!(matches!(dispatch, MatmulDispatch::Gemm(_)), expect_gemm, "{label} routed to the wrong path");
}

fn fill(
    count: usize,
    phase: usize,
) -> Vec<bf16> {
    (0..count).map(|i| bf16::from_f32((((i + phase) % 13) as f32) * 0.01 - 0.06)).collect()
}

/// A dense bf16 matmul at one shape, with the weight matrix rotating cold. Used
/// both as the anchor and as the 16-bits-per-weight reference.
struct DenseCell {
    weights: Pool,
    a: Allocation<Metal>,
    d: Allocation<Metal>,
    m: u32,
    n: u32,
    k: u32,
}

impl DenseCell {
    fn new(
        context: &Arc<MetalContext>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Self {
        Self {
            weights: cold_pool(context, fill((n * k) as usize, 5)),
            a: alloc_allocation_with_data::<Metal, bf16>(context, &fill((m * k) as usize, 0)),
            d: alloc_allocation::<Metal, bf16>(context, (m * n) as usize),
            m,
            n,
            k,
        }
    }

    fn arguments(&mut self) -> MatmulArguments<'_, '_, '_, Metal> {
        MatmulArguments {
            a: MatmulA::FullPrecision {
                values: &self.a,
                offset: 0,
            },
            b: MatmulB::FullPrecision {
                b: self.weights.next_mut(),
            },
            b_leading_dimension: None,
            b_transpose: true,
            d: &mut self.d,
            d_transform: MatmulDOps::none(),
            gather_indices: None,
            m: self.m,
            n: self.n,
            k: self.k,
        }
    }
}

/// The bf16 dense thermal anchor, registered first and last in a pass.
///
/// `settle_first` runs [`SETTLE`] of untimed dense work INSIDE the closure that
/// criterion only calls for a benchmark it actually selected — criterion filters
/// at the MEASUREMENT, not at the bench function, so a settle in the bench body
/// would burn two minutes of a shared GPU for a bench nobody asked for. It runs
/// ONCE: criterion invokes that closure again for every sample
/// (`iter_encode_loop_named` uses `iter_custom`), so an unguarded settle here
/// would run 21 times — 42 minutes, not two.
#[allow(clippy::too_many_arguments)]
fn anchor_cell(
    group: &mut BenchmarkGroup<'_, WallTime>,
    context: &Arc<MetalContext>,
    group_path: &str,
    name: &str,
    matmul: &mut MetalMatmul,
    m: u32,
    n: u32,
    k: u32,
    settle_first: bool,
) {
    let mut anchor = DenseCell::new(context, m, n, k);
    let mut settled = false;
    group.bench_function(BenchmarkId::from_parameter(name), |bencher| {
        if settle_first && !settled {
            settled = true;
            let mut settle = DenseCell::new(context, 128, n, k);
            let start = Instant::now();
            while start.elapsed() < SETTLE {
                let mut encoder = Encoder::<Metal>::new(context).unwrap();
                for _ in 0..16 {
                    matmul.encode(settle.arguments(), &mut encoder).expect("settle encode failed");
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            }
        }
        let benchmark_path = format!("{group_path}/{name}");
        iter_encode_loop_named::<Metal, _>(context, bencher, &benchmark_path, |encoder| {
            matmul.encode(anchor.arguments(), encoder).expect("bf16 anchor");
        });
    });
    sleep(CELL_COOLDOWN);
}

// ---------------------------------------------------------------------------
// the trellis cell
// ---------------------------------------------------------------------------

/// The config the prototype study benchmarked: `k = 3`, so 3 bits per weight
/// plus a 32-bit header per row.
const CONFIG: TrellisConfig = TrellisConfig::new(32, 3);

/// `(label, N, K)`, named after the Qwen3.6-27B matrices they come from: the
/// two starved N = 5120 matrices, the wide one, and the one the GEMM tile
/// target leaves completely unsplit.
const SHAPES: [(&str, u32, u32); 4] = [
    ("in_proj_16480x5120", 16480, 5120),
    ("mlp_down_5120x17408", 5120, 17408),
    ("out_proj_5120x6144", 5120, 6144),
    ("mlp_up_34816x5120", 34816, 5120),
];

/// The production INT4 weight group.
const GROUP_SIZE: u32 = 32;

/// The int8 activation-quantize pass, plus the identity RHT factors it reads.
struct ActivationQuantize {
    transform: ActivationTransform<Metal>,
    factors: Allocation<Metal>,
}

impl ActivationQuantize {
    /// `sum_group_size` is `None` for symmetric weights and
    /// `Some(min(weight_group, ACTIVATION_SCALE_GROUP_SIZE))` for weights
    /// carrying a zero point, whose GEMM reads activation row sums.
    fn new(
        context: &MetalContext,
        sum_group_size: Option<u32>,
        k: u32,
    ) -> Self {
        Self {
            transform: ActivationTransform::<Metal>::quantize(
                context,
                DataType::BF16,
                ACTIVATION_SCALE_GROUP_SIZE,
                sum_group_size,
            )
            .expect("activation quantize transform"),
            factors: alloc_allocation_with_data::<Metal, i32>(context, &vec![1i32; k as usize]),
        }
    }
}

/// The tape, its per-row scales, and the activations.
///
/// `quantize` is present exactly when the cell feeds the int8 GEMM; without it
/// the cell hands the GEMV bf16 activations, which is what production does
/// below `M = 9`. Only the tape rotates through the cold pool: it is the only
/// buffer big enough to matter.
struct TrellisCell {
    tapes: Pool,
    row_scales: Allocation<Metal>,
    quantize: Option<ActivationQuantize>,
    a_bf16: Allocation<Metal>,
    a_int8: Allocation<Metal>,
    a_scales: Allocation<Metal>,
    d: Allocation<Metal>,
    m: u32,
    n: u32,
    k: u32,
}

impl TrellisCell {
    fn new(
        context: &Arc<MetalContext>,
        m: u32,
        n: u32,
        k: u32,
        int8_activations: bool,
    ) -> Self {
        let words = TrellisTape::random(CONFIG, n, k, 0x1234_5EED).words;
        let row_scales: Vec<bf16> = (0..n).map(|r| bf16::from_f32(0.002 + 0.001 * ((r % 19) as f32) / 19.0)).collect();
        Self {
            tapes: cold_pool(context, words),
            row_scales: alloc_allocation_with_data::<Metal, bf16>(context, &row_scales),
            quantize: int8_activations.then(|| ActivationQuantize::new(context, None, k)),
            a_bf16: alloc_allocation_with_data::<Metal, bf16>(context, &fill((m * k) as usize, 0)),
            a_int8: alloc_allocation::<Metal, i8>(context, (m * k) as usize),
            a_scales: alloc_allocation::<Metal, f32>(context, (m * (k / ACTIVATION_SCALE_GROUP_SIZE)) as usize),
            d: alloc_allocation::<Metal, bf16>(context, (m * n) as usize),
            m,
            n,
            k,
        }
    }

    fn arguments(&mut self) -> MatmulArguments<'_, '_, '_, Metal> {
        let a = if self.quantize.is_some() {
            MatmulA::Int8Symmetric {
                values: &self.a_int8,
                scales: &self.a_scales,
                group_sums: None,
                group_size: ACTIVATION_SCALE_GROUP_SIZE,
            }
        } else {
            MatmulA::FullPrecision {
                values: &self.a_bf16,
                offset: 0,
            }
        };
        MatmulArguments {
            a,
            b: MatmulB::Trellis {
                b: self.tapes.next_mut(),
                scales: &self.row_scales,
                config: CONFIG,
            },
            b_leading_dimension: None,
            b_transpose: true,
            d: &mut self.d,
            d_transform: MatmulDOps::none(),
            gather_indices: None,
            m: self.m,
            n: self.n,
            k: self.k,
        }
    }

    fn shape(&mut self) -> MatmulShape {
        MatmulShape::from_arguments(&self.arguments())
    }

    /// The whole cell: the activation quantize where production pays one, then
    /// the matmul.
    fn encode(
        &mut self,
        matmul: &mut MetalMatmul,
        encoder: &mut Encoder<Metal>,
    ) {
        if let Some(quantize) = self.quantize.as_ref() {
            quantize.transform.encode_quantize(
                &self.a_bf16,
                &mut self.a_int8,
                &mut self.a_scales,
                None,
                &quantize.factors,
                self.m,
                self.k,
                encoder,
            );
        }
        matmul.encode(self.arguments(), encoder).expect("trellis matmul");
    }
}

// ---------------------------------------------------------------------------
// the two passes
// ---------------------------------------------------------------------------

/// Prefill widths: the trellis GEMM against the shipped a8w4 GEMM.
#[uzu_bench]
fn bench_trellis_gemm(c: &mut Criterion) {
    let context = shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    let group_path = format!("{}/Kernel/TrellisGemm", type_short_name::<Metal>());
    let mut matmul =
        MetalMatmul::new(&context, bf16::data_type(), bf16::data_type(), bf16::data_type()).expect("MatmulKernel");

    let mut group = c.benchmark_group(group_path.clone());
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    // in_proj at M = 32, the middle of the swept batch widths.
    let (anchor_label, anchor_n, anchor_k) = SHAPES[0];
    let anchor_m = 32u32;
    let anchor = |group: &mut BenchmarkGroup<'_, WallTime>, matmul: &mut MetalMatmul, edge: &str, settle: bool| {
        let name = format!("anchor_bf16_{edge}/{anchor_label}/M{anchor_m}");
        anchor_cell(group, &context, &group_path, &name, matmul, anchor_m, anchor_n, anchor_k, settle);
    };
    anchor(&mut group, &mut matmul, "start", true);

    for (label, n, k) in SHAPES {
        for m in [16u32, 32, 64] {
            {
                let mut trellis = TrellisCell::new(&context, m, n, k, true);
                report_plan(&context, &matmul, &format!("trellis/{label}/M{m}"), &trellis.shape(), true);
                cell(&mut group, &context, &group_path, &format!("trellis/{label}/M{m}"), |encoder| {
                    trellis.encode(&mut matmul, encoder);
                });
            }

            // The shipped INT4 dispatch, routed, with its own activation
            // quantize. `ScaleZeroPoint` weights read activation row sums, so
            // this pass emits them; `a8_activation_plan` sizes that group as
            // `min(weight_group, activation_group)`.
            {
                let sum_group_size = GROUP_SIZE.min(ACTIVATION_SCALE_GROUP_SIZE);
                let input = QuantInput::<bf16>::new(m, k, n, GROUP_SIZE, 4, QuantizationMethod::ScaleZeroPoint, 42)
                    .with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, Some(sum_group_size));
                let quantize = ActivationQuantize::new(&context, Some(sum_group_size), k);
                let mut pool = ColdPool::new(input.weight_buffer_bytes(), || {
                    QuantBuffers::<Metal, bf16>::allocate(&context, &input)
                });
                report_plan(
                    &context,
                    &matmul,
                    &format!("a8w4/{label}/M{m}"),
                    &MatmulShape::from_arguments(&quant_arguments(pool.next_mut(), &input)),
                    true,
                );
                cell(&mut group, &context, &group_path, &format!("a8w4/{label}/M{m}"), |encoder| {
                    let buffers = pool.next_mut();
                    quantize.transform.encode_quantize(
                        &buffers.x,
                        buffers.prepared_a.as_mut().expect("prepared activations"),
                        buffers.prepared_a_scales.as_mut().expect("prepared activation scales"),
                        buffers.prepared_a_group_sums.as_mut(),
                        &quantize.factors,
                        m,
                        k,
                        encoder,
                    );
                    matmul.encode(quant_arguments(buffers, &input), encoder).expect("a8w4 GEMM");
                });
            }

            {
                let mut dense = DenseCell::new(&context, m, n, k);
                cell(&mut group, &context, &group_path, &format!("bf16/{label}/M{m}"), |encoder| {
                    matmul.encode(dense.arguments(), encoder).expect("bf16 GEMM");
                });
            }
        }
    }

    anchor(&mut group, &mut matmul, "end", false);
}

/// Decode widths: the trellis GEMV against the shipped W4 zero-point G32 qmv
/// route — the one `qmv/routes.rs` is tuned for at these exact shapes, and the
/// number the trellis GEMV has to beat to be worth its ~9 instructions per
/// weight. `bf16_dense` is the bandwidth ceiling at 16 bits per weight.
#[uzu_bench]
fn bench_trellis_gemv(c: &mut Criterion) {
    let context = shared_metal_context();
    let group_path = format!("{}/Kernel/TrellisGemv", type_short_name::<Metal>());
    let mut matmul =
        MetalMatmul::new(&context, bf16::data_type(), bf16::data_type(), bf16::data_type()).expect("MatmulKernel");

    let mut group = c.benchmark_group(group_path.clone());
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let (anchor_label, anchor_n, anchor_k) = SHAPES[0];
    let anchor_m = 1u32;
    let anchor = |group: &mut BenchmarkGroup<'_, WallTime>, matmul: &mut MetalMatmul, edge: &str, settle: bool| {
        let name = format!("anchor_bf16_{edge}/{anchor_label}/M{anchor_m}");
        anchor_cell(group, &context, &group_path, &name, matmul, anchor_m, anchor_n, anchor_k, settle);
    };
    anchor(&mut group, &mut matmul, "start", true);

    for (label, n, k) in SHAPES {
        for m in [1u32, 2, 4, 6, 8] {
            {
                let mut trellis = TrellisCell::new(&context, m, n, k, false);
                report_plan(&context, &matmul, &format!("trellis/{label}/M{m}"), &trellis.shape(), false);
                cell(&mut group, &context, &group_path, &format!("trellis/{label}/M{m}"), |encoder| {
                    trellis.encode(&mut matmul, encoder);
                });
            }

            {
                let input = QuantInput::<bf16>::new(m, k, n, GROUP_SIZE, 4, QuantizationMethod::ScaleZeroPoint, 42);
                let mut pool = ColdPool::new(input.weight_buffer_bytes(), || {
                    QuantBuffers::<Metal, bf16>::allocate(&context, &input)
                });
                cell(&mut group, &context, &group_path, &format!("int4_auto/{label}/M{m}"), |encoder| {
                    matmul.encode(quant_arguments(pool.next_mut(), &input), encoder).expect("int4 GEMV");
                });
            }

            {
                let mut dense = DenseCell::new(&context, m, n, k);
                cell(&mut group, &context, &group_path, &format!("bf16_dense/{label}/M{m}"), |encoder| {
                    matmul.encode(dense.arguments(), encoder).expect("bf16 GEMV");
                });
            }
        }
    }

    anchor(&mut group, &mut matmul, "end", false);
}
