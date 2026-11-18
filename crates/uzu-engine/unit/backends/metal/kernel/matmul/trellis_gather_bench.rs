use std::{collections::HashSet, ops::Range};

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Backend, Kernels,
            gpu_types::trellis::TrellisCodebook,
            kernel::matmul::{
                MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulGather, MatmulKernel, MatmulOutput, TrellisFormat,
            },
        },
        metal::{Metal, context::MetalContext},
    },
    data_type::DataType,
    tests::{
        cold_pool::ColdPool,
        helpers::{create_buffer, create_buffer_with_data},
        matmul::iter_encode_loop,
        util::{shared_metal_context, type_short_name},
    },
};

type MetalBuffer = <Metal as Backend>::GlobalGpuBuffer;
type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

// The weaver's candidate readout of Qwen3.5 9B: every tree node scores its candidates against the output embedding,
// one node in the first step of a round and four in each later step.
const VOCABULARY_SIZE: usize = 248_320;
const MODEL_DIMENSION: usize = 4096;
const CANDIDATE_COUNT: usize = 512;
const STEP_NODE_COUNTS: [usize; 2] = [1, 4];
const FORMAT: TrellisFormat = TrellisFormat {
    vector_width: 4,
    transition_bits: 7,
    restart_columns: Some(64),
};
const CODEBOOK: TrellisCodebook = TrellisCodebook {
    scale: 0.052_037_477,
    offsets: [-0.082_059_78, -0.077_588_536, -0.078_143_574, -0.081_098_95],
};

// The data follows the gather inputs sampled from a "Tell me about London" generation with that model.
const ACTIVATION_DEVIATION: f32 = 35.0;
const ACTIVATION_SCALES: Range<f32> = 0.026..0.05;
const ROW_SCALES: Range<f32> = 0.012..0.019;
/// Candidates the odd node of a step has in common with the others.
const COMMON_CANDIDATE_COUNT: usize = CANDIDATE_COUNT * 2 / 3;

/// Candidates concentrate on low token ids: half of them are below 6k and 90% below 42k.
fn candidate_id(rng: &mut SmallRng) -> u32 {
    let vocabulary_fraction: f64 = (0..4).map(|_| rng.random::<f64>()).product();
    (vocabulary_fraction * VOCABULARY_SIZE as f64) as u32
}

/// A node's candidates are distinct and in no particular id order. The row starts with `common`.
fn candidate_row(
    rng: &mut SmallRng,
    common: &[u32],
) -> Vec<u32> {
    let mut row = common.to_vec();
    let mut taken: HashSet<u32> = common.iter().copied().collect();
    while row.len() < CANDIDATE_COUNT {
        let id = candidate_id(rng);
        if taken.insert(id) {
            row.push(id);
        }
    }
    row
}

/// Nodes of a step mostly share a depth and with it their candidates; one node is at another depth, whose candidates
/// overlap by two thirds.
fn step_candidates(
    rng: &mut SmallRng,
    node_count: usize,
) -> Vec<u32> {
    let shared = candidate_row(rng, &[]);
    let mut candidates = shared.repeat(node_count);
    if node_count > 1 {
        let odd_node = candidates.len() - CANDIDATE_COUNT;
        candidates[odd_node..].copy_from_slice(&candidate_row(rng, &shared[..COMMON_CANDIDATE_COUNT]));
    }
    candidates
}

fn step_weight_row_count(node_count: usize) -> usize {
    if node_count > 1 {
        2 * CANDIDATE_COUNT - COMMON_CANDIDATE_COUNT
    } else {
        CANDIDATE_COUNT
    }
}

struct StepBuffers {
    activations: MetalBuffer,
    column_group_sums: MetalBuffer,
    activation_scales: MetalBuffer,
    candidates: MetalBuffer,
}

impl StepBuffers {
    fn allocate(
        context: &MetalContext,
        rng: &mut SmallRng,
        node_count: usize,
    ) -> Self {
        let distribution = Normal::new(0.0, ACTIVATION_DEVIATION).unwrap();
        let activations: Vec<i8> = (0..node_count * MODEL_DIMENSION)
            .map(|_| distribution.sample(rng).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let (nodes, _) = activations.as_chunks::<MODEL_DIMENSION>();
        let column_group_sums: Vec<f32> = nodes
            .iter()
            .flat_map(|node| {
                (0..4).map(|group| node.iter().skip(group).step_by(4).map(|&value| f32::from(value)).sum())
            })
            .collect();
        let activation_scales: Vec<f32> = (0..node_count).map(|_| rng.random_range(ACTIVATION_SCALES)).collect();
        Self {
            activations: create_buffer_with_data::<Metal, i8>(context, &activations),
            column_group_sums: create_buffer_with_data::<Metal, f32>(context, &column_group_sums),
            activation_scales: create_buffer_with_data::<Metal, f32>(context, &activation_scales),
            candidates: create_buffer_with_data::<Metal, u32>(context, &step_candidates(rng, node_count)),
        }
    }
}

#[uzu_bench]
fn bench_trellis_gather(c: &mut Criterion) {
    let context = shared_metal_context();
    let mut rng = SmallRng::seed_from_u64(42);
    let row_bytes = FORMAT.code_row_bytes(MODEL_DIMENSION as u32) as usize;
    let mut codes = vec![0u8; VOCABULARY_SIZE * row_bytes];
    rng.fill(&mut codes[..]);
    let codes = create_buffer_with_data::<Metal, u8>(&context, &codes);
    let row_scales: Vec<f32> = (0..VOCABULARY_SIZE).map(|_| rng.random_range(ROW_SCALES)).collect();
    let row_scales = create_buffer_with_data::<Metal, f32>(&context, &row_scales);
    let mut matmul =
        <MetalMatmul as MatmulKernel>::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).unwrap();

    let mut group =
        c.benchmark_group(format!("{}/Kernel/TrellisGather/Qwen35WeaverReadout", type_short_name::<Metal>()));
    for node_count in STEP_NODE_COUNTS {
        // A step reads only its candidates' rows, so that is what a copy keeps out of the cache. Hot ids recur across
        // steps as they do in the model.
        let mut steps = ColdPool::new(step_weight_row_count(node_count) * row_bytes, || {
            StepBuffers::allocate(&context, &mut rng, node_count)
        });
        // In the model a step waits for the kernel that prepares its activations. Writing one output makes the steps
        // wait for each other instead of overlapping.
        let mut output = create_buffer::<Metal, bf16>(&context, node_count * CANDIDATE_COUNT);
        group.throughput(Throughput::Elements((node_count * CANDIDATE_COUNT * MODEL_DIMENSION) as u64));
        let shape = format!("m{node_count}_k{MODEL_DIMENSION}_n{CANDIDATE_COUNT}");
        group.bench_function(BenchmarkId::from_parameter(shape), |bencher| {
            iter_encode_loop::<Metal, _>(&context, bencher, |command_buffer| {
                let step = steps.next_mut();
                let arguments = MatmulArguments {
                    a: MatmulA::Trellis {
                        values: &step.activations,
                        column_group_sums: &step.column_group_sums,
                        scales: &step.activation_scales,
                    },
                    b: MatmulB::Trellis {
                        codes: &codes,
                        row_scales: &row_scales,
                        codebook: CODEBOOK,
                        format: FORMAT,
                    },
                    b_leading_dimension: None,
                    b_transpose: true,
                    output: MatmulOutput::new(&mut output, MatmulDOps::none()),
                    gather: Some(MatmulGather {
                        indices: &step.candidates,
                        first_row: 0,
                        row_count: VOCABULARY_SIZE as u32,
                    }),
                    m: node_count as u32,
                    n: CANDIDATE_COUNT as u32,
                    k: MODEL_DIMENSION as u32,
                };
                matmul.encode(arguments, command_buffer).expect("encode gathered readout");
            });
        });
    }
    group.finish();
}
