use half::{bf16, f16};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::MiraiSEmbeddingLookupKernel},
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
};

const VOCAB_SIZE: u32 = 40;
const MODEL_DIM: u32 = 256;

/// Looks `token_ids` up in the random table of `seed` on backend `B`.
fn lookup<B: Backend>(
    seed: u64,
    token_ids: &[u32],
) -> Vec<bf16> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let codes: Vec<u8> = (0..VOCAB_SIZE * MODEL_DIM / 4).map(|_| rng.random()).collect();
    let row_scales: Vec<bf16> = (0..VOCAB_SIZE).map(|_| bf16::from_f32(rng.random_range(0.01f32..0.1))).collect();
    let ladder_indices: Vec<u8> = (0..VOCAB_SIZE * MODEL_DIM / 128).map(|_| rng.random()).collect();
    let ladder: Vec<f16> = (0..16).map(|index| f16::from_f32(2.0f32.powf(index as f32 / 2.0 - 5.5))).collect();
    let table: Vec<i8> = (0..256 * 4).map(|_| rng.random_range(-4i8..=4)).collect();
    let factors: Vec<i32> = (0..MODEL_DIM).map(|_| [1, -1][rng.random_range(0..2)]).collect();
    let context = B::Context::new().unwrap();
    let kernel = <B::Kernels as Kernels>::MiraiSEmbeddingLookupKernel::new(&context).unwrap();
    let batch = token_ids.len() as u32;
    let mut output = alloc_allocation::<B, bf16>(&context, (batch * MODEL_DIM) as usize);
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &alloc_allocation_with_data::<B, u32>(&context, token_ids),
        &alloc_allocation_with_data::<B, u8>(&context, &codes),
        &alloc_allocation_with_data::<B, bf16>(&context, &row_scales),
        &alloc_allocation_with_data::<B, u8>(&context, &ladder_indices),
        &alloc_allocation_with_data::<B, f16>(&context, &ladder),
        &alloc_allocation_with_data::<B, i8>(&context, &table),
        &alloc_allocation_with_data::<B, i32>(&context, &factors),
        &mut output,
        batch,
        VOCAB_SIZE,
        MODEL_DIM,
        1.5,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&output)
}

/// Every backend's lookup matches the CPU's to bf16 rounding, and a token past the vocabulary reads zeros.
#[uzu_test]
fn mirai_s_embedding_lookup_matches_cpu() {
    let token_ids = [3u32, 0, 39, VOCAB_SIZE, 17];
    let cpu = lookup::<Cpu>(7, &token_ids);
    let dim = MODEL_DIM as usize;
    assert!(cpu[3 * dim..][..dim].iter().all(|&value| value == bf16::ZERO));
    assert!(cpu.iter().any(|&value| value != bf16::ZERO));
    for_each_non_cpu_backend!(|B| {
        for (index, (actual, expected)) in lookup::<B>(7, &token_ids).iter().zip(&cpu).enumerate() {
            let (actual, expected) = (actual.to_f32(), expected.to_f32());
            let close = (actual - expected).abs() <= expected.abs() / 128.0 + 1e-3;
            assert!(close, "{} index {index}: {actual} vs {expected}", std::any::type_name::<B>());
        }
    });
}
