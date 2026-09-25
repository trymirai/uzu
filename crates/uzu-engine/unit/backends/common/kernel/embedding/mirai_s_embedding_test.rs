use half::{bf16, f16};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::MiraiSEmbeddingLookupKernel},
        cpu::kernel::activation_transform::hadamard_transform,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_backend},
};

const VOCAB_SIZE: u32 = 40;
const MODEL_DIM: u32 = 256;
const INPUT_SCALE: f32 = 1.5;

struct Table {
    codes: Vec<u8>,
    row_scales: Vec<bf16>,
    ladder_indices: Vec<u8>,
    ladder: Vec<f16>,
    table: Vec<i8>,
    factors: Vec<i32>,
}

fn get_output<B: Backend>(
    table: &Table,
    token_ids: &[u32],
) -> Vec<bf16> {
    let context = B::Context::new().unwrap();
    let kernel = <B::Kernels as Kernels>::MiraiSEmbeddingLookupKernel::new(&context).unwrap();
    let batch = token_ids.len() as u32;
    let mut output = alloc_allocation::<B, bf16>(&context, (batch * MODEL_DIM) as usize);
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(
        &alloc_allocation_with_data::<B, u32>(&context, token_ids),
        &alloc_allocation_with_data::<B, u8>(&context, &table.codes),
        &alloc_allocation_with_data::<B, bf16>(&context, &table.row_scales),
        &alloc_allocation_with_data::<B, u8>(&context, &table.ladder_indices),
        &alloc_allocation_with_data::<B, f16>(&context, &table.ladder),
        &alloc_allocation_with_data::<B, i8>(&context, &table.table),
        &alloc_allocation_with_data::<B, i32>(&context, &table.factors),
        &mut output,
        batch,
        VOCAB_SIZE,
        MODEL_DIM,
        INPUT_SCALE,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&output)
}

/// Each row decodes to row_scale * ladder[index] * table[code][c % 4] * input_scale, rounded to bf16, and every
/// 32 columns go through the Walsh-Hadamard transform / sqrt(32) times the output factors.
#[uzu_test]
fn mirai_s_embedding_lookup_matches_reference() {
    let mut rng = SmallRng::seed_from_u64(7);
    let table = Table {
        codes: (0..VOCAB_SIZE * MODEL_DIM / 4).map(|_| rng.random()).collect(),
        row_scales: (0..VOCAB_SIZE).map(|_| bf16::from_f32(rng.random_range(0.01f32..0.1))).collect(),
        ladder_indices: (0..VOCAB_SIZE * MODEL_DIM / 128).map(|_| rng.random()).collect(),
        ladder: (0..16).map(|index| f16::from_f32(2.0f32.powf(index as f32 / 2.0 - 5.5))).collect(),
        table: (0..256 * 4).map(|_| rng.random_range(-4i8..=4)).collect(),
        factors: (0..MODEL_DIM).map(|_| [1, -1][rng.random_range(0..2)]).collect(),
    };
    let token_ids = [3u32, 0, 39, VOCAB_SIZE, 17];
    let model_dim = MODEL_DIM as usize;

    for_each_backend!(|B| {
        let output = get_output::<B>(&table, &token_ids);
        for (index, &token) in token_ids.iter().enumerate() {
            let token = token as usize;
            let row = &output[index * model_dim..][..model_dim];
            if token >= VOCAB_SIZE as usize {
                assert!(row.iter().all(|value| value.to_f32() == 0.0));
                continue;
            }
            let mut decoded: Vec<f32> = (0..model_dim)
                .map(|column| {
                    let group = column / 64;
                    let ladder_index =
                        (table.ladder_indices[token * model_dim / 128 + group / 2] >> (4 * (group % 2))) & 15;
                    let point = table.table[table.codes[token * model_dim / 4 + column / 4] as usize * 4 + column % 4];
                    let value = table.row_scales[token].to_f32()
                        * table.ladder[ladder_index as usize].to_f32()
                        * point as f32
                        * INPUT_SCALE;
                    bf16::from_f32(value).to_f32()
                })
                .collect();
            for (block, (actual, decoded)) in
                row.as_chunks::<32>().0.iter().zip(decoded.as_chunks_mut::<32>().0).enumerate()
            {
                let magnitude: f32 = decoded.iter().map(|value| value.abs()).sum();
                hadamard_transform(decoded);
                for (lane, (actual, value)) in actual.iter().zip(decoded.iter()).enumerate() {
                    let expected = value * table.factors[32 * block + lane] as f32;
                    let tolerance = expected.abs() / 256.0 + magnitude * 1e-6;
                    assert!(
                        (actual.to_f32() - expected).abs() <= tolerance,
                        "{} token {token} column {}",
                        std::any::type_name::<B>(),
                        32 * block + lane
                    );
                }
            }
        }
    });
}
