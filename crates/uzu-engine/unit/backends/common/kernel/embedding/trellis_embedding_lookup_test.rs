use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context, Kernels, kernel::TrellisEmbeddingLookupKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{
            buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend, submit_command_buffer,
        },
    },
};

const VOCAB_SIZE: u32 = 3;
const TOKEN_IDS: [u32; 3] = [2, 0, VOCAB_SIZE];
const CODEBOOK: [f32; 5] = [0.08, 0.3, -0.25, 0.125, -0.0625];

// (vector width, transition bits, restart columns)
type Format = (u32, u32, u32);

fn lookup<B: Backend>(
    format: Format,
    model_dim: u32,
    word_order: bool,
) -> Vec<bf16> {
    let (vector_width, transition_bits, restart_columns) = format;
    let mixing_dimension = model_dim >> model_dim.trailing_zeros();
    let mut rng = SmallRng::seed_from_u64(u64::from(model_dim));
    // At least a row's bytes per entry: both sides compute the row stride themselves.
    let mut codes: Vec<u8> = (0..VOCAB_SIZE * model_dim).map(|_| rng.random()).collect();
    // V4 restart tapes are stored as little-endian 32-bit words; the CPU reference reads bytes.
    if word_order && vector_width == 4 && restart_columns != 0 {
        codes.as_chunks_mut::<4>().0.iter_mut().for_each(|word| word.reverse());
    }
    let row_scales: Vec<f32> = (0..VOCAB_SIZE).map(|_| rng.random_range(0.01..0.05)).collect();
    let signs: Vec<f32> = (0..model_dim).map(|_| [1.0, -1.0][rng.random_range(0..2)]).collect();
    let mixing: Vec<f32> = (0..mixing_dimension * mixing_dimension).map(|_| rng.random_range(-0.5..0.5)).collect();

    let context = <B as Backend>::Context::new().unwrap();
    let context = context.as_ref();
    let kernel = <<B as Backend>::Kernels as Kernels>::TrellisEmbeddingLookupKernel::new(
        context,
        vector_width,
        transition_bits,
        restart_columns,
    )
    .unwrap();
    let token_ids = create_buffer_with_data::<B, u32>(context, &TOKEN_IDS);
    let codes = create_buffer_with_data::<B, u8>(context, &codes);
    let row_scales = create_buffer_with_data::<B, f32>(context, &row_scales);
    let signs = create_buffer_with_data::<B, f32>(context, &signs);
    let mixing = create_buffer_with_data::<B, f32>(context, &mixing);
    let mut output = create_buffer::<B, bf16>(context, TOKEN_IDS.len() * model_dim as usize);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &token_ids,
        &codes,
        &row_scales,
        &CODEBOOK,
        &signs,
        &mixing,
        &mut output,
        TOKEN_IDS.len() as u32,
        VOCAB_SIZE,
        model_dim,
        1.5,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    buffer_to_vec::<B, bf16>(&output)
}

// Widths 2048 (no mixing) and 2560 (5x5 mixing) are the Qwen3.5 2B and 4B model dims.
#[rstest]
#[test_attr(uzu_test)]
#[case::v4_t8_r64_2048((4, 8, 64), 2048)]
#[case::v4_t8_r64_2560((4, 8, 64), 2560)]
#[case::v2_t6_2560((2, 6, 0), 2560)]
fn trellis_embedding_lookup_matches_cpu(
    #[case] format: Format,
    #[case] model_dim: u32,
) {
    let expected = lookup::<Cpu>(format, model_dim, false);
    let in_vocab = 2 * model_dim as usize;
    assert!(expected[..in_vocab].iter().any(|value| *value != bf16::ZERO), "in-vocab rows are empty");
    assert!(expected[in_vocab..].iter().all(|value| *value == bf16::ZERO), "out-of-vocab row is not zero");
    for_each_non_cpu_backend!(|B| {
        assert_eq_float(&expected, &lookup::<B>(format, model_dim, true), 0.02, std::any::type_name::<B>());
    });
}
