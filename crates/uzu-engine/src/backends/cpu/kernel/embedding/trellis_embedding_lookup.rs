use half::bf16;
use uzu_engine_macros::kernel;

use crate::backends::cpu::kernel::activation_transform::hadamard_transform;

const STATE_BITS: usize = 16;
const WEIGHT_BIAS: i32 = 54;

#[kernel(TrellisEmbeddingLookup)]
pub fn trellis_embedding_lookup(
    token_ids: *const u32,
    codes: *const u8,
    row_scales: *const f32,
    codebook: &[f32; 5],
    signs: *const f32,
    mixing: *const f32,
    output: *mut bf16,
    batch_size: u32,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
    #[specialize] vector_width: u32,
    #[specialize] transition_bits: u32,
    #[specialize] restart_columns: u32,
) {
    let (vector_width, transition_bits, restart_columns) =
        (vector_width as usize, transition_bits as usize, restart_columns as usize);
    let columns = model_dim as usize;
    let width = 1 << columns.trailing_zeros();
    let mixing_dimension = columns / width;
    let block_bits = |block_columns: usize| STATE_BITS + (block_columns / vector_width - 1) * transition_bits;
    let row_bytes = match columns.checked_div(restart_columns) {
        None => block_bits(columns).div_ceil(8),
        Some(blocks) => (blocks * block_bits(restart_columns)).div_ceil(8).next_multiple_of(16),
    };
    let signs = unsafe { std::slice::from_raw_parts(signs, columns) };
    let mixing = unsafe { std::slice::from_raw_parts(mixing, mixing_dimension * mixing_dimension) };

    for batch in 0..batch_size as usize {
        let output_row = unsafe { std::slice::from_raw_parts_mut(output.add(batch * columns), columns) };
        let token = unsafe { *token_ids.add(batch) };
        if token >= vocab_size {
            output_row.fill(bf16::ZERO);
            continue;
        }
        let token = token as usize;
        let code_row = unsafe { std::slice::from_raw_parts(codes.add(token * row_bytes), row_bytes) };
        let row_scale = unsafe { *row_scales.add(token) };

        // Rotated row, decoded as the trellis GEMM does: a state hash gives one biased weight byte per column.
        let rotated: Vec<f32> = (0..columns)
            .map(|column| {
                let (block_first_bit, block_column) = match restart_columns {
                    0 => (0, column),
                    _ => (column / restart_columns * block_bits(restart_columns), column % restart_columns),
                };
                let first_bit = block_first_bit + block_column / vector_width * transition_bits;
                let state = (first_bit..first_bit + STATE_BITS)
                    .fold(0u32, |state, bit| state << 1 | u32::from(code_row[bit / 8] >> (7 - bit % 8) & 1));
                let hash = state.wrapping_mul(0xCFCC_B83F).wrapping_add(0x584B_4AA3);
                let hash = (hash ^ hash >> 16).wrapping_mul(0x85EB_CA6B);
                let byte = ((hash ^ hash >> 16) >> (8 * (column % vector_width))) & 0xFF;
                let field_sum: u32 = (0..4).map(|field| (byte >> (2 * field)) & 3).sum();
                let level = (8 * field_sum + ((3 * (byte & 15)) & 15)) as i32 - WEIGHT_BIAS;
                let value = (codebook[0] * level as f32 + codebook[1 + column % 4]) * row_scale;
                bf16::from_f32(value).to_f32()
            })
            .collect();

        // Undo the Kronecker rotation: Hadamard over the width axis of [width, mixing], then mix, then flip signs.
        let mut hadamard = vec![0.0f32; columns];
        for mixing_index in 0..mixing_dimension {
            let mut lane: Vec<f32> = (0..width).map(|row| rotated[row * mixing_dimension + mixing_index]).collect();
            hadamard_transform(&mut lane);
            for (row, value) in lane.into_iter().enumerate() {
                hadamard[row * mixing_dimension + mixing_index] = value;
            }
        }
        for (column, value) in output_row.iter_mut().enumerate() {
            let row_start = column / mixing_dimension * mixing_dimension;
            let mixing_column = column % mixing_dimension;
            let mixed: f32 = (0..mixing_dimension)
                .map(|source| hadamard[row_start + source] * mixing[source * mixing_dimension + mixing_column])
                .sum();
            *value = bf16::from_f32(mixed * signs[column] * input_scale);
        }
    }
}
