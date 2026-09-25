use half::{bf16, f16};
use uzu_engine_macros::kernel;

use crate::backends::{
    common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE, cpu::kernel::activation_transform::hadamard_transform,
};

const BLOCK_SIZE: usize = HADAMARD_TRANSFORM_BLOCK_SIZE as usize;

#[kernel(MiraiSEmbeddingLookup)]
pub fn mirai_s_embedding_lookup(
    token_ids: *const u32,
    codes: *const u8,
    row_scales: *const bf16,
    ladder_indices: *const u8,
    ladder: *const f16,
    table: *const i8,
    output_hadamard_factors: *const i32,
    output: *mut bf16,
    batch_size: u32,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
) {
    let model_dim = model_dim as usize;
    unsafe {
        for batch_index in 0..batch_size as usize {
            let token = *token_ids.add(batch_index) as usize;
            let output = std::slice::from_raw_parts_mut(output.add(batch_index * model_dim), model_dim);
            if token >= vocab_size as usize {
                output.fill(bf16::ZERO);
                continue;
            }
            let row_scale = (*row_scales.add(token)).to_f32();
            for block_start in (0..model_dim).step_by(BLOCK_SIZE) {
                let mut block: [f32; BLOCK_SIZE] = std::array::from_fn(|lane| {
                    let column = block_start + lane;
                    let group = column / 64;
                    let ladder_index =
                        (*ladder_indices.add(token * (model_dim / 128) + group / 2) >> (4 * (group % 2))) & 15;
                    let code = *codes.add(token * (model_dim / 4) + column / 4) as usize;
                    let point = *table.add(4 * code + column % 4);
                    let value = row_scale * (*ladder.add(ladder_index as usize)).to_f32() * point as f32 * input_scale;
                    bf16::from_f32(value).to_f32()
                });
                hadamard_transform(&mut block);
                for (lane, value) in block.into_iter().enumerate() {
                    let factor = *output_hadamard_factors.add(block_start + lane);
                    output[block_start + lane] = bf16::from_f32(value * factor as f32);
                }
            }
        }
    }
}
