//! Lane-ordered weight units for the packed-W4 small-batch GEMM (`GemmBLayout::SmallBatch`, read by
//! `schedules/small_batch/packed.h`).
//!
//! One unit per (32-row tile, weight group), `32 lanes x 4 tile rows x (group / 32)` code words in the order the lanes
//! consume them (one 16 B load per lane), then bf16 scales of the 8 lane rows x 4 tile rows, then the offsets per lane
//! row: 4 zero-point nibbles (one ushort), or 4 bf16 biases. A unit is exactly as large as the production codes,
//! scales and zero points of the same rows (biases: scales and biases).
use half::bf16;

use crate::backends::common::kernel::matmul::QuantParamsStrides;

#[derive(Clone, Copy)]
pub enum Offsets<'a> {
    ZeroPoints(&'a [u8], QuantParamsStrides),
    Biases(&'a [bf16]),
}

pub fn pack_small_batch_units(
    n: usize,
    k: usize,
    group: usize,
    codes: &[u32],
    scales: &[bf16],
    scale_strides: QuantParamsStrides,
    offsets: Offsets,
) -> Vec<u32> {
    let groups = k / group;
    let words_per_lane = group / 32;
    let code_words = 32 * 4 * words_per_lane;
    let offset_words = match offsets {
        Offsets::ZeroPoints(..) => 4,
        Offsets::Biases(_) => 16,
    };
    let unit_words = code_words + 16 + offset_words;
    let scale_index =
        |row: usize, g: usize| row * scale_strides.output_stride as usize + g * scale_strides.group_stride as usize;
    let offset = |row: usize, g: usize| -> u16 {
        match offsets {
            Offsets::ZeroPoints(zero_points, strides) => {
                let index = row * strides.output_stride as usize + g * strides.group_stride as usize;
                ((zero_points[index / 2] >> (4 * (index % 2))) & 0xF) as u16
            },
            Offsets::Biases(biases) => biases[scale_index(row, g)].to_bits(),
        }
    };
    let mut units = vec![0u32; (n / 32) * groups * unit_words];
    let tiles_per_thread = (n / 32).div_ceil(std::thread::available_parallelism().map_or(1, |threads| threads.get()));
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in units.chunks_mut(tiles_per_thread * groups * unit_words).enumerate() {
            let (scale_index, offset) = (&scale_index, &offset);
            scope.spawn(move || {
                for (tile_in_chunk, tile_units) in chunk.chunks_mut(groups * unit_words).enumerate() {
                    let tile = chunk_index * tiles_per_thread + tile_in_chunk;
                    for (g, unit) in tile_units.chunks_mut(unit_words).enumerate() {
                        for lane in 0..32usize {
                            let quad = lane / 4;
                            let lane_row = (quad & 4) + (lane / 2) % 4;
                            let pair = (quad & 2) + lane % 2;
                            for tile_row in 0..4usize {
                                let row = tile * 32 + tile_row * 8 + lane_row;
                                for w in 0..words_per_lane {
                                    unit[lane * 4 * words_per_lane + tile_row * words_per_lane + w] =
                                        codes[row * (k / 8) + g * (group / 8) + pair * words_per_lane + w];
                                }
                            }
                        }
                        let halves: &mut [u16] = bytemuck::cast_slice_mut(&mut unit[code_words..]);
                        for lane_row in 0..8usize {
                            for tile_row in 0..4usize {
                                let row = tile * 32 + tile_row * 8 + lane_row;
                                halves[lane_row * 4 + tile_row] = scales[scale_index(row, g)].to_bits();
                                match offsets {
                                    Offsets::ZeroPoints(..) => {
                                        halves[32 + lane_row] |= offset(row, g) << (4 * tile_row);
                                    },
                                    Offsets::Biases(_) => halves[32 + lane_row * 4 + tile_row] = offset(row, g),
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    units
}
