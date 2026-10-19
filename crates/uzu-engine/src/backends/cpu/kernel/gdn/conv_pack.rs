use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(DeltaNetConvPack)]
#[variants(T, f32, bf16)]
#[variants(KERNEL_SIZE, 4)]
pub fn delta_net_conv_pack<T: DataTypeRepr + Float, const KERNEL_SIZE: u32>(
    state_in: *const f32,
    x: *const T,
    packed: *mut f32,
    suffix_len: u32,
    tile_tokens: u32,
    row_stride: u32,
    conv_dim: u32,
) {
    let taps = KERNEL_SIZE as usize - 1;
    let tile_tokens = tile_tokens as usize;
    let row_stride = row_stride as usize;
    let conv_dim = conv_dim as usize;

    for tile in 0..(suffix_len as usize).div_ceil(tile_tokens) {
        for tap in 0..taps {
            for channel in 0..conv_dim {
                let value = unsafe {
                    if tile == 0 {
                        *state_in.add(channel * taps + tap)
                    } else {
                        (*x.add((tile * tile_tokens - taps + tap) * row_stride + channel)).to_f32().unwrap()
                    }
                };
                unsafe { *packed.add((tile * taps + tap) * conv_dim + channel) = value };
            }
        }
    }
}
