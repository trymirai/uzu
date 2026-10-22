use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{backends::common::gpu_types::activation_type::ActivationType, data_type::DataTypeRepr};

// Multi-token causal conv1d with SiLU for DeltaNet, in place on in_proj. Each token tile starts from its packed rows (see
// DeltaNetConvPack), so tiles are independent; the last tile writes the new conv state.
//
// packed:      [num_tiles, KERNEL_SIZE - 1, conv_dim]
// conv_weight: [conv_dim, KERNEL_SIZE]
// bias:        [conv_dim] (optional)
// in_proj:     [suffix_len, out_stride] - first conv_dim channels overwritten
// state_out:   [conv_dim, KERNEL_SIZE - 1]
#[kernel(DeltaNetConvScan)]
#[variants(T, f32, bf16)]
#[variants(KERNEL_SIZE, 4)]
pub fn delta_net_conv_scan<T: DataTypeRepr + Float, const KERNEL_SIZE: u32>(
    packed: *const f32,
    conv_weight: *const f32,
    #[optional(has_bias)] bias: Option<*const f32>,
    in_proj: *mut T,
    state_out: *mut f32,
    suffix_len: u32,
    tile_tokens: u32,
    conv_dim: u32,
    out_stride: u32,
    #[specialize] has_bias: bool,
) {
    let kernel_size = KERNEL_SIZE as usize;
    let taps = kernel_size - 1;
    let suffix_len = suffix_len as usize;
    let tile_tokens = tile_tokens as usize;
    let conv_dim = conv_dim as usize;
    let out_stride = out_stride as usize;

    for tile in 0..suffix_len.div_ceil(tile_tokens) {
        for channel in 0..conv_dim {
            let mut window: Vec<f32> =
                (0..taps).map(|tap| unsafe { *packed.add((tile * taps + tap) * conv_dim + channel) }).collect();
            for token in tile * tile_tokens..((tile + 1) * tile_tokens).min(suffix_len) {
                let sample = unsafe { (*in_proj.add(token * out_stride + channel)).to_f32().unwrap() };
                let mut acc = if has_bias {
                    unsafe { *bias.unwrap().add(channel) }
                } else {
                    0.0f32
                };
                for (tap, &previous) in window.iter().enumerate() {
                    acc += unsafe { *conv_weight.add(channel * kernel_size + tap) } * previous;
                }
                acc += unsafe { *conv_weight.add(channel * kernel_size + taps) } * sample;
                unsafe {
                    *in_proj.add(token * out_stride + channel) = T::from(ActivationType::SILU.activate(acc)).unwrap();
                }
                window.remove(0);
                window.push(sample);
            }
            if (tile + 1) * tile_tokens >= suffix_len {
                for (tap, &value) in window.iter().enumerate() {
                    unsafe { *state_out.add(channel * taps + tap) = value };
                }
            }
        }
    }
}
