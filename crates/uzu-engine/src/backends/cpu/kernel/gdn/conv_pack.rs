use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(Conv1dPack)]
#[variants(StateT, f32, bf16)]
#[variants(InputT, f32, bf16)]
pub fn conv1d_pack<StateT: DataTypeRepr + Float, InputT: DataTypeRepr + Float>(
    state_in: *const StateT,
    x: *const InputT,
    padded: *mut StateT,
    state_stride: u32,
    row_stride: u32,
    suffix_len: u32,
    num_channels: u32,
) {
    let state_stride = state_stride as usize;
    let row_stride = row_stride as usize;
    let suffix_len = suffix_len as usize;
    let num_channels = num_channels as usize;

    unsafe {
        for channel_idx in 0..num_channels {
            for row_idx in 0..state_stride + suffix_len {
                let padded_index = row_idx * row_stride + channel_idx;
                if row_idx < state_stride {
                    let state_index = channel_idx * state_stride + row_idx;
                    *padded.add(padded_index) = *state_in.add(state_index);
                } else {
                    let token = row_idx - state_stride;
                    let x_index = token * row_stride + channel_idx;
                    *padded.add(padded_index) = StateT::from((*x.add(x_index)).to_f32().unwrap()).unwrap();
                }
            }
        }
    }
}
