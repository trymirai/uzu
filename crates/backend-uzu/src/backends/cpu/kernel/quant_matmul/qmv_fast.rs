use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::{ArrayElement, backends::cpu::kernel::quant_matmul::qmv::qmv};

#[kernel(QuantizedMatmulQmvFast)]
#[variants(T, f32, f16, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4, 8)]
#[variants(LORA_RANK, 16)]
pub fn quantized_matmul_qmv_fast<
    T: ArrayElement + Float,
    const GROUP_SIZE: u32,
    const BITS: u32,
    const LORA_RANK: u32,
>(
    weights: *const u32,
    scales: *const T,
    #[optional(use_zero_points)] zero_points: Option<*const u8>,
    #[optional(use_mlx_quant)] biases: Option<*const T>,
    input: *const T,
    output: *mut T,
    #[optional(use_hadamard)] hadamard_factors: Option<*const i32>,
    #[optional(use_lora)] h_input: Option<*const T>,
    #[optional(use_lora)] adapter_up: Option<*const T>,
    #[optional(use_lora)] lora_scale: Option<f32>,
    in_vec_size: u32,
    out_vec_size: u32,
    batch_size: u32,
    #[specialize] use_zero_points: bool,
    #[specialize] use_mlx_quant: bool,
    #[specialize] use_hadamard: bool,
    #[specialize] use_lora: bool,
) {
    if use_hadamard {
        unimplemented!("not supported yet");
    }
    qmv::<T>(
        weights,
        scales,
        zero_points,
        biases,
        input,
        output,
        in_vec_size as usize,
        out_vec_size as usize,
        batch_size as usize,
        use_zero_points,
        use_mlx_quant,
        GROUP_SIZE as usize,
        BITS as usize,
    );
    if use_lora {
        let rank = LORA_RANK as usize;
        let h = h_input.unwrap();
        let a_up = adapter_up.unwrap();
        let scale = lora_scale.unwrap();
        for b in 0..(batch_size as usize) {
            for o in 0..(out_vec_size as usize) {
                let mut delta = 0f32;
                for r in 0..rank {
                    delta += unsafe { (*h.add(b * rank + r)).to_f32().unwrap() }
                        * unsafe { (*a_up.add(o * rank + r)).to_f32().unwrap() };
                }
                let prev = unsafe { (*output.add(b * out_vec_size as usize + o)).to_f32().unwrap() };
                unsafe {
                    *output.add(b * out_vec_size as usize + o) = T::from(prev + scale * delta).unwrap();
                }
            }
        }
    }
}
