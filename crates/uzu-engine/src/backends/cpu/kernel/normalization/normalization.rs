use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::{
        common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE, cpu::kernel::activation_transform::hadamard_transform,
    },
};

#[kernel(Normalization)]
#[variants(InputT, f32, f16, bf16)]
#[variants(AffineT, f32, f16, bf16)]
#[variants(OutputT, f32, f16, bf16)]
#[variants(AccumT, f32)]
pub fn normalization<
    InputT: ArrayElement + Float,
    AffineT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    #[optional(!in_place)] input: Option<*const InputT>,
    #[optional(has_scales)] scales: Option<*const AffineT>,
    #[optional(has_biases)] biases: Option<*const AffineT>,
    output: *mut OutputT,
    #[optional(copy_to_shortcut)] shortcut: Option<*mut InputT>,
    #[optional(use_hadamard)] hadamard_factors: Option<*const i32>,
    batch_size: u32,
    element_count: u32,
    epsilon: f32,
    scale_offset: f32,
    post_layer_scalar: f32,
    #[specialize] in_place: bool,
    #[specialize] subtract_mean: bool,
    #[specialize] full_layer: bool,
    #[specialize] copy_to_shortcut: bool,
    #[specialize] residual_add: bool,
    #[specialize] use_hadamard: bool,
    #[specialize] scale_residual_sum: bool,
    #[specialize] scale_output: bool,
    #[specialize] has_biases: bool,
    #[specialize] has_scales: bool,
) {
    assert_eq!(shortcut.is_some(), copy_to_shortcut);
    assert_eq!(hadamard_factors.is_some(), use_hadamard);
    assert_eq!(biases.is_some(), has_biases);
    assert_eq!(scales.is_some(), has_scales);
    assert!(copy_to_shortcut || !residual_add);
    assert!(!use_hadamard || element_count.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE));

    let input = match in_place {
        true => output as *const InputT,
        false => input.unwrap(),
    };

    let element_count = element_count as usize;
    let epsilon = AccumT::from(epsilon).unwrap();
    let scale_offset = AccumT::from(scale_offset).unwrap();
    let element_count_accum = AccumT::from(element_count).unwrap();
    let mut row = vec![OutputT::zero(); element_count];

    for batch in 0..(batch_size as usize) {
        let batch_offset = batch * element_count;

        // The stored value of an element after the first pass: the shortcut after a residual add, otherwise the input
        let stored = |i: usize| unsafe {
            match residual_add {
                true => AccumT::from(*shortcut.unwrap().add(batch_offset + i)).unwrap(),
                false => AccumT::from(*input.add(batch_offset + i)).unwrap(),
            }
        };

        // Residual add or copy to shortcut, and the sum of squares of the RMS path
        let mut sum_sq = AccumT::zero();
        for i in 0..element_count {
            let input_val = unsafe { *input.add(batch_offset + i) };
            let mut val = input_val;
            if copy_to_shortcut {
                let skip_ptr = unsafe { shortcut.unwrap().add(batch_offset + i) };
                if residual_add {
                    val = val + unsafe { *skip_ptr };
                    if scale_residual_sum {
                        val = InputT::from(val.to_f32().unwrap() * post_layer_scalar).unwrap();
                    }
                }
                unsafe { *skip_ptr = val };
            }
            if !subtract_mean {
                let accum_val = AccumT::from(val).unwrap();
                sum_sq = sum_sq + accum_val * accum_val;
            }
        }
        // Shifted two-pass variance around the first stored element: deltas are exact for near-constant rows, and
        // squared deviations from their mean avoid the cancellation of E[x²] - mean²
        let (pivot, mean_delta) = if subtract_mean && element_count > 0 {
            let pivot = stored(0);
            let delta_sum = (0..element_count).fold(AccumT::zero(), |sum, i| sum + (stored(i) - pivot));
            let mean_delta = delta_sum / element_count_accum;
            sum_sq = (0..element_count).fold(AccumT::zero(), |sum, i| {
                let deviation = (stored(i) - pivot) - mean_delta;
                sum + deviation * deviation
            });
            (pivot, mean_delta)
        } else {
            (AccumT::zero(), AccumT::zero())
        };
        let variance = sum_sq / element_count_accum;
        let rms_inv = AccumT::from((variance + epsilon).to_f32().unwrap().sqrt().recip()).unwrap();

        // Normalization and scaling
        for i in 0..element_count {
            let normalized: AccumT = ((stored(i) - pivot) - mean_delta) * rms_inv;
            let mut result: OutputT = if has_scales {
                let scale_val = unsafe { AccumT::from(*scales.unwrap().add(i)).unwrap() };
                if full_layer {
                    // Full-layer: keep everything in accumulation precision
                    let scale_with_offset: AccumT = scale_val + scale_offset;
                    OutputT::from(normalized * scale_with_offset).unwrap()
                } else {
                    // Only-normalization: cast down to output precision for the scale multiply
                    let normalized_out = OutputT::from(normalized).unwrap();
                    let scale_with_offset_out = OutputT::from(scale_val + scale_offset).unwrap();
                    normalized_out * scale_with_offset_out
                }
            } else {
                OutputT::from(normalized).unwrap()
            };
            if has_biases {
                let bias = unsafe { AccumT::from(*biases.unwrap().add(i)).unwrap() };
                result = OutputT::from(AccumT::from(result).unwrap() + bias).unwrap();
            }
            row[i] = result;
        }

        // Input RHT like the Metal kernel: apply the factors, then transform each block, before output scaling
        if let Some(factors) = hadamard_factors {
            for block_start in (0..element_count).step_by(HADAMARD_TRANSFORM_BLOCK_SIZE as usize) {
                let mut block: [f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize] = std::array::from_fn(|lane| {
                    let factor = unsafe { *factors.add(block_start + lane) } as f32;
                    row[block_start + lane].to_f32().unwrap() * factor
                });
                hadamard_transform(&mut block);
                for (lane, value) in block.into_iter().enumerate() {
                    row[block_start + lane] = OutputT::from(value).unwrap();
                }
            }
        }

        for (i, &value) in row.iter().enumerate() {
            let mut result = value;
            if scale_output {
                // The scalar is FP32, as in the Metal kernel
                result = OutputT::from(result.to_f32().unwrap() * post_layer_scalar).unwrap();
            }
            unsafe { *output.add(batch_offset + i) = result };
        }
    }
}
