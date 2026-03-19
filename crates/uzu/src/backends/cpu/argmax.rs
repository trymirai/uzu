#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use num_traits::NumCast;

use crate::ArrayElement;

pub fn optimized_argmax<T: ArrayElement>(input: &[T]) -> usize {
    if input.is_empty() {
        return 0;
    }

    unsafe {
        let len = input.len();
        let ptr = input.as_ptr();

        let mut max_val = NumCast::from(*ptr).unwrap_or(f32::NEG_INFINITY);
        let mut max_idx = 0;
        let mut i = 1;

        // 32x aggressive unrolling for medium arrays with manual optimization
        while i + 32 <= len {
            // Process 4 chunks of 8 elements each for better instruction pipeline utilization
            for chunk in 0..4 {
                let base = i + chunk * 8;
                for j in 0..8 {
                    let val: f32 = NumCast::from(*ptr.add(base + j)).unwrap_or(f32::NEG_INFINITY);
                    if val > max_val || (val == max_val && base + j < max_idx) {
                        max_val = val;
                        max_idx = base + j;
                    }
                }
            }
            i += 32;
        }

        // Handle remaining 16x chunks
        while i + 16 <= len {
            for j in 0..16 {
                let val: f32 = NumCast::from(*ptr.add(i + j)).unwrap_or(f32::NEG_INFINITY);
                if val > max_val || (val == max_val && i + j < max_idx) {
                    max_val = val;
                    max_idx = i + j;
                }
            }
            i += 16;
        }

        // Handle remaining 8x chunks
        while i + 8 <= len {
            for j in 0..8 {
                let val: f32 = NumCast::from(*ptr.add(i + j)).unwrap_or(f32::NEG_INFINITY);
                if val > max_val || (val == max_val && i + j < max_idx) {
                    max_val = val;
                    max_idx = i + j;
                }
            }
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            let val: f32 = NumCast::from(*ptr.add(i)).unwrap_or(f32::NEG_INFINITY);
            if val > max_val || (val == max_val && i < max_idx) {
                max_val = val;
                max_idx = i;
            }
            i += 1;
        }

        max_idx
    }
}

pub fn simple_argmax<T: ArrayElement>(input: &[T]) -> usize {
    input
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(best_index, best_value), (index, &value)| {
            let value_f32 = NumCast::from(value).unwrap_or(f32::NEG_INFINITY);
            if value_f32 > best_value || (value_f32 == best_value && index < best_index) {
                (index, value_f32)
            } else {
                (best_index, best_value)
            }
        })
        .0
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_argmax_f32(input: &[f32]) -> usize {
    unsafe {
        if input.is_empty() {
            return 0;
        }

        let len = input.len();
        let ptr = input.as_ptr();

        if len < 4 {
            return simple_argmax(input);
        }

        // Initialize with first 4 elements
        let mut max_vals = vld1q_f32(ptr);
        let mut max_indices = vdupq_n_u32(0);
        let increment = vdupq_n_u32(4);

        // Set initial indices [0, 1, 2, 3]
        max_indices = vsetq_lane_u32(0, max_indices, 0);
        max_indices = vsetq_lane_u32(1, max_indices, 1);
        max_indices = vsetq_lane_u32(2, max_indices, 2);
        max_indices = vsetq_lane_u32(3, max_indices, 3);

        let mut current_indices = max_indices;
        let mut i = 4;

        // Process 4 elements at a time
        while i + 4 <= len {
            let vals = vld1q_f32(ptr.add(i));
            current_indices = vaddq_u32(current_indices, increment);

            // Compare values
            let mask = vcgtq_f32(vals, max_vals);

            // Update max values and indices where vals > max_vals
            max_vals = vbslq_f32(mask, vals, max_vals);
            max_indices = vbslq_u32(mask, current_indices, max_indices);

            i += 4;
        }

        // Extract the maximum from the vector
        let max_vals_array: [f32; 4] = std::mem::transmute(max_vals);
        let max_indices_array: [u32; 4] = std::mem::transmute(max_indices);

        let mut global_max = max_vals_array[0];
        let mut global_max_idx = max_indices_array[0] as usize;

        for j in 1..4 {
            if max_vals_array[j] > global_max
                || (max_vals_array[j] == global_max && (max_indices_array[j] as usize) < global_max_idx)
            {
                global_max = max_vals_array[j];
                global_max_idx = max_indices_array[j] as usize;
            }
        }

        // Handle remaining elements
        while i < len {
            let val = *ptr.add(i);
            if val > global_max || (val == global_max && i < global_max_idx) {
                global_max = val;
                global_max_idx = i;
            }
            i += 1;
        }

        global_max_idx
    }
}

#[cfg(target_arch = "aarch64")]
pub fn neon_optimized_argmax<T: ArrayElement>(input: &[T]) -> usize {
    // Check if NEON is available at runtime
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return simple_argmax(input);
    }

    // For f32, use direct NEON implementation
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        unsafe {
            let f32_slice = std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len());
            return neon_argmax_f32(f32_slice);
        }
    }

    // For other types, fall back to simple
    simple_argmax(input)
}

#[cfg(test)]
#[path = "../../../tests_unit/backends/cpu/argmax_test.rs"]
mod tests;
