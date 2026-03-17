use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(QuantizedMatmulQmmTransposed)]
#[variants(T, f32, f16, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmm_transposed<T: ArrayElement + Float, const GROUP_SIZE: i32, const BITS: i32>(
    #[allow(unused)] w: *const u32,
    #[allow(unused)] scales: *const T,
    #[allow(unused)]
    #[optional(use_zero_points)]
    zero_points: Option<*const u8>,
    #[allow(unused)]
    #[optional(use_mlx_quant)]
    biases: Option<*const T>,
    #[allow(unused)] x: *const T,
    #[allow(unused)] y: *mut T,
    #[allow(unused)] k: i32,
    #[allow(unused)] n: i32,
    #[allow(unused)] m: i32,
    #[allow(unused)]
    #[specialize]
    use_zero_points: bool,
    #[allow(unused)]
    #[specialize]
    use_mlx_quant: bool,
    #[allow(unused)]
    #[specialize]
    aligned_n: bool,
) {
    // In the transposed layout, weights are stored as [k, n/pack_factor].
    // Each row ki has n elements packed into n/pack_factor u32s.
    // scales: [k, n/group_size], biases: [k, n/group_size]
    let k = k as usize;
    let n = n as usize;
    let m = m as usize;
    let pf = super::pack_factor(BITS);
    let w_row_stride = n / pf;
    let num_groups = (n + GROUP_SIZE as usize - 1) / GROUP_SIZE as usize;

    for batch in 0..m {
        // Zero the output row
        for col in 0..n {
            unsafe { *y.add(batch * n + col) = T::zero() };
        }

        for ki in 0..k {
            let x_val = unsafe { (*x.add(batch * k + ki)).to_f32().unwrap() };
            if x_val == 0.0 {
                continue;
            }
            let w_row = unsafe { w.add(ki * w_row_stride) };
            let scales_row = unsafe { scales.add(ki * num_groups) };

            for col in 0..n {
                let group_idx = col / GROUP_SIZE as usize;
                let pack_idx = col / pf;
                let idx_in_pack = col % pf;
                let packed = unsafe { *w_row.add(pack_idx) };
                let quant_val = super::extract_quant(packed, idx_in_pack, BITS);

                let scale = unsafe { (*scales_row.add(group_idx)).to_f32().unwrap() };
                let dequant = if use_mlx_quant {
                    let bias = unsafe { (*biases.unwrap().add(ki * num_groups + group_idx)).to_f32().unwrap() };
                    scale * quant_val as f32 + bias
                } else {
                    let zp = unsafe { super::dequant_zero_point(zero_points.unwrap().add(ki * ((num_groups + (if BITS == 4 { 1 } else { 0 })) / (if BITS == 4 { 2 } else { 1 }))), group_idx, BITS) };
                    scale * (quant_val as f32 - zp)
                };

                unsafe {
                    let out_idx = batch * n + col;
                    let prev = (*y.add(out_idx)).to_f32().unwrap();
                    *y.add(out_idx) = T::from(prev + dequant * x_val).unwrap();
                };
            }
        }
    }
}
