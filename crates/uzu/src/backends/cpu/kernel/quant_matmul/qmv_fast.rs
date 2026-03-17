use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(QuantizedMatmulQmvFast)]
#[variants(T, f32, f16, bf16)]
#[variants(GROUP_SIZE, 32, 64, 128)]
#[variants(BITS, 4, 8)]
pub fn quantized_matmul_qmv_fast<T: ArrayElement + Float, const GROUP_SIZE: i32, const BITS: i32>(
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
) {
    let k = k as usize;
    let n = n as usize;
    let m = m as usize;
    let pf = super::pack_factor(BITS);
    let w_row_stride = k / pf;
    let num_groups = (k + GROUP_SIZE as usize - 1) / GROUP_SIZE as usize;

    for batch in 0..m {
        for row in 0..n {
            let mut acc = 0.0f32;
            let w_row = unsafe { w.add(row * w_row_stride) };
            let scales_row = unsafe { scales.add(row * num_groups) };

            for ki in 0..k {
                let dequant = unsafe {
                    super::dequantize_element::<T, GROUP_SIZE, BITS>(
                        w_row,
                        scales_row,
                        zero_points.map(|zp| zp.add(row * ((num_groups + (if BITS == 4 { 1 } else { 0 })) / (if BITS == 4 { 2 } else { 1 })))),
                        biases.map(|b| b.add(row * num_groups)),
                        ki,
                        use_mlx_quant,
                    )
                };
                let x_val = unsafe { (*x.add(batch * k + ki)).to_f32().unwrap() };
                acc += dequant * x_val;
            }

            unsafe { *y.add(batch * n + row) = T::from(acc).unwrap() };
        }
    }
}
