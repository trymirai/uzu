use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

fn moe_gather<T: ArrayElement + Float>(
    x: *const T,
    bucketed_ids: *const i32,
    x_perm: *mut T,
    sumk_buf: *const u32,
    d_model: u32,
    t: u32,
    k: u32,
) {
    let _ = (t, k);
    let total_rows = unsafe { *sumk_buf } as usize;
    let d_model = d_model as usize;
    if total_rows == 0 || d_model == 0 {
        return;
    }

    unsafe {
        for row in 0..total_rows {
            let token = *bucketed_ids.add(row);
            if token < 0 {
                continue;
            }
            let src = x.add(token as usize * d_model);
            let dst = x_perm.add(row * d_model);
            for col in 0..d_model {
                *dst.add(col) = *src.add(col);
            }
        }
    }
}

#[kernel(MoeGatherXPerm2D)]
#[variants(T, f32, f16, bf16)]
pub fn moe_gather_x_perm2_d<T: ArrayElement + Float>(
    x: *const T,
    bucketed_ids: *const i32,
    x_perm: *mut T,
    sumk_buf: *const u32,
    d_model: u32,
    t: u32,
    k: u32,
) {
    moe_gather(x, bucketed_ids, x_perm, sumk_buf, d_model, t, k);
}

#[kernel(MoeGatherXPerm1D)]
#[variants(T, f32, f16, bf16)]
pub fn moe_gather_x_perm1_d<T: ArrayElement + Float>(
    x: *const T,
    bucketed_ids: *const i32,
    x_perm: *mut T,
    sumk_buf: *const u32,
    d_model: u32,
    t: u32,
    k: u32,
) {
    moe_gather(x, bucketed_ids, x_perm, sumk_buf, d_model, t, k);
}
