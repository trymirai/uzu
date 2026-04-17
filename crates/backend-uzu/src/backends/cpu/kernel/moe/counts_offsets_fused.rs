use dsl::kernel;
use num_traits::Float;

use crate::ArrayElement;

#[kernel(MoeCountsOffsetsFused)]
pub fn moe_counts_offsets_fused(
    topk_ids: *const i32,
    offsets: *mut u32,
    sum_k_out: *mut u32,
    partials: *mut u32,
    t_input: u32,
    e_input: u32,
    k_input: u32,
) {
    let e = e_input as usize;
    let t = t_input as usize;
    let k = k_input as usize;

    if e == 0 {
        unsafe {
            *offsets = 0;
            *sum_k_out = 0;
        }
        return;
    }

    // Phase 1: Count tokens per expert (histogram)
    let mut counts = vec![0u32; e];
    for ti in 0..t {
        let base = ti * k;
        for kk in 0..k {
            let eid = unsafe { *topk_ids.add(base + kk) };
            if eid >= 0 {
                let ue = eid as usize;
                if ue < e {
                    counts[ue] += 1;
                }
            }
        }
    }

    // Write partials (same as counts for single-threadgroup case)
    for i in 0..e {
        unsafe { *partials.add(i) = counts[i] };
    }

    // Phase 2: Exclusive prefix scan to produce offsets
    let mut sum = 0u32;
    for i in 0..e {
        unsafe { *offsets.add(i) = sum };
        sum += counts[i];
    }
    unsafe {
        *offsets.add(e) = sum;
        *sum_k_out = sum;
    }
}
