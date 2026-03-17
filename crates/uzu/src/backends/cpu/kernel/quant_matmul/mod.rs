use num_traits::Float;

use crate::ArrayElement;

pub mod qmm;
pub mod qmm_transposed;
pub mod qmm_transposed_64x64;
pub mod qmv;
pub mod qmv_fast;
pub mod qvm;

#[inline(always)]
fn pack_factor(bits: i32) -> usize {
    (32 / bits) as usize
}

#[inline(always)]
fn extract_quant(packed: u32, idx_in_pack: usize, bits: i32) -> i32 {
    let mask = (1u32 << bits) - 1;
    let shift = idx_in_pack as u32 * bits as u32;
    ((packed >> shift) & mask) as i32
}

#[inline(always)]
unsafe fn dequantize_element<T: ArrayElement + Float, const GROUP_SIZE: i32, const BITS: i32>(
    w_row: *const u32,
    scales_row: *const T,
    zero_points_row: Option<*const u8>,
    biases_row: Option<*const T>,
    elem_idx: usize,
    use_mlx_quant: bool,
) -> f32 {
    unsafe {
        let pf = pack_factor(BITS);
        let pack_idx = elem_idx / pf;
        let idx_in_pack = elem_idx % pf;
        let packed = *w_row.add(pack_idx);
        let quant_val = extract_quant(packed, idx_in_pack, BITS);

        let group_idx = elem_idx / GROUP_SIZE as usize;
        let scale = (*scales_row.add(group_idx)).to_f32().unwrap();

        if use_mlx_quant {
            let bias = (*biases_row.unwrap().add(group_idx)).to_f32().unwrap();
            scale * quant_val as f32 + bias
        } else {
            let zp = dequant_zero_point(zero_points_row.unwrap(), group_idx, BITS);
            scale * (quant_val as f32 - zp)
        }
    }
}

#[inline(always)]
unsafe fn dequant_zero_point(zero_points: *const u8, group_idx: usize, bits: i32) -> f32 {
    unsafe {
        if bits == 4 {
            let byte_idx = group_idx / 2;
            let byte = *zero_points.add(byte_idx);
            if (group_idx & 1) == 0 {
                (byte & 0x0F) as f32
            } else {
                ((byte >> 4) & 0x0F) as f32
            }
        } else {
            *zero_points.add(group_idx) as f32
        }
    }
}
