use crate::{
    backends::common::gpu_types::{
        GemmParams,
        gemm::{GemmDTransform, GemmTiling},
    },
    data_type::DataType,
};

pub(crate) fn quant_params(
    m: u32,
    n: u32,
    k: u32,
    tiling: GemmTiling,
    use_mxu: bool,
    group_size: u32,
    ab_scale: f32,
) -> GemmParams {
    GemmParams {
        M: m,
        N: n,
        K: k,
        leading_dimension_a: k,
        leading_dimension_b: k,
        leading_dimension_d: n,
        threadgroups_per_row: n.div_ceil(tiling.block_n()),
        threadgroups_per_column: m.div_ceil(tiling.block_m()),
        aligned_inner_iterations: split_k_step(tiling, use_mxu, group_size, false).map_or(0, |step| k / step),
        use_morton: false,
        ab_scale,
    }
}

pub(crate) fn split_k_step(
    tiling: GemmTiling,
    use_mxu: bool,
    group_size: u32,
    full_precision: bool,
) -> Option<u32> {
    let step = if use_mxu && !full_precision {
        group_size
    } else {
        tiling.block_k()
    };
    (step != 0).then_some(step)
}

pub(crate) fn split_k_output_supported(
    output_transform: GemmDTransform,
    n: u32,
    weights_data_type: DataType,
    output_data_type: DataType,
) -> bool {
    if !output_transform.contains(GemmDTransform::BIAS) {
        return true;
    }
    n.is_multiple_of(4) && weights_data_type == output_data_type
}

pub(crate) fn select_split_k(
    m: u32,
    n: u32,
    k: u32,
    tiling: GemmTiling,
    use_mxu: bool,
    group_size: u32,
    full_precision: bool,
    zero_point_4bit: bool,
    target_tiles: u32,
) -> u32 {
    let base_tiles = n.div_ceil(tiling.block_n()) * m.div_ceil(tiling.block_m());
    if base_tiles == 0 {
        return 1;
    }
    if !((m as u64) * (n as u64)).is_multiple_of(4) {
        return 1;
    }
    let mut split_k = (target_tiles / base_tiles).max(1);
    let step = match split_k_step(tiling, use_mxu, group_size, full_precision) {
        Some(s) => s,
        None => return 1,
    };
    let mut align = if use_mxu || full_precision {
        step
    } else {
        step.max(group_size)
    };
    if zero_point_4bit {
        align = align.max(2 * group_size);
    }
    split_k = split_k.min((k / align).max(1));
    while split_k > 1 && !k.is_multiple_of(split_k * align) {
        split_k -= 1;
    }
    split_k
}

pub(crate) fn select_simdgroup_tiling(
    m: u32,
    n: u32,
    k: u32,
) -> GemmTiling {
    if 2 * m.max(n) > k {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else {
        GemmTiling::Tile64x32x32_Simdgroups2x2
    }
}

pub(crate) fn select_mxu_tiling(
    m: u32,
    n: u32,
    k: u32,
) -> GemmTiling {
    if m < 64 && n >= 64 {
        if n == k {
            return if m < 16 && k <= 2560 {
                GemmTiling::Tile16x32x256_Simdgroups1x1
            } else {
                GemmTiling::Tile32x64x256_Simdgroups2x2
            };
        }
        return if m < 16 {
            select_small_m_mxu_tiling(n, k)
        } else {
            select_base_mxu_tiling(m, n)
        };
    }
    select_base_mxu_tiling(m, n)
}

pub(crate) fn select_base_mxu_tiling(
    m: u32,
    n: u32,
) -> GemmTiling {
    if m >= 256 && n >= 128 {
        GemmTiling::Tile128x128x256_Simdgroups4x4
    } else if n < 64 {
        GemmTiling::Tile64x32x256_Simdgroups4x1
    } else if m < 64 {
        GemmTiling::Tile32x64x256_Simdgroups2x2
    } else {
        GemmTiling::Tile64x64x256_Simdgroups2x2
    }
}

fn select_small_m_mxu_tiling(
    n: u32,
    k: u32,
) -> GemmTiling {
    if k > n {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    if n > 32_u32.saturating_mul(k) {
        return GemmTiling::Tile16x32x256_Simdgroups1x1;
    }
    if (k >= 4096 && n >= 4_u32.saturating_mul(k)) || (k == 2560 && n >= 6_u32.saturating_mul(k)) {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    GemmTiling::Tile32x64x256_Simdgroups2x2
}

pub(crate) fn select_mxu_quant_tiling(
    m: u32,
    n: u32,
    group_size: u32,
) -> GemmTiling {
    let tiling = select_base_mxu_tiling(m, n);
    if tiling.fits_quant_group_size(group_size) {
        tiling
    } else {
        GemmTiling::Tile64x64x256_Simdgroups2x2
    }
}

pub(crate) fn select_mxu_a8w8_tiling(
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
) -> GemmTiling {
    let tiling = select_mxu_tiling(m, n, k);
    if tiling.fits_quant_group_size(group_size) {
        tiling
    } else {
        GemmTiling::Tile64x64x256_Simdgroups2x2
    }
}

pub(crate) fn select_quant_tiling(
    m: u32,
    n: u32,
    group_size: u32,
) -> GemmTiling {
    if group_size < 32 {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else if m < 32 {
        GemmTiling::Tile8x32x32_Simdgroups1x1
    } else if m >= 64 && n <= 2048 {
        GemmTiling::Tile32x32x32_Simdgroups2x2
    } else if m >= 64 && n >= 6144 && n.is_multiple_of(64) {
        GemmTiling::Tile64x64x32_Simdgroups2x2
    } else {
        GemmTiling::Tile32x32x32_Simdgroups2x2
    }
}
