use super::tiling::{
    quant_params, select_base_mxu_tiling, select_mxu_a8w8_tiling, select_mxu_quant_tiling, select_mxu_tiling,
    select_quant_tiling, select_simdgroup_tiling, select_split_k, split_k_output_supported,
};
use crate::{
    backends::{
        common::{
            Allocation, BufferArg,
            gpu_types::{
                ACTIVATION_QUANTIZATION_GROUP_SIZE, GemmAPrologueKind, GemmParams,
                gemm::{GemmAlignment, GemmBPrologueKind, GemmDTransform, GemmTiling},
            },
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError},
        },
        metal::{Metal, error::MetalError},
    },
    data_type::DataType,
};

/// Resolved bind set for GEMM encode — distinct from public [`MatmulArguments`].
pub(crate) struct GemmEncodeArgs<'a, 'b, 'd, WB> {
    pub a: Option<&'a Allocation<Metal>>,
    pub a_offset: usize,
    pub a_int8: Option<&'a Allocation<Metal>>,
    pub a_scales: Option<&'a Allocation<Metal>>,
    pub a_row_sums: Option<&'a Allocation<Metal>>,
    pub a_prologue: GemmAPrologueKind,

    pub weights: WB,
    pub scales: Option<&'b Allocation<Metal>>,
    pub biases: Option<&'b Allocation<Metal>>,
    pub zero_points: Option<&'b Allocation<Metal>>,
    pub b_prologue: GemmBPrologueKind,
    pub bits_per_b: Option<u32>,
    pub group_size: Option<u32>,

    pub d: &'d mut Allocation<Metal>,
    pub output_bias: Option<&'d Allocation<Metal>>,
    pub rht_factors: Option<&'d Allocation<Metal>>,

    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub b_transpose: bool,
    pub b_leading_dimension: Option<u32>,
    pub ab_scale: f32,
    pub output_transform: GemmDTransform,
}

pub(crate) struct GemmPlan {
    pub use_mxu: bool,
    pub tiling: GemmTiling,
    pub alignment: GemmAlignment,
    pub transpose_b: bool,
    pub params: GemmParams,
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub split_k: u32,
}

pub(crate) fn validate_layout<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
    arguments: &MatmulArguments<'a, 'b, 'd, Metal, TB>
) -> Result<(), MetalError> {
    let is_quant = !matches!(arguments.b, MatmulB::FullPrecision { .. });
    if !is_quant {
        return Ok(());
    }
    let d_mask = arguments.d_transform.mask();
    if d_mask.contains(GemmDTransform::ACCUMULATE) {
        return Err(MatmulError::UnsupportedDOp {
            bit: GemmDTransform::ACCUMULATE,
            path: "QuantGemm",
        }
        .into());
    }
    assert!(
        !d_mask.contains(GemmDTransform::BIAS | GemmDTransform::RHT),
        "QuantGemm with both output bias and output RHT is not supported: bias must be applied after RHT",
    );
    if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
        return Err(MatmulError::UnsupportedLayout {
            path: "QuantGemm",
        }
        .into());
    }
    Ok(())
}

fn validate_int8_a(
    use_mxu: bool,
    a_group_size: u32,
    k: u32,
    b_prologue: GemmBPrologueKind,
    bits_per_b: Option<u32>,
    group_size: Option<u32>,
    row_sums: Option<&Allocation<Metal>>,
) -> Result<(), MetalError> {
    let b_ok_for_int8 =
        matches!(b_prologue, GemmBPrologueKind::ScaleSymmetricDequant | GemmBPrologueKind::ScaleZeroPointDequant);
    if !use_mxu
        || a_group_size != ACTIVATION_QUANTIZATION_GROUP_SIZE
        || !k.is_multiple_of(a_group_size)
        || !b_ok_for_int8
        || bits_per_b != Some(8)
        || group_size != Some(a_group_size)
    {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "int8 activations require MXU, group size 32, and matching 8-bit symmetric/ZP weights",
        }
        .into());
    }
    if matches!(b_prologue, GemmBPrologueKind::ScaleZeroPointDequant) && row_sums.is_none() {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "int8 activations with ZP weights require row_sums",
        }
        .into());
    }
    Ok(())
}

impl<'a, 'b, 'd, WB> GemmEncodeArgs<'a, 'b, 'd, WB> {
    pub(crate) fn a_is_int8(&self) -> bool {
        self.a_prologue == GemmAPrologueKind::Int8Symmetric
    }
}

pub(crate) fn resolve_full_precision<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
    arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>
) -> Result<GemmEncodeArgs<'a, 'b, 'd, TB>, MetalError> {
    let ab_scale = arguments.d_transform.ab_scale;
    let output_bias = arguments.d_transform.bias;
    let rht_factors = arguments.d_transform.rht_factors;
    let output_transform = arguments.d_transform.mask();
    let b_prologue = arguments.b.b_prologue();
    let bits_per_b = arguments.b.bits_per_b();
    let group_size = arguments.b.group_size();

    let MatmulArguments {
        a,
        b,
        b_leading_dimension,
        b_transpose,
        d,
        m,
        n,
        k,
        ..
    } = arguments;

    let MatmulB::FullPrecision {
        b: weights,
    } = b
    else {
        unreachable!("resolve_full_precision requires FullPrecision B");
    };
    let MatmulA::FullPrecision {
        values: a,
        offset: a_offset,
    } = a
    else {
        return Err(MatmulError::IncompatibleA {
            path: "Gemm",
            reason: "int8 activations require symmetric 8-bit weights",
        }
        .into());
    };

    Ok(GemmEncodeArgs {
        a: Some(a),
        a_offset,
        a_int8: None,
        a_scales: None,
        a_row_sums: None,
        a_prologue: GemmAPrologueKind::FullPrecision,
        weights,
        scales: None,
        biases: None,
        zero_points: None,
        b_prologue,
        bits_per_b,
        group_size,
        d,
        output_bias,
        rht_factors,
        m,
        n,
        k,
        b_transpose,
        b_leading_dimension,
        ab_scale,
        output_transform,
    })
}

pub(crate) fn resolve_quant<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
    arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
    use_mxu: bool,
) -> Result<GemmEncodeArgs<'a, 'b, 'd, &'b Allocation<Metal>>, MetalError> {
    let ab_scale = arguments.d_transform.ab_scale;
    let output_bias = arguments.d_transform.bias;
    let rht_factors = arguments.d_transform.rht_factors;
    let output_transform = arguments.d_transform.mask();
    let b_prologue = arguments.b.b_prologue();
    let bits_per_b = arguments.b.bits_per_b();
    let group_size = arguments.b.group_size();

    let MatmulArguments {
        a,
        b,
        b_leading_dimension,
        b_transpose,
        d,
        m,
        n,
        k,
        ..
    } = arguments;

    let (weights, scales, biases, zero_points) = match b {
        MatmulB::ScaleBiasDequant {
            b: w,
            scales,
            biases,
            ..
        } => (w, Some(scales), Some(biases), None),
        MatmulB::ScaleZeroPointDequant {
            b: w,
            scales,
            zero_points,
            ..
        } => (w, Some(scales), None, Some(zero_points)),
        MatmulB::ScaleSymmetricDequant {
            b: w,
            scales,
            ..
        } => (w, Some(scales), None, None),
        MatmulB::FullPrecision {
            ..
        } => unreachable!("resolve_quant requires quantized B"),
    };

    let (a, a_offset, a_int8, a_scales, a_row_sums, a_prologue) = match a {
        MatmulA::FullPrecision {
            values,
            offset,
        } => (Some(values), offset, None, None, None, GemmAPrologueKind::FullPrecision),
        MatmulA::Int8Symmetric {
            values,
            scales: a_scales,
            row_sums,
            group_size: a_group_size,
        } => {
            validate_int8_a(use_mxu, a_group_size, k, b_prologue, bits_per_b, group_size, row_sums)?;
            (None, 0, Some(values), Some(a_scales), row_sums, GemmAPrologueKind::Int8Symmetric)
        },
    };

    Ok(GemmEncodeArgs {
        a,
        a_offset,
        a_int8,
        a_scales,
        a_row_sums,
        a_prologue,
        weights,
        scales,
        biases,
        zero_points,
        b_prologue,
        bits_per_b,
        group_size,
        d,
        output_bias,
        rht_factors,
        m,
        n,
        k,
        b_transpose,
        b_leading_dimension,
        ab_scale,
        output_transform,
    })
}

impl GemmPlan {
    pub(crate) fn build_full_precision<WB>(
        args: &GemmEncodeArgs<'_, '_, '_, WB>,
        use_mxu: bool,
        weights_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        let tiling = if use_mxu {
            if args.b_transpose {
                select_mxu_tiling(args.m, args.n, args.k)
            } else {
                select_base_mxu_tiling(args.m, args.n)
            }
        } else {
            select_simdgroup_tiling(args.m, args.n, args.k)
        };

        let threadgroups_per_row = args.n.div_ceil(tiling.block_n());
        let threadgroups_per_column = args.m.div_ceil(tiling.block_m());

        let (use_morton, group_count_x, group_count_y) = if use_mxu {
            let max_dim = threadgroups_per_row.max(threadgroups_per_column);
            let min_dim = threadgroups_per_row.min(threadgroups_per_column);
            let morton_dim = max_dim.next_power_of_two();
            let morton_total = morton_dim.saturating_mul(morton_dim);
            let actual_total = threadgroups_per_row.saturating_mul(threadgroups_per_column);
            let use_morton = min_dim > 1 && morton_total <= 4_u32.saturating_mul(actual_total);
            if use_morton {
                (true, morton_total, 1)
            } else {
                (false, threadgroups_per_row, threadgroups_per_column)
            }
        } else {
            (false, threadgroups_per_row, threadgroups_per_column)
        };

        let alignment = GemmAlignment::new(
            args.m.is_multiple_of(tiling.block_m()),
            args.n.is_multiple_of(tiling.block_n()),
            args.k.is_multiple_of(tiling.block_k()),
        );

        let split_k = if args.b_transpose && args.b_leading_dimension.is_none() {
            let split_k = select_split_k(args.m, args.n, args.k, tiling, use_mxu, 0, true, false, 512);
            if split_k > 1
                && split_k_output_supported(args.output_transform, args.n, weights_data_type, output_data_type)
            {
                split_k
            } else {
                1
            }
        } else {
            1
        };

        let default_ldb = if args.b_transpose {
            args.k
        } else {
            args.n
        };
        let params = GemmParams {
            M: args.m,
            N: args.n,
            K: args.k,
            leading_dimension_a: args.k,
            leading_dimension_b: args.b_leading_dimension.unwrap_or(default_ldb),
            leading_dimension_d: args.n,
            threadgroups_per_row,
            threadgroups_per_column,
            aligned_inner_iterations: args.k / tiling.block_k(),
            use_morton,
            ab_scale: args.ab_scale,
        };

        Self {
            use_mxu,
            tiling,
            alignment,
            transpose_b: args.b_transpose,
            params,
            group_count_x,
            group_count_y,
            split_k,
        }
    }

    pub(crate) fn build_quant<WB>(
        args: &GemmEncodeArgs<'_, '_, '_, WB>,
        use_mxu: bool,
        weights_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        let group_size = args.group_size.unwrap_or(0);
        let tiling = if use_mxu {
            if args.a_is_int8() {
                select_mxu_a8w8_tiling(args.m, args.n, args.k, group_size)
            } else {
                select_mxu_quant_tiling(args.m, args.n, group_size)
            }
        } else {
            select_quant_tiling(args.m, args.n, group_size)
        };
        let alignment = GemmAlignment::new(
            args.m.is_multiple_of(tiling.block_m()),
            args.n.is_multiple_of(tiling.block_n()),
            args.k.is_multiple_of(tiling.block_k()),
        );
        let params = quant_params(args.m, args.n, args.k, tiling, use_mxu, group_size, args.ab_scale);
        let group_count_x = args.n.div_ceil(tiling.block_n());
        let group_count_y = args.m.div_ceil(tiling.block_m());

        let zero_point_4bit = args.zero_points.is_some() && args.bits_per_b == Some(4);
        // A8W8 kernels are cheaper per tile; a lower target avoids oversplitting MN-heavy shapes.
        let split_k_target = if args.a_is_int8() {
            128
        } else {
            512
        };
        let split_k =
            select_split_k(args.m, args.n, args.k, tiling, use_mxu, group_size, false, zero_point_4bit, split_k_target);
        let split_k = if split_k > 1
            && split_k_output_supported(args.output_transform, args.n, weights_data_type, output_data_type)
        {
            split_k
        } else {
            1
        };

        Self {
            use_mxu,
            tiling,
            alignment,
            transpose_b: true,
            params,
            group_count_x,
            group_count_y,
            split_k,
        }
    }
}

pub(crate) fn select_mxu_tiling_for_args<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
    arguments: &MatmulArguments<'a, 'b, 'd, Metal, TB>,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
) -> Option<GemmTiling> {
    if ![weights_data_type, input_data_type, output_data_type]
        .into_iter()
        .all(|data_type| matches!(data_type, DataType::BF16 | DataType::F32))
    {
        return None;
    }

    match &arguments.b {
        MatmulB::FullPrecision {
            ..
        } => Some(if arguments.b_transpose {
            select_mxu_tiling(arguments.m, arguments.n, arguments.k)
        } else {
            select_base_mxu_tiling(arguments.m, arguments.n)
        }),
        MatmulB::ScaleBiasDequant {
            ..
        }
        | MatmulB::ScaleZeroPointDequant {
            ..
        }
        | MatmulB::ScaleSymmetricDequant {
            ..
        } => {
            if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
                return None;
            }
            let group_size = arguments.b.group_size().unwrap_or(0);
            let tiling = if arguments.a.is_int8() {
                select_mxu_a8w8_tiling(arguments.m, arguments.n, arguments.k, group_size)
            } else {
                select_mxu_quant_tiling(arguments.m, arguments.n, group_size)
            };
            if arguments.a.is_int8() {
                (group_size != 0 && arguments.k.is_multiple_of(group_size)).then_some(tiling)
            } else {
                arguments.k.is_multiple_of(tiling.block_k()).then_some(tiling)
            }
        },
    }
}
