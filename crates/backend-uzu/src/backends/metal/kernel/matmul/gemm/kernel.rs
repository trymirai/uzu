use std::collections::{HashMap, hash_map::Entry};

use super::specialization::GemmSpecialization;
use crate::{
    backends::{
        common::{
            Allocation, AsBufferRangeRef, Backend, Buffer, Encoder,
            gpu_types::{
                GemmParams, HadamardTransformOrder,
                gemm::{GemmAlignment, GemmDTransform, GemmTiling},
            },
            kernel::{
                HadamardTransformKernel, Kernels, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulB, MatmulError, MatmulQuantCombo},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{GemmMetalKernel, GemmSplitKReduceMetalKernel, TensorAddBiasMetalKernel},
            metal_extensions::DeviceExt,
        },
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy)]
pub enum GemmDispatchPath {
    Simdgroup,
    Mxu,
}

pub struct GemmKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmMetalKernel>,
    pub bias_add: TensorAddBiasMetalKernel,
    pub hadamard: <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel,
    // Single-dispatch reduction of split-K partials (sum over split_k), with the elementwise
    // epilogue (scale, then per-column bias) fused in — like the main GEMM finalize. Two
    // bias-specialized variants; RHT (cross-column) stays a separate post-pass.
    split_k_reduce: GemmSplitKReduceMetalKernel,
    split_k_reduce_bias: GemmSplitKReduceMetalKernel,
}

impl GemmKernel {
    pub(crate) fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        let bias_add = TensorAddBiasMetalKernel::new(context, output_data_type, weights_data_type, true)?;
        let hadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            output_data_type,
            HadamardTransformOrder::Output,
        )?;
        let split_k_reduce = GemmSplitKReduceMetalKernel::new(context, output_data_type, false)?;
        let split_k_reduce_bias = GemmSplitKReduceMetalKernel::new(context, output_data_type, true)?;
        let mut kernel = Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            kernels: HashMap::new(),
            bias_add,
            hadamard,
            split_k_reduce,
            split_k_reduce_bias,
        };
        for specialization in GemmSpecialization::precompile_configs(weights_data_type) {
            kernel.get_or_create(context, specialization)?;
        }
        Ok(kernel)
    }

    pub fn preheat_quant_combo(
        &mut self,
        context: &MetalContext,
        combo: MatmulQuantCombo,
    ) -> Result<(), MetalError> {
        for specialization in GemmSpecialization::quant_combo_specs(self.weights_data_type, combo) {
            self.get_or_create(context, specialization)?;
        }
        Ok(())
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemmSpecialization,
    ) -> Result<&GemmMetalKernel, MetalError> {
        match self.kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    specialization.use_mxu,
                    specialization.b_prologue,
                    specialization.bits_per_b.unwrap_or(0),
                    specialization.group_size.unwrap_or(0),
                    specialization.output_transform,
                    specialization.alignment,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub fn encode<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let mxu_eligible_for_quant = match &arguments.b {
            MatmulB::FullPrecision {
                ..
            } => true,
            MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                ..
            } => {
                arguments.b_transpose
                    && arguments.b_leading_dimension.is_none_or(|ld| ld == arguments.k)
                    && arguments.b_offset == 0
                    && arguments.k.is_multiple_of(select_mxu_tiling(arguments.m, arguments.n).block_k())
            },
        };
        let path = if encoder.context().device.supports_mxu()
            && [self.weights_data_type, self.input_data_type, self.output_data_type]
                .into_iter()
                .all(|data_type| matches!(data_type, DataType::BF16 | DataType::F32))
            && mxu_eligible_for_quant
        {
            GemmDispatchPath::Mxu
        } else {
            GemmDispatchPath::Simdgroup
        };
        self.encode_dispatch_path(arguments, path, encoder)
    }

    pub fn encode_dispatch_path<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        path: GemmDispatchPath,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        if matches!(path, GemmDispatchPath::Mxu) {
            assert!(
                encoder.context().device.supports_mxu(),
                "GemmDispatchPath::Mxu requested on hardware without MXU support",
            );
            assert!(
                [self.weights_data_type, self.input_data_type, self.output_data_type]
                    .into_iter()
                    .all(|data_type| matches!(data_type, DataType::BF16 | DataType::F32)),
                "GemmDispatchPath::Mxu requires BF16 or F32 data types, got weights {:?}, input {:?}, output {:?}",
                self.weights_data_type,
                self.input_data_type,
                self.output_data_type,
            );
        }

        let is_quant = !matches!(arguments.b, MatmulB::FullPrecision { .. });
        if is_quant {
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
            if !arguments.b_transpose || arguments.b_leading_dimension.is_some() || arguments.b_offset != 0 {
                return Err(MatmulError::UnsupportedLayout {
                    path: "QuantGemm",
                }
                .into());
            }
        }

        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let output_transform = arguments.d_transform.mask();

        let b_prologue = arguments.b.b_prologue();
        let bits_per_b = arguments.b.bits_per_b();
        let group_size = arguments.b.group_size();

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        let use_mxu = matches!(path, GemmDispatchPath::Mxu);

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                let tiling = if use_mxu {
                    select_mxu_tiling(m, n)
                } else {
                    select_simdgroup_tiling(m, n, k)
                };

                let threadgroups_per_row = n.div_ceil(tiling.block_n());
                let threadgroups_per_column = m.div_ceil(tiling.block_m());

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

                let alignment =
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);

                let default_ldb = if b_transpose {
                    k
                } else {
                    n
                };
                let params = GemmParams {
                    M: m,
                    N: n,
                    K: k,
                    leading_dimension_a: k,
                    leading_dimension_b: b_leading_dimension.unwrap_or(default_ldb),
                    leading_dimension_d: n,
                    threadgroups_per_row,
                    threadgroups_per_column,
                    aligned_inner_iterations: k / tiling.block_k(),
                    use_morton,
                    ab_scale,
                };

                let specialization = GemmSpecialization {
                    weights_data_type: self.weights_data_type,
                    tiling,
                    use_mxu,
                    output_transform,
                    alignment,
                    transpose_b: b_transpose,
                    b_prologue,
                    bits_per_b,
                    group_size,
                };
                specialization.validate()?;
                let kernel = self.get_or_create(encoder.context(), specialization)?;
                kernel.encode(
                    (a, a_offset),
                    (weights, b_offset),
                    &mut *d,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    output_bias,
                    rht_factors,
                    std::slice::from_ref(&params),
                    group_count_x,
                    group_count_y,
                    1, // group_count_z: no split-K on this path
                    encoder,
                );
            },
            quant_b @ (MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                ..
            }) => {
                let (weights, scales, biases, zero_points) = match quant_b {
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
                    _ => unreachable!(),
                };

                let tiling = if use_mxu {
                    select_mxu_quant_tiling(m, n, group_size.unwrap_or(0))
                } else {
                    select_quant_tiling(m, n, group_size.unwrap_or(0))
                };
                let alignment =
                    GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, k % tiling.block_k() == 0);
                let params = quant_params(m, n, k, tiling, use_mxu, group_size.unwrap_or(0), ab_scale);
                let group_count_x = n.div_ceil(tiling.block_n());
                let group_count_y = m.div_ceil(tiling.block_m());

                // Split-K when the base tiling under-fills the GPU; scale+bias epilogue is fused
                // into the reduce and RHT runs as a post-pass. Covers both the simdgroup and MXU
                // paths. ACCUMULATE (read-modify-write on d) and zero-point quant are unsupported.
                let split_k = if zero_points.is_none() && !output_transform.contains(GemmDTransform::ACCUMULATE) {
                    select_split_k(m, n, k, tiling, use_mxu, group_size.unwrap_or(0))
                } else {
                    1
                };
                if split_k > 1 {
                    return self.encode_quant_split_k(
                        a,
                        a_offset,
                        weights,
                        b_offset,
                        scales,
                        biases,
                        &mut *d,
                        m,
                        n,
                        k,
                        ab_scale,
                        use_mxu,
                        tiling,
                        b_prologue,
                        bits_per_b,
                        group_size,
                        split_k,
                        output_transform,
                        output_bias,
                        rht_factors,
                        encoder,
                    );
                }

                let specialization = GemmSpecialization {
                    weights_data_type: self.weights_data_type,
                    tiling,
                    use_mxu,
                    output_transform,
                    alignment,
                    transpose_b: true,
                    b_prologue,
                    bits_per_b,
                    group_size,
                };
                specialization.validate()?;
                let kernel = self.get_or_create(encoder.context(), specialization)?;
                kernel.encode(
                    (a, a_offset),
                    (weights, b_offset),
                    &mut *d,
                    scales,
                    biases,
                    zero_points,
                    output_bias,
                    rht_factors,
                    std::slice::from_ref(&params),
                    group_count_x,
                    group_count_y,
                    1, // group_count_z: split-K (>1) returns early above
                    encoder,
                );
            },
        }

        Ok(())
    }

    /// Split-K dispatch: ONE kernel launch over a 3D grid whose z-axis is the split factor, so
    /// all `split_k * base_tiles` threadgroups run concurrently and fill an under-occupied GPU on
    /// tall-skinny / small-N prefill. Each threadgroup takes its partition from
    /// `threadgroup_position.z`, reduces only its K-slice (output_transform NONE), and writes a
    /// partial into `temp[partition]`; the partials are then summed into `d`.
    #[allow(clippy::too_many_arguments)]
    fn encode_quant_split_k<'x, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        a: &TB,
        a_offset: usize,
        weights: &Allocation<Metal>,
        b_offset: usize,
        scales: Option<&Allocation<Metal>>,
        biases: Option<&Allocation<Metal>>,
        d: &mut Allocation<Metal>,
        m: u32,
        n: u32,
        k: u32,
        ab_scale: f32,
        use_mxu: bool,
        tiling: GemmTiling,
        b_prologue: crate::backends::common::gpu_types::gemm::GemmBPrologueKind,
        bits_per_b: Option<u32>,
        group_size: Option<u32>,
        split_k: u32,
        output_transform: GemmDTransform,
        output_bias: Option<&Allocation<Metal>>,
        rht_factors: Option<&Allocation<Metal>>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let kp = k / split_k;
        // K-step of the quant inner loop (QUANT_BK=group_size on MXU, block_k on simdgroup);
        // select_split_k guarantees kp is a multiple of it.
        let k_step = quant_k_step(tiling, use_mxu, group_size.unwrap_or(0)).unwrap_or(1);
        let base_gx = n.div_ceil(tiling.block_n());
        let base_gy = m.div_ceil(tiling.block_m());
        // Partitions are step-aligned, so force the K-alignment bit: this compiles out the
        // leftover-K tail (whose full-K `leftover_block_depth` would otherwise be wrong).
        let alignment = GemmAlignment::new(m % tiling.block_m() == 0, n % tiling.block_n() == 0, true);
        let part_spec = GemmSpecialization {
            weights_data_type: self.weights_data_type,
            tiling,
            use_mxu,
            output_transform: GemmDTransform::empty(),
            alignment,
            transpose_b: true,
            b_prologue,
            bits_per_b,
            group_size,
        };
        part_spec.validate()?;

        let elem = (m as usize) * (n as usize);
        let slice_bytes = elem * self.output_data_type.size_in_bytes();
        // Pooled GPU-timeline scratch, recycled across calls to avoid a fresh device allocation per
        // dispatch. Written by the partition GEMM and consumed by the reduce; never read on the CPU.
        let mut temp = encoder.allocate_scratch(split_k as usize * slice_bytes)?;

        let params = GemmParams {
            M: m,
            N: n,
            K: k,
            leading_dimension_a: k,
            leading_dimension_b: k,
            leading_dimension_d: n,
            threadgroups_per_row: base_gx,
            threadgroups_per_column: base_gy,
            aligned_inner_iterations: kp / k_step,
            use_morton: false,
            ab_scale: 1.0, // partials are raw; the epilogue is applied once after the reduce
        };
        // Single launch over a native 3D grid (z == split_k); the kernel reads its partition from
        // threadgroup_position.z and writes temp[partition * m * n + ..].
        let part_kernel = self.get_or_create(encoder.context(), part_spec)?;
        part_kernel.encode(
            (a, a_offset),
            (weights, b_offset),
            &mut temp,
            scales,
            biases,
            None::<&Allocation<Metal>>,
            None::<&Allocation<Metal>>,
            None::<&Allocation<Metal>>,
            std::slice::from_ref(&params),
            base_gx,
            base_gy,
            split_k,
            encoder,
        );

        // Sum the split_k partials into d in a single dispatch; the reduce processes 4 elements
        // per thread (M*N is a multiple of 4 since N is a multiple of the block-N tile).
        debug_assert_eq!(elem % 4, 0, "split-K reduce requires M*N divisible by 4");
        let group_count = ((elem as u32) / 4).div_ceil(256);
        // Fuse the elementwise epilogue (scale, then per-column bias) into the reduce, matching the
        // main GEMM finalize. For quant, BIAS and RHT are mutually exclusive (asserted upstream).
        let has_bias = output_transform.contains(GemmDTransform::BIAS);
        let out_scale = if output_transform.contains(GemmDTransform::SCALE) {
            ab_scale
        } else {
            1.0
        };
        let reduce = if has_bias {
            &self.split_k_reduce_bias
        } else {
            &self.split_k_reduce
        };
        let bias_arg = if has_bias {
            output_bias
        } else {
            None
        };
        reduce.encode((&temp, 0usize), &mut *d, bias_arg, elem as u32, split_k, group_count, n, out_scale, encoder);

        // RHT is a cross-column transform — applied separately, as the main GEMM finalize also does.
        if output_transform.contains(GemmDTransform::RHT) {
            if let Some(factors) = rht_factors {
                self.hadamard.encode(&mut *d, factors, n, m, encoder);
            }
        }
        Ok(())
    }
}

fn quant_params(
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
        // The MXU quant K-loop steps by QUANT_BK == group_size; the simdgroup one steps by the
        // tile's block_k. (FP uses block_k on both, but FP doesn't go through quant_params.)
        aligned_inner_iterations: quant_k_step(tiling, use_mxu, group_size).map_or(0, |step| k / step),
        use_morton: false,
        ab_scale,
    }
}

/// K-step (in elements) of the quant inner loop: `group_size` (= QUANT_BK) on the MXU path,
/// the tile's `block_k` on the simdgroup path. A partition's K-slice must be a multiple of this.
fn quant_k_step(
    tiling: GemmTiling,
    use_mxu: bool,
    group_size: u32,
) -> Option<u32> {
    let step = if use_mxu {
        group_size
    } else {
        tiling.block_k()
    };
    (step != 0).then_some(step)
}

/// Split-K factor: target ~512 threadgroups by splitting the K reduction, but only when the base
/// tiling under-fills the GPU. Returns 1 (no split) unless every partition is group- and
/// block_k-aligned, so partition pointer offsets are exact and the leftover-K tail compiles out.
fn select_split_k(
    m: u32,
    n: u32,
    k: u32,
    tiling: GemmTiling,
    use_mxu: bool,
    group_size: u32,
) -> u32 {
    let base_tiles = n.div_ceil(tiling.block_n()) * m.div_ceil(tiling.block_m());
    if base_tiles == 0 {
        return 1;
    }
    let mut split_k = (512 / base_tiles).max(1);
    // Each partition's K-slice must be a whole number of quant K-steps (and groups). On MXU the
    // step is group_size; on simdgroup it's lcm(group_size, block_k) (powers of two → max).
    let step = match quant_k_step(tiling, use_mxu, group_size) {
        Some(s) => s,
        None => return 1,
    };
    let align = if use_mxu {
        step
    } else {
        step.max(group_size)
    };
    split_k = split_k.min((k / align).max(1));
    while split_k > 1 && !k.is_multiple_of(split_k * align) {
        split_k -= 1;
    }
    split_k
}

fn select_simdgroup_tiling(
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

fn select_mxu_tiling(
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

fn select_mxu_quant_tiling(
    m: u32,
    n: u32,
    group_size: u32,
) -> GemmTiling {
    if m >= 256 && n >= 128 && GemmTiling::Tile128x128x256_Simdgroups4x4.fits_quant_group_size(group_size) {
        GemmTiling::Tile128x128x256_Simdgroups4x4
    } else if n < 64 {
        GemmTiling::Tile64x32x256_Simdgroups4x1
    } else if m < 64 {
        GemmTiling::Tile32x64x256_Simdgroups2x2
    } else {
        GemmTiling::Tile64x64x256_Simdgroups2x2
    }
}

fn select_quant_tiling(
    m: u32,
    n: u32,
    group_size: u32,
) -> GemmTiling {
    if group_size < 32 {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else if m < 32 {
        GemmTiling::Tile8x32x32_Simdgroups1x1
    } else if m >= 64 && n <= 2048 {
        // BM=32 (Tile32x32x32) doubles the threadgroup count vs BM=64 for small-N / down-proj
        // shapes (e.g. 64x3584x1024), raising occupancy.
        GemmTiling::Tile32x32x32_Simdgroups2x2
    } else if m >= 64 && n >= 6144 && n.is_multiple_of(64) {
        GemmTiling::Tile64x64x32_Simdgroups2x2
    } else {
        GemmTiling::Tile32x32x32_Simdgroups2x2
    }
}
