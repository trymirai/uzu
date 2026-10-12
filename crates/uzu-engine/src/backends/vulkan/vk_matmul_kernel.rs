use std::{
    collections::{HashMap, hash_map::Entry},
    hash::Hash,
};

use super::{
    Error, GemmVulkanKernel, GemvVulkanKernel, QuantizedGemmVulkanKernel, QuantizedGemvVulkanKernel,
    StridedGemmVulkanKernel, StridedGemvVulkanKernel, TrellisGemvVulkanKernel, VkBuffer, VkCommandBufferEncoding,
    VkContext, VkGemvTile, Vulkan,
};
use crate::{
    backends::common::{
        BufferMut, BufferRef,
        gpu_types::{
            HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMethod, QuantizationMode,
            gemm::{GemmDTransform, GemmTiling},
        },
        kernel::{
            ActivationTransform,
            matmul::{MatmulA, MatmulArguments, MatmulB, MatmulKernel, QuantizedCorrection, TrellisFormat},
        },
    },
    data_type::DataType,
};

/// K columns per Metal GemmTrellis step; its single-partition dispatch defines only whole steps.
const TRELLIS_K_STEP: u32 = 64;

/// The common MatmulKernel over the raw Vulkan entries, each pipeline compiled on first use for its specializations:
/// the epilogue flags (scale, accumulate, bias, soft cap), gather, and the quantization mode, method, stored sign,
/// group and A8 group and code layout. Gather and one or two rows take the Gemv form, larger batches Gemm; output RHT
/// follows the CPU's order, the epilogue without bias, then the transform and the bias.
pub struct VkMatmulKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    gemv: HashMap<([bool; 4], bool, VkGemvTile), GemvVulkanKernel>,
    gemm: HashMap<([bool; 4], GemmTiling, u32), GemmVulkanKernel>,
    strided_gemv: HashMap<([bool; 4], bool), StridedGemvVulkanKernel>,
    strided_gemm: HashMap<[bool; 4], StridedGemmVulkanKernel>,
    quantized_gemv: HashMap<
        ([bool; 4], bool, (QuantizationMode, QuantizationMethod, bool, u32), VkGemvTile),
        QuantizedGemvVulkanKernel,
    >,
    quantized_gemm: HashMap<
        ([bool; 4], (QuantizationMode, QuantizationMethod, bool, u32), GemmTiling, u32, bool),
        QuantizedGemmVulkanKernel,
    >,
    trellis: HashMap<TrellisFormat, TrellisGemvVulkanKernel>,
    output_rht: HashMap<bool, ActivationTransform<Vulkan>>,
}

impl MatmulKernel for VkMatmulKernel {
    type Backend = Vulkan;

    fn new(
        _context: &VkContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, Error> {
        let data_types = [weights_data_type, input_data_type, output_data_type];
        if !data_types.iter().all(|data_type| matches!(data_type, DataType::BF16 | DataType::F32)) {
            return Err(Error::KernelVariant {
                kernel: "Matmul",
                data_types: Box::new(data_types),
            });
        }
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            gemv: HashMap::new(),
            gemm: HashMap::new(),
            strided_gemv: HashMap::new(),
            strided_gemm: HashMap::new(),
            quantized_gemv: HashMap::new(),
            quantized_gemm: HashMap::new(),
            trellis: HashMap::new(),
            output_rht: HashMap::new(),
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Vulkan,
            impl BufferRef<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
            impl BufferMut<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
        >,
        command_buffer: &mut VkCommandBufferEncoding,
    ) -> Result<(), Error> {
        let MatmulArguments {
            a,
            b,
            b_leading_dimension,
            b_transpose,
            output,
            gather_indices,
            m,
            n,
            k,
        } = arguments;
        let context = command_buffer.context().clone();
        let (d, ops, row_stride) = (output.values, output.ops, output.row_stride.unwrap_or(n));
        let (weights_type, input_type, output_type) =
            (self.weights_data_type, self.input_data_type, self.output_data_type);

        if let MatmulB::Trellis {
            codes,
            row_scales,
            codebook,
            format,
        } = b
        {
            let MatmulA::Trellis {
                values,
                column_group_sums,
                scales,
            } = a
            else {
                return Err(precondition("TrellisGemv", "Trellis A with Trellis B"));
            };
            if output_type != DataType::BF16 {
                return Err(Error::KernelVariant {
                    kernel: "TrellisGemv",
                    data_types: Box::new([output_type]),
                });
            }
            let restart_columns = format.restart_columns.unwrap_or(0);
            if !ops.mask().is_empty()
                || !b_transpose
                || b_leading_dimension.is_some()
                || gather_indices.is_some()
                || row_stride < n
                || !k.is_multiple_of(TRELLIS_K_STEP)
                || (restart_columns != 0 && !k.is_multiple_of(restart_columns))
            {
                return Err(precondition(
                    "TrellisGemv",
                    "transposed B without leading dimension, gather or epilogue, row stride >= n, k % 64 == 0 and \
                     whole restart blocks",
                ));
            }
            let kernel = cached(&mut self.trellis, format, || {
                TrellisGemvVulkanKernel::new(&context, format.vector_width, format.transition_bits, restart_columns)
            })?;
            // SAFETY: the common contract makes callers pass Trellis A of [m, k] codes, [m, 4] column-group sums and m
            // scales, n code rows of row_bytes(k), n row scales, the 5-value codebook and BF16 D rows of `row_stride`,
            // D aliasing nothing.
            unsafe {
                kernel.encode(
                    VkBuffer::resolve_ref(values),
                    VkBuffer::resolve_ref(column_group_sums),
                    VkBuffer::resolve_ref(scales),
                    VkBuffer::resolve_ref(codes),
                    VkBuffer::resolve_ref(row_scales),
                    VkBuffer::resolve_ref(codebook),
                    VkBuffer::resolve_ref(d.as_ref()),
                    k,
                    n,
                    m,
                    row_stride,
                    command_buffer,
                )
            };
            return Ok(());
        }

        if row_stride != n {
            return Err(precondition("Matmul", "output row stride == n outside Trellis"));
        }
        let (bias, rht_factors) = (ops.bias, ops.rht_factors);
        if rht_factors.is_some() && !n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            return Err(precondition("Matmul", "n % 32 == 0 under output RHT"));
        }
        let ops = match rht_factors {
            Some(_) => ops.without(GemmDTransform::RHT | GemmDTransform::BIAS),
            None => ops,
        };
        let mask = ops.mask();
        let flags = [GemmDTransform::SCALE, GemmDTransform::ACCUMULATE, GemmDTransform::BIAS, GemmDTransform::SOFT_CAP]
            .map(|bit| mask.contains(bit));
        let [has_scale, accumulate, has_bias, has_soft_cap] = flags;
        let (ab_scale, soft_cap) = (has_scale.then_some(ops.ab_scale), ops.soft_cap);
        let bias_range = ops.bias.map(VkBuffer::resolve_ref);
        let gathered = gather_indices.is_some();
        let gather = gather_indices.map(VkBuffer::resolve_ref);
        // Gather is Gemv's alone; one and two rows take Gemv, larger batches Gemm.
        let gemv = gathered || m <= 8;
        let topology = context.physical_device().asahi_topology;
        let limits = context.physical_device().properties.limits;
        let d_range = VkBuffer::resolve_ref(d.as_ref());
        // Full-precision A starts `offset` elements into its range, as the CPU reads it.
        let a_bytes = |offset: usize| offset * input_type.size_in_bytes()..;

        match (a, b) {
            (
                MatmulA::FullPrecision {
                    values,
                    offset,
                },
                MatmulB::FullPrecision {
                    b,
                },
            ) => {
                let (a_range, b_range) =
                    (VkBuffer::resolve_ref(values.subrange(a_bytes(offset))), VkBuffer::resolve_ref(b));
                let leading_dimension = b_leading_dimension.unwrap_or(if b_transpose {
                    k
                } else {
                    n
                });
                let (b_row_stride, b_inner_stride) = match b_transpose {
                    true => (leading_dimension, 1),
                    false => (1, leading_dimension),
                };
                // SAFETY (every dispatch below): the common contract makes callers pass A of [m, k] elements from
                // `offset`, B of n rows at the given strides, D of [m, n], n bias values and [m, n] gather indices below
                // B's rows, D aliasing nothing.
                match (b_transpose && leading_dimension == k, gemv) {
                    (true, true) => {
                        let tile = VkGemvTile::dense(topology, m, n, k, limits.max_compute_work_group_count[0]);
                        let kernel = cached(&mut self.gemv, (flags, gathered, tile), || {
                            GemvVulkanKernel::new(
                                &context,
                                input_type,
                                weights_type,
                                output_type,
                                has_scale,
                                accumulate,
                                has_bias,
                                has_soft_cap,
                                gathered,
                                tile.output_row_tile(),
                                tile.k_split,
                                tile.input_aligned,
                                tile.full_tile,
                            )
                        })?;
                        unsafe {
                            kernel.encode(
                                b_range,
                                a_range,
                                d_range,
                                bias_range,
                                gather,
                                k,
                                n,
                                m,
                                ab_scale,
                                soft_cap,
                                command_buffer,
                            )
                        };
                    },
                    (true, false) => {
                        let (tiling, alignment, simdgroups) = gemm_tile(m, n, k, None, topology);
                        let kernel = cached(&mut self.gemm, (flags, tiling, alignment), || {
                            GemmVulkanKernel::new(
                                &context,
                                input_type,
                                weights_type,
                                output_type,
                                has_scale,
                                accumulate,
                                has_bias,
                                has_soft_cap,
                                tiling,
                                alignment,
                                simdgroups,
                                gemm_storage(tiling, input_type, true),
                                gemm_storage(tiling, weights_type, false),
                            )
                        })?;
                        unsafe {
                            kernel.encode(
                                b_range,
                                a_range,
                                d_range,
                                bias_range,
                                k,
                                n,
                                m,
                                ab_scale,
                                soft_cap,
                                command_buffer,
                            )
                        };
                    },
                    (false, true) => {
                        let kernel = cached(&mut self.strided_gemv, (flags, gathered), || {
                            StridedGemvVulkanKernel::new(
                                &context,
                                input_type,
                                weights_type,
                                output_type,
                                has_scale,
                                accumulate,
                                has_bias,
                                has_soft_cap,
                                gathered,
                            )
                        })?;
                        unsafe {
                            kernel.encode(
                                b_range,
                                a_range,
                                d_range,
                                bias_range,
                                gather,
                                k,
                                n,
                                m,
                                b_row_stride,
                                b_inner_stride,
                                ab_scale,
                                soft_cap,
                                command_buffer,
                            )
                        };
                    },
                    (false, false) => {
                        let kernel = cached(&mut self.strided_gemm, flags, || {
                            StridedGemmVulkanKernel::new(
                                &context,
                                input_type,
                                weights_type,
                                output_type,
                                has_scale,
                                accumulate,
                                has_bias,
                                has_soft_cap,
                            )
                        })?;
                        unsafe {
                            kernel.encode(
                                b_range,
                                a_range,
                                d_range,
                                bias_range,
                                k,
                                n,
                                m,
                                b_row_stride,
                                b_inner_stride,
                                ab_scale,
                                soft_cap,
                                command_buffer,
                            )
                        };
                    },
                }
            },
            (
                MatmulA::FullPrecision {
                    values,
                    offset,
                },
                MatmulB::Quantized(quantized),
            ) => {
                // Quantized B is always [n, k] codes, as the CPU reads it whatever the transpose and leading dimension.
                let method = match quantized.correction {
                    QuantizedCorrection::Symmetric => QuantizationMethod::ScaleSymmetric,
                    QuantizedCorrection::Biases(_) => QuantizationMethod::ScaleBias,
                    QuantizedCorrection::ZeroPoints(_) => QuantizationMethod::ScaleZeroPoint,
                };
                let mode = match quantized.mode {
                    QuantizationMode::I8 => QuantizationMode::U8,
                    mode => mode,
                };
                let quantization = (mode, method, quantized.signed_codes, quantized.group_size);
                let scale_strides = quantized.params.scale_strides();
                let zero_point_strides = quantized.zero_points().map(|_| quantized.zero_point_strides());
                let (zero_point_output_stride, zero_point_group_stride) = (
                    zero_point_strides.map(|strides| strides.output_stride),
                    zero_point_strides.map(|strides| strides.group_stride),
                );
                let (codes, scales) = (VkBuffer::resolve_ref(quantized.codes), VkBuffer::resolve_ref(quantized.scales));
                let (biases, zero_points) =
                    (quantized.biases().map(VkBuffer::resolve_ref), quantized.zero_points().map(VkBuffer::resolve_ref));
                let (signed_codes, group_size) = (quantized.signed_codes, quantized.group_size);
                // SAFETY (every dispatch below): the common contract makes callers pass n rows of k packed codes, their
                // metadata planes at the given strides, A of [m, k] elements from `offset`, D of [m, n], n bias values and [m, n] gather indices below B's rows, D aliasing nothing.
                let a_range = VkBuffer::resolve_ref(values.subrange(a_bytes(offset)));
                if gemv {
                    let tile = VkGemvTile::quantized_shape(
                        topology,
                        if mode == QuantizationMode::U4 {
                            4
                        } else {
                            8
                        },
                        group_size,
                        m,
                        n,
                        k,
                        input_type == DataType::BF16 && output_type == DataType::BF16,
                        gathered,
                    );
                    let kernel = cached(&mut self.quantized_gemv, (flags, gathered, quantization, tile), || {
                        QuantizedGemvVulkanKernel::new(
                            &context,
                            input_type,
                            weights_type,
                            output_type,
                            has_scale,
                            accumulate,
                            has_bias,
                            has_soft_cap,
                            gathered,
                            mode,
                            method,
                            signed_codes,
                            group_size,
                            tile.output_row_tile(),
                            tile.num_simdgroups,
                            tile.input_row_tile,
                            tile.reduction_lanes,
                            tile.group_lanes,
                            tile.input_aligned,
                            tile.full_tile,
                            tile.metal_layout,
                        )
                    })?;
                    unsafe {
                        kernel.encode(
                            codes,
                            scales,
                            biases,
                            zero_points,
                            a_range,
                            d_range,
                            bias_range,
                            gather,
                            k,
                            n,
                            m,
                            scale_strides.output_stride,
                            scale_strides.group_stride,
                            zero_point_output_stride,
                            zero_point_group_stride,
                            ab_scale,
                            soft_cap,
                            command_buffer,
                        )
                    };
                } else {
                    let (tiling, alignment, simdgroups) = gemm_tile(m, n, k, Some(group_size), topology);
                    let metal_layout = group_size.is_multiple_of([32, 32, 16, 32, 32][tiling as usize])
                        && (mode != QuantizationMode::U4 || k.is_multiple_of(2));
                    let kernel = cached(
                        &mut self.quantized_gemm,
                        (flags, quantization, tiling, alignment, metal_layout),
                        || {
                            QuantizedGemmVulkanKernel::new(
                                &context,
                                input_type,
                                weights_type,
                                output_type,
                                has_scale,
                                accumulate,
                                has_bias,
                                has_soft_cap,
                                mode,
                                method,
                                signed_codes,
                                group_size,
                                tiling,
                                alignment,
                                if metal_layout {
                                    simdgroups
                                } else {
                                    8
                                },
                                gemm_storage(tiling, input_type, true),
                                gemm_storage(tiling, weights_type, false),
                                metal_layout,
                            )
                        },
                    )?;
                    unsafe {
                        kernel.encode(
                            codes,
                            scales,
                            biases,
                            zero_points,
                            a_range,
                            d_range,
                            bias_range,
                            k,
                            n,
                            m,
                            scale_strides.output_stride,
                            scale_strides.group_stride,
                            zero_point_output_stride,
                            zero_point_group_stride,
                            ab_scale,
                            soft_cap,
                            command_buffer,
                        )
                    };
                }
            },
            _ => {
                return Err(precondition(
                    "Matmul",
                    "full-precision A with full-precision or quantized B, Trellis A with Trellis B",
                ));
            },
        }

        if let Some(factors) = rht_factors {
            let transform = cached(&mut self.output_rht, bias.is_some(), || {
                ActivationTransform::output_rht(context.as_ref(), output_type, bias.map(|_| weights_type), true)
            })?;
            transform.encode_fp_in_place(d, factors, bias, m, n, command_buffer);
        }
        Ok(())
    }
}

fn precondition(
    kernel: &'static str,
    condition: &'static str,
) -> Error {
    Error::KernelPrecondition {
        kernel,
        condition,
    }
}

fn cached<K: Eq + Hash, V>(
    cache: &mut HashMap<K, V>,
    key: K,
    create: impl FnOnce() -> Result<V, Error>,
) -> Result<&V, Error> {
    Ok(match cache.entry(key) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(create()?),
    })
}

// Metal gemm/policy's non-MXU choices; no MXU geometry enters Vulkan.
fn gemm_tile(
    m: u32,
    n: u32,
    k: u32,
    group: Option<u32>,
    topology: Option<(u32, u32)>,
) -> (GemmTiling, u32, u32) {
    use GemmTiling::*;
    let tiling = match group {
        None if 2_u32.saturating_mul(m.max(n)) > k => Tile64x64x16_Simdgroups2x2,
        None => Tile64x32x32_Simdgroups2x2,
        Some(group) if group < 32 => Tile64x64x16_Simdgroups2x2,
        Some(_) if m < 32 => {
            if topology.is_some_and(|(_, family)| family > 8) || m <= 8 || m.is_multiple_of(8) {
                Tile8x32x32_Simdgroups1x1
            } else {
                Tile32x32x32_Simdgroups2x2
            }
        },
        Some(_) if m >= 64 && n >= 6144 && n.is_multiple_of(64) => Tile64x64x32_Simdgroups2x2,
        Some(_) => Tile32x32x32_Simdgroups2x2,
    };
    let index = tiling as usize;
    let alignment = u32::from(m.is_multiple_of([8, 64, 64, 64, 32][index]))
        | (u32::from(n.is_multiple_of([32, 32, 64, 64, 32][index])) * 2)
        | (u32::from(k.is_multiple_of([32, 32, 16, 32, 32][index])) * 4);
    (
        tiling,
        alignment,
        if index == 0 {
            1
        } else {
            4
        },
    )
}

fn gemm_storage(
    tiling: GemmTiling,
    data_type: DataType,
    left: bool,
) -> u32 {
    let index = tiling as usize;
    let rows = if left {
        [8, 64, 64, 64, 32][index]
    } else {
        [32, 32, 64, 64, 32][index]
    };
    rows * ([32, 32, 16, 32, 32][index] + 16 / data_type.size_in_bytes() as u32)
}
