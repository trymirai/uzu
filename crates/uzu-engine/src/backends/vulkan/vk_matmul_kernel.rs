use std::{
    collections::{HashMap, hash_map::Entry},
    hash::Hash,
};

use futures::executor::block_on;

use super::{
    Error, GemmBPrologue, GemmInt8K16VulkanKernel, GemmInt8NvVulkanKernel, GemmInt8VulkanKernel,
    GemmPrepareBLayoutVulkanKernel, GemmTensorVulkanKernel, GemmVulkanKernel, GemvInt8Native128VulkanKernel,
    GemvInt8VulkanKernel, GemvVulkanKernel, TrellisGemvVulkanKernel, VkBuffer, VkCommandBufferEncoding, VkContext,
    VkDeviceCapabilities, VkGemvTile, Vulkan,
    gpu_types::{gemm::GemmTiling, gemm_b_layout::GemmBLayout},
};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::{
            gemm::{GemmBPrologueKind, GemmDTransform},
            hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
            quantization::QuantizationMode,
        },
        kernel::{
            ActivationQuantization, ActivationTransform,
            activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
            matmul::{
                ActivationFormat, Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulKernel, MatmulShape,
                QuantParamsLayout, TrellisFormat,
            },
        },
    },
    data_type::DataType,
};

/// K columns per GemmTrellis step; its single-partition dispatch defines only whole steps.
const TRELLIS_K_STEP: u32 = 64;

/// The common MatmulKernel over Gemv, Gemm (scalar and cooperative-matrix cores), their A8 variants and TrellisGemv,
/// each pipeline compiled on first use for its specializations.
pub struct VkMatmulKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    gemv: HashMap<([bool; 4], GemmBPrologue, VkGemvTile, GemmBLayout), GemvVulkanKernel>,
    gemm: HashMap<([bool; 3], GemmBPrologue, GemmTiling, u32), GemmVulkanKernel>,
    gemm_tensor: HashMap<([bool; 3], GemmBPrologue, GemmTiling), GemmTensorVulkanKernel>,
    gemv_int8: HashMap<([bool; 3], GemmBPrologue, u32, u32, GemmBLayout), GemvInt8VulkanKernel>,
    gemv_int8_native_128: HashMap<([bool; 3], GemmBPrologue, u32, u32, GemmBLayout), GemvInt8Native128VulkanKernel>,
    gemm_int8: HashMap<([bool; 3], GemmBPrologue, u32, u32, u32, GemmBLayout), GemmInt8VulkanKernel>,
    gemm_int8_nv: HashMap<([bool; 3], GemmBPrologue, u32, u32, u32, GemmBLayout), GemmInt8NvVulkanKernel>,
    gemm_int8_k16: HashMap<([bool; 3], GemmBPrologue, u32, u32, u32, GemmBLayout), GemmInt8K16VulkanKernel>,
    trellis: HashMap<TrellisFormat, TrellisGemvVulkanKernel>,
    output_rht: HashMap<bool, ActivationTransform<Vulkan>>,
}

impl MatmulKernel for VkMatmulKernel {
    type Backend = Vulkan;
    type Error = Error;
    type WeightsLayout = GemmBLayout;

    fn new(
        _context: &VkContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, Error> {
        let data_types = [weights_data_type, input_data_type, output_data_type];
        if !data_types.iter().all(|data_type| matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)) {
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
            gemm_tensor: HashMap::new(),
            gemv_int8: HashMap::new(),
            gemv_int8_native_128: HashMap::new(),
            gemm_int8: HashMap::new(),
            gemm_int8_nv: HashMap::new(),
            gemm_int8_k16: HashMap::new(),
            trellis: HashMap::new(),
            output_rht: HashMap::new(),
        })
    }

    /// A8 where the integer kernels take the shape.
    fn select_activation_quantization(
        &self,
        shape: &MatmulShape,
        context: &VkContext,
    ) -> Option<ActivationQuantization> {
        let (bits, group_size) = (shape.b_bits?, shape.b_group_size?);
        let scale_bias = shape.b_prologue == GemmBPrologueKind::ScaleBiasDequant;
        let quantization = ActivationQuantization::new(
            ACTIVATION_SCALE_GROUP_SIZE,
            group_size,
            scale_bias,
            Int8CodeLayout::for_right_bits(bits)?,
        )?;
        let capabilities = context.physical_device().capabilities;
        (capabilities.contains(VkDeviceCapabilities::INTEGER_DOT_PRODUCT)
            && matches!(self.input_data_type, DataType::BF16 | DataType::F16)
            && self.output_data_type == self.input_data_type
            && !shape.a_full_precision
            && !shape.gathered
            && match shape.params_layout {
                Some(QuantParamsLayout::GroupOutput) => true,
                Some(QuantParamsLayout::OutputGroup) => {
                    capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT)
                },
                None => false,
            }
            && int8_scheme(shape.b_prologue, bits)
            && matches!(group_size, 32 | 64 | 128)
            && shape.b_transpose
            && shape.b_leading_dimension.is_none()
            && shape.k.is_multiple_of(ACTIVATION_SCALE_GROUP_SIZE)
            && shape.k.is_multiple_of(group_size))
        .then_some(quantization)
    }

    /// A8 from 2 rows on integer GEMM or unsigned-dot GEMV devices, otherwise up to 8 rows.
    fn select_activation_format(
        &self,
        bf16_shape: &MatmulShape,
        weights_layout: GemmBLayout,
        context: &VkContext,
    ) -> ActivationFormat {
        let a8_shape = MatmulShape {
            a_full_precision: false,
            ..*bf16_shape
        };
        let capabilities = context.physical_device().capabilities;
        let rows = bf16_shape.m >= 1
            && (capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT)
                || (bf16_shape.m >= 2
                    && (bf16_shape.m <= 8 || capabilities.intersects(VkDeviceCapabilities::INT8_TENSOR_CORES))));
        let int8 = match weights_layout {
            GemmBLayout::Tiled => bf16_shape.m >= 2,
            GemmBLayout::RowMajor => rows && self.select_activation_quantization(&a8_shape, context).is_some(),
        };
        match int8 {
            true => ActivationFormat::Int8,
            false => ActivationFormat::Bf16,
        }
    }

    /// Packs codes into tiles for integer GEMM through one temporary copy; unsigned-dot devices keep rows for GEMV.
    fn optimize_weights_layout(
        &self,
        a8_shape: &MatmulShape,
        weights: &mut <Vulkan as Backend>::GlobalBuffer,
        context: &VkContext,
    ) -> Result<GemmBLayout, Error> {
        let (n, k) = (a8_shape.n, a8_shape.k);
        let capabilities = context.physical_device().capabilities;
        let Some(bits) = a8_shape.b_bits.filter(|_| {
            capabilities.intersects(VkDeviceCapabilities::INT8_TENSOR_CORES)
                && !capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT)
                && n.is_multiple_of(32)
                && k.is_multiple_of(128)
                && self.select_activation_quantization(a8_shape, context).is_some()
        }) else {
            return Ok(GemmBLayout::RowMajor);
        };
        let bytes = u64::from(n) * u64::from(k) * u64::from(bits) / 8;
        let rows = context.create_buffer(bytes as usize)?;
        let mut command_buffer = context.create_command_buffer(Some("prepare GEMM B layout"), None)?;
        command_buffer.encode_copy(weights, 0..bytes, &rows, 0)?;
        let kernel = GemmPrepareBLayoutVulkanKernel::new(&context.arc())?;
        // SAFETY: both ranges hold the n x k codes, which the copy above wrote to `rows`; the buffers do not alias.
        unsafe { kernel.encode((&rows, 0..bytes), (&*weights, 0..bytes), k, n, bits / 2, &mut command_buffer) };
        block_on(command_buffer.try_end_encoding()?.submit().wait_until_completed())?;
        Ok(GemmBLayout::Tiled)
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
        let context = command_buffer.context().arc();
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
        // Full-precision A starts `offset` elements into its range, as the CPU reads it; A8 is its codes, scales, the
        // group sums ScaleBias needs and the code layout.
        let (a_range, int8) = match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => (Some(VkBuffer::resolve_ref(values.subrange(offset * self.input_data_type.size_in_bytes()..))), None),
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums,
                scale_group_size,
                code_layout,
            } => (
                None,
                Some((
                    VkBuffer::resolve_ref(values),
                    VkBuffer::resolve_ref(scales),
                    group_sums.map(VkBuffer::resolve_ref),
                    scale_group_size,
                    code_layout,
                )),
            ),
            MatmulA::Trellis {
                ..
            } => {
                return Err(precondition(
                    "Matmul",
                    "full-precision or A8 A with full-precision or quantized B, Trellis A with Trellis B",
                ));
            },
        };
        let (bias, rht_factors) = (ops.bias, ops.rht_factors);
        if rht_factors.is_some() && !n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            return Err(precondition("Matmul", "n % 32 == 0 under output RHT"));
        }
        let ops = match rht_factors {
            Some(_) => ops.without(GemmDTransform::RHT | GemmDTransform::BIAS),
            None => ops,
        };
        let mask = ops.mask();
        let flags =
            [GemmDTransform::SCALE, GemmDTransform::BIAS, GemmDTransform::SOFT_CAP].map(|bit| mask.contains(bit));
        let [has_scale, has_bias, has_soft_cap] = flags;
        let (ab_scale, soft_cap) = (has_scale.then_some(ops.ab_scale), ops.soft_cap);
        let bias_range = ops.bias.map(VkBuffer::resolve_ref);
        let gathered = gather_indices.is_some();
        let gather = gather_indices.map(VkBuffer::resolve_ref);
        let d_range = VkBuffer::resolve_ref(d.as_ref());
        let physical_device = context.physical_device();
        let capabilities = physical_device.capabilities;

        // B's words, prologue, metadata planes with strides and full-precision element strides; quantized B is [n, k].
        let b_layout = b.weights_layout();
        let (b_range, prologue, planes, b_strides) = match b {
            MatmulB::FullPrecision {
                b,
            } => {
                let leading_dimension = b_leading_dimension.unwrap_or(if b_transpose {
                    k
                } else {
                    n
                });
                let strides = match b_transpose {
                    true => (leading_dimension, 1),
                    false => (1, leading_dimension),
                };
                let prologue = (GemmBPrologueKind::FullPrecision, QuantizationMode::U8, false, 0);
                (VkBuffer::resolve_ref(b), prologue, None, Some(strides))
            },
            MatmulB::Quantized(quantized) => {
                let mode = match quantized.mode {
                    QuantizationMode::I8 => QuantizationMode::U8,
                    mode => mode,
                };
                let prologue = (quantized.prologue(), mode, quantized.signed_codes, quantized.group_size);
                let scale_strides = quantized.params.scale_strides();
                let zero_point_strides = quantized.zero_points().map(|_| quantized.zero_point_strides());
                let planes = (
                    VkBuffer::resolve_ref(quantized.scales),
                    quantized.biases().map(VkBuffer::resolve_ref),
                    quantized.zero_points().map(VkBuffer::resolve_ref),
                    (scale_strides.output_stride, scale_strides.group_stride),
                    zero_point_strides.map(|strides| (strides.output_stride, strides.group_stride)),
                );
                (VkBuffer::resolve_ref(quantized.codes), prologue, Some(planes), None)
            },
            MatmulB::Trellis {
                ..
            } => unreachable!("Trellis B returned above"),
        };
        let (scheme, mode, signed_codes, group_size) = prologue;
        let quantized = scheme != GemmBPrologueKind::FullPrecision;
        let bits = if mode == QuantizationMode::U4 {
            4
        } else {
            8
        };
        let (scales, biases, zero_points, scale_strides, zero_point_strides) = match planes {
            Some((scales, biases, zero_points, scale_strides, zero_point_strides)) => {
                (Some(scales), biases, zero_points, Some(scale_strides), zero_point_strides)
            },
            None => (None, None, None, None, None),
        };
        let (scale_output, scale_group) = scale_strides.unzip();
        let (zero_point_output, zero_point_group) = zero_point_strides.unzip();
        // Gemm's loaders stage contiguous B rows; quantized ones whole 16-value runs of a group from byte-aligned rows.
        let gemm_layout = match quantized {
            false => b_strides == Some((k, 1)),
            true => (mode != QuantizationMode::U4 || k.is_multiple_of(2)) && group_size.is_multiple_of(16),
        };
        // Tiled B, whole 8 x 64 tiles of quantized codes, is read by the A8 kernels and Gemv; Gemm stages rows.
        if b_layout == GemmBLayout::Tiled
            && (!quantized || !n.is_multiple_of(32) || !k.is_multiple_of(128) || (int8.is_none() && !gathered && m > 8))
        {
            return Err(precondition(
                "Matmul",
                "Tiled B quantized in whole 32 x 128 tiles, with A8 A or at most 8 rows of full-precision A",
            ));
        }

        // SAFETY (every dispatch below): the common contract makes callers pass A of [m, k] elements from `offset` or
        // A8 codes of [m, k] with their [m, k / 128] scales and [m, k / group] group sums, B of n rows at the given
        // strides or n rows of k packed codes with their metadata planes at the given strides, D of [m, n], n bias
        // values and [m, n] gather indices below B's rows, D aliasing nothing; A8 split K's partials are fresh scratch
        // of their shape.
        if let Some((codes, a_scales, group_sums, scale_group_size, code_layout)) = int8 {
            // Up to 8 rows and unsigned-dot devices take GemvInt8, the others GemmInt8.
            if gathered
                || !matches!(group_size, 32 | 64 | 128)
                || scale_group_size != ACTIVATION_SCALE_GROUP_SIZE
                || !k.is_multiple_of(ACTIVATION_SCALE_GROUP_SIZE)
                || Some(code_layout) != Int8CodeLayout::for_right_bits(bits)
                || !int8_scheme(scheme, bits)
                || (scheme == GemmBPrologueKind::ScaleBiasDequant) != group_sums.is_some()
            {
                return Err(precondition(
                    "Matmul",
                    "A8 without gather against U4 or U8 symmetric B in groups of 32, 64 or 128, 128-value activation \
                     scales, k % 128 == 0, the weights' code layout and group sums exactly for ScaleBias",
                ));
            }
            let (scales, (scale_output, scale_group)) =
                (scales.expect("quantized B has scales"), scale_strides.expect("quantized B has scale strides"));
            if m <= 8
                || !capabilities.intersects(VkDeviceCapabilities::INT8_TENSOR_CORES)
                || capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT)
            {
                let rows = match m {
                    0..=2 => 4,
                    3..=4 => 2,
                    _ => 1,
                };
                let input_rows = m.clamp(1, 8);
                // The same arguments for both entry points: subgroups of 32, or native subgroups of 128 lanes.
                macro_rules! gemv_int8 {
                    ($cache:expr, $kernel:ident) => {
                        let kernel = cached(&mut $cache, (flags, prologue, rows, input_rows, b_layout), || {
                            $kernel::new(
                                &context,
                                weights_type,
                                output_type,
                                has_scale,
                                has_bias,
                                has_soft_cap,
                                scheme,
                                mode,
                                signed_codes,
                                group_size,
                                rows,
                                input_rows,
                                b_layout,
                                capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT),
                            )
                        })?;
                        unsafe {
                            kernel.encode(
                                b_range,
                                scales,
                                biases,
                                zero_points,
                                codes,
                                a_scales,
                                group_sums,
                                d_range,
                                bias_range,
                                k,
                                n,
                                m,
                                scale_output,
                                scale_group,
                                zero_point_output,
                                zero_point_group,
                                ab_scale,
                                soft_cap,
                                command_buffer,
                            )
                        };
                    };
                }
                if capabilities.contains(VkDeviceCapabilities::GEMV_INT8_NATIVE_128) {
                    gemv_int8!(self.gemv_int8_native_128, GemvInt8Native128VulkanKernel);
                } else {
                    gemv_int8!(self.gemv_int8, GemvInt8VulkanKernel);
                }
            } else {
                let limits = &physical_device.properties.limits;
                let element_fold = capabilities.contains(VkDeviceCapabilities::INT8_ELEMENT_FOLD);
                let wide_groups =
                    limits.max_compute_work_group_invocations >= 256 && limits.max_compute_work_group_size[1] >= 8;
                let row_block = match m {
                    0..=16 => 16,
                    17..=32 => 32,
                    _ if wide_groups
                        && limits.max_compute_shared_memory_size
                            >= if element_fold {
                                24_576
                            } else {
                                36_864
                            } =>
                    {
                        64
                    },
                    _ => 32,
                };
                let num_simdgroups = if row_block == 64 || (row_block == 32 && element_fold && wide_groups) {
                    8
                } else {
                    4
                };
                let tiles = n.div_ceil(64) * m.div_ceil(row_block);
                let splits = int8_splits(tiles, k);
                // Split K's partial sums and fields, the command buffer's scratch.
                let partials = (splits > 1)
                    .then(|| command_buffer.allocate_scratch_for_shape(&[splits * (m * n + tiles)], DataType::U32))
                    .transpose()?;
                let partials_range = partials.as_ref().map(VkBuffer::resolve_ref);
                // One whole-K phase, or split K's partial phase then its reduction, ordered by the `partials` barrier.
                // GemmInt8Nv folds weight groups in registers; the other cores need fold scratch in shared memory.
                let shared_words = match element_fold {
                    true => 2 * (row_block + 64) * 5,
                    false => 2 * (row_block + 64) * 5 + 4 * 64 * (row_block / 16),
                };
                let phases = if splits == 1 {
                    0..1
                } else {
                    1..3
                };
                // The same arguments for every core and phase.
                macro_rules! gemm_int8 {
                    ($cache:expr, $kernel:ident, $phase:ident) => {{
                        let kernel =
                            cached(&mut $cache, (flags, prologue, row_block, $phase, splits, b_layout), || {
                                $kernel::new(
                                    &context,
                                    weights_type,
                                    output_type,
                                    has_scale,
                                    has_bias,
                                    has_soft_cap,
                                    scheme,
                                    mode,
                                    signed_codes,
                                    group_size,
                                    row_block,
                                    shared_words,
                                    $phase,
                                    splits,
                                    b_layout,
                                    num_simdgroups,
                                )
                            })?;
                        unsafe {
                            kernel.encode(
                                b_range.clone(),
                                scales.clone(),
                                biases.clone(),
                                zero_points.clone(),
                                codes.clone(),
                                a_scales.clone(),
                                group_sums.clone(),
                                d_range.clone(),
                                bias_range.clone(),
                                partials_range.clone(),
                                k,
                                n,
                                m,
                                scale_output,
                                scale_group,
                                zero_point_output,
                                zero_point_group,
                                ab_scale,
                                soft_cap,
                                command_buffer,
                            )
                        }
                    }};
                }
                for phase in phases {
                    if element_fold {
                        gemm_int8!(self.gemm_int8_nv, GemmInt8NvVulkanKernel, phase);
                    } else if capabilities.contains(VkDeviceCapabilities::INT8_TENSOR_CORES_K16) {
                        gemm_int8!(self.gemm_int8_k16, GemmInt8K16VulkanKernel, phase);
                    } else {
                        gemm_int8!(self.gemm_int8, GemmInt8VulkanKernel, phase);
                    }
                }
            }
        } else {
            let a_range = a_range.expect("full-precision A outside A8");
            if gathered || m <= 8 || !gemm_layout {
                let tile = match (quantized, gemm_layout) {
                    (false, true) => VkGemvTile::dense(n, k),
                    (false, false) => VkGemvTile::reader(),
                    (true, _) => VkGemvTile::quantized_shape(
                        physical_device.properties.vendor_id,
                        bits,
                        group_size,
                        m,
                        n,
                        k,
                        matches!(input_type, DataType::BF16 | DataType::F16) && output_type == input_type,
                        gathered,
                        b_layout,
                    ),
                };
                let kernel = cached(
                    &mut self.gemv,
                    ([has_scale, has_bias, has_soft_cap, gathered], prologue, tile, b_layout),
                    || {
                        GemvVulkanKernel::new(
                            &context,
                            input_type,
                            weights_type,
                            output_type,
                            has_scale,
                            has_bias,
                            has_soft_cap,
                            gathered,
                            scheme,
                            mode,
                            signed_codes,
                            group_size,
                            tile.output_row_tile(),
                            tile.num_simdgroups,
                            tile.input_row_tile,
                            tile.reduction_lanes,
                            tile.group_lanes,
                            tile.k_split,
                            tile.input_aligned,
                            tile.full_tile,
                            tile.metal_layout,
                            b_layout,
                        )
                    },
                )?;
                let (b_row_stride, b_column_stride) = b_strides.unzip();
                unsafe {
                    kernel.encode(
                        b_range,
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
                        b_row_stride,
                        b_column_stride,
                        scale_output,
                        scale_group,
                        zero_point_output,
                        zero_point_group,
                        ab_scale,
                        soft_cap,
                        command_buffer,
                    )
                };
            } else if physical_device.float_tensor_cores(input_type) && input_type == weights_type {
                let max_rows = if physical_device.properties.limits.max_compute_shared_memory_size >= 41_984 {
                    u32::MAX
                } else {
                    64
                };
                let tiling = tensor_tiling(m.min(max_rows));
                let kernel = cached(&mut self.gemm_tensor, (flags, prologue, tiling), || {
                    GemmTensorVulkanKernel::new(
                        &context,
                        weights_type,
                        output_type,
                        has_scale,
                        has_bias,
                        has_soft_cap,
                        scheme,
                        mode,
                        signed_codes,
                        group_size,
                        tiling,
                        tiling.simdgroups(),
                        tiling.tensor_staging(),
                    )
                })?;
                unsafe {
                    kernel.encode(
                        b_range,
                        scales,
                        biases,
                        zero_points,
                        a_range,
                        d_range,
                        bias_range,
                        k,
                        n,
                        m,
                        scale_output,
                        scale_group,
                        zero_point_output,
                        zero_point_group,
                        ab_scale,
                        soft_cap,
                        command_buffer,
                    )
                };
            } else {
                let tiling = scalar_tiling(m, n, k, quantized.then_some(group_size));
                let alignment = alignment(tiling, m, n, k);
                let kernel = cached(&mut self.gemm, (flags, prologue, tiling, alignment), || {
                    GemmVulkanKernel::new(
                        &context,
                        input_type,
                        weights_type,
                        output_type,
                        has_scale,
                        has_bias,
                        has_soft_cap,
                        scheme,
                        mode,
                        signed_codes,
                        group_size,
                        tiling,
                        alignment,
                        tiling.simdgroups(),
                        tiling.scalar_staging(true, input_type.size_in_bytes() as u32),
                        tiling.scalar_staging(false, weights_type.size_in_bytes() as u32),
                    )
                })?;
                unsafe {
                    kernel.encode(
                        b_range,
                        scales,
                        biases,
                        zero_points,
                        a_range,
                        d_range,
                        bias_range,
                        k,
                        n,
                        m,
                        scale_output,
                        scale_group,
                        zero_point_output,
                        zero_point_group,
                        ab_scale,
                        soft_cap,
                        command_buffer,
                    )
                };
            }
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

/// The cooperative core's block for m rows: one row block up to 64 rows, so each weight is read once.
fn tensor_tiling(m: u32) -> GemmTiling {
    match m {
        0..=16 => GemmTiling::Tile16x64x64_Simdgroups1x4x2,
        17..=32 => GemmTiling::Tile32x64x64_Simdgroups1x4x2,
        33..=64 => GemmTiling::Tile64x128x32_Simdgroups2x4x1,
        _ => GemmTiling::Tile128x128x32_Simdgroups2x4x1,
    }
}

/// The scalar core's tiling, `group` the quantization group, which must hold whole K slices of the tiling.
fn scalar_tiling(
    m: u32,
    n: u32,
    k: u32,
    group: Option<u32>,
) -> GemmTiling {
    use GemmTiling::*;
    let tiling = match group {
        None if 2_u32.saturating_mul(m.max(n)) > k => Tile64x64x16_Simdgroups2x2,
        None => Tile64x32x32_Simdgroups2x2,
        Some(group) if group < 32 => Tile64x64x16_Simdgroups2x2,
        Some(_) if m < 32 => {
            if m <= 8 || m.is_multiple_of(8) {
                Tile8x32x32_Simdgroups1x1
            } else {
                Tile32x32x32_Simdgroups2x2
            }
        },
        Some(_) if m >= 64 && n >= 6144 && n.is_multiple_of(64) => Tile64x64x32_Simdgroups2x2,
        Some(_) => Tile32x32x32_Simdgroups2x2,
    };
    match group {
        Some(group) if !group.is_multiple_of(tiling.block_k()) => Tile64x64x16_Simdgroups2x2,
        _ => tiling,
    }
}

/// The GemmAlignment bits of whole M, N and K blocks of the scalar tiling.
fn alignment(
    tiling: GemmTiling,
    m: u32,
    n: u32,
    k: u32,
) -> u32 {
    u32::from(m.is_multiple_of(tiling.block_m()))
        | (u32::from(n.is_multiple_of(tiling.block_n())) * 2)
        | (u32::from(k.is_multiple_of(tiling.block_k())) * 4)
}

/// GemmInt8's K splits for narrow grids: at most `TARGET` workgroups, K runs of at least `COLUMNS`.
fn int8_splits(
    tiles: u32,
    k: u32,
) -> u32 {
    const NARROW: u32 = 512;
    const TARGET: u32 = 640;
    const COLUMNS: u32 = 512;
    let fits = |splits: u32| tiles * splits <= TARGET && k.is_multiple_of(128 * splits) && k / splits >= COLUMNS;
    [8, 4, 2].into_iter().find(|&splits| tiles < NARROW && fits(splits)).unwrap_or(1)
}

/// Whether the A8 kernels take B of `scheme` and `bits`: U4 codes under any scheme, U8 symmetric ones.
fn int8_scheme(
    scheme: GemmBPrologueKind,
    bits: u32,
) -> bool {
    match bits {
        4 => scheme != GemmBPrologueKind::FullPrecision,
        8 => scheme == GemmBPrologueKind::ScaleSymmetricDequant,
        _ => false,
    }
}
