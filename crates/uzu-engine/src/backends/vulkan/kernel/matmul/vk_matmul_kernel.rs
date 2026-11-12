use std::{
    collections::{HashMap, hash_map::Entry},
    hash::Hash,
};

use futures::executor::block_on;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu,
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
                    TrellisFormat,
                },
            },
        },
        vulkan::{
            ActivationQuantizeVulkanKernel, Error, GemmInt8K16VulkanKernel, GemmInt8NvVulkanKernel,
            GemmInt8VulkanKernel, GemmPrepareBLayoutVulkanKernel, GemmTensorVulkanKernel, GemmVulkanKernel,
            GemvInt8VulkanKernel, GemvVulkanKernel, TrellisGemvVulkanKernel, VkBuffer, VkCommandBufferEncoding,
            VkContext, VkDeviceCapabilities, VkKernelError, Vulkan,
            gpu_types::{gemm_b_layout::GemmBLayout, gemm_tiling::GemmTiling},
            kernel::matmul::{VkGemvTile, VkMatmulPlan},
        },
    },
    data_type::DataType,
};

const TRELLIS_K_STEP: u32 = 64;

type GemmBPrologue = (GemmBPrologueKind, QuantizationMode, bool, u32);

pub struct VkMatmulKernel {
    data_types: [DataType; 3],
    gemv: HashMap<([bool; 4], GemmBPrologue, VkGemvTile, GemmBLayout), GemvVulkanKernel>,
    gemm: HashMap<([bool; 3], GemmBPrologue, GemmTiling, u32), GemmVulkanKernel>,
    gemm_tensor: HashMap<([bool; 3], GemmBPrologue, GemmTiling), GemmTensorVulkanKernel>,
    gemv_int8: HashMap<([bool; 5], GemmBPrologue, u32, u32, GemmBLayout), GemvInt8VulkanKernel>,
    gemm_int8: HashMap<([bool; 4], GemmBPrologue, VkMatmulPlan, u32, GemmBLayout), GemmInt8VulkanKernel>,
    gemm_int8_nv: HashMap<([bool; 4], GemmBPrologue, VkMatmulPlan, u32, GemmBLayout), GemmInt8NvVulkanKernel>,
    gemm_int8_k16: HashMap<([bool; 4], GemmBPrologue, VkMatmulPlan, u32, GemmBLayout), GemmInt8K16VulkanKernel>,
    trellis: HashMap<TrellisFormat, TrellisGemvVulkanKernel>,
    output_rht: HashMap<bool, ActivationTransform<Vulkan>>,
    quantize_a: HashMap<(bool, u32), ActivationQuantizeVulkanKernel>,
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
        if !data_types.iter().all(|data_type| matches!(data_type, DataType::BF16 | DataType::F32)) {
            return Err(VkKernelError::KernelVariant {
                kernel: "Matmul",
                data_types: Box::new(data_types),
            }
            .into());
        }
        Ok(Self {
            data_types,
            gemv: HashMap::new(),
            gemm: HashMap::new(),
            gemm_tensor: HashMap::new(),
            gemv_int8: HashMap::new(),
            gemm_int8: HashMap::new(),
            gemm_int8_nv: HashMap::new(),
            gemm_int8_k16: HashMap::new(),
            trellis: HashMap::new(),
            output_rht: HashMap::new(),
            quantize_a: HashMap::new(),
        })
    }

    fn select_activation_quantization(
        &self,
        shape: &MatmulShape,
        context: &VkContext,
    ) -> Option<ActivationQuantization> {
        let capabilities = context.physical_device().capabilities;
        VkMatmulPlan::a8_quantization(shape, self.data_types, capabilities)
    }

    fn activation_preparation(
        &self,
        shape: &MatmulShape,
        _weights_layout: GemmBLayout,
        context: &VkContext,
    ) -> ActivationFormat {
        self.select_activation_quantization(shape, context).map_or(ActivationFormat::Bf16, ActivationFormat::Int8)
    }

    fn select_activation_format(
        &self,
        bf16_shape: &MatmulShape,
        weights_layout: GemmBLayout,
        prepared: ActivationFormat,
        context: &VkContext,
    ) -> ActivationFormat {
        let capabilities = context.physical_device().capabilities;
        match prepared {
            ActivationFormat::Int8(_) if VkMatmulPlan::a8_rows(bf16_shape.m, weights_layout, capabilities) => prepared,
            _ => ActivationFormat::Bf16,
        }
    }

    fn optimize_weights_layout(
        &self,
        a8_shape: &MatmulShape,
        weights: &mut <Vulkan as Backend>::GlobalGpuBuffer,
        context: &VkContext,
    ) -> Result<(GemmBLayout, bool), Error> {
        let prepared_shape = MatmulShape {
            signed_codes: a8_shape.signed_codes || a8_shape.b_bits == Some(8),
            ..*a8_shape
        };
        if self.select_activation_quantization(&prepared_shape, context).is_none() {
            return Ok((GemmBLayout::RowMajor, a8_shape.signed_codes));
        }
        if !a8_shape.signed_codes && prepared_shape.signed_codes {
            let mut codes = wait(context.copyout::<u8>(&*weights))?;
            let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(&mut codes);
            words.iter_mut().for_each(|word| *word ^= 0x8080_8080_8080_8080);
            prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= 0x80);
            let mut signed = context.create_buffer_cpu_gpu(codes.len())?;
            signed.copyin(&codes);
            *weights = signed.handoff();
        }
        let (n, k) = (a8_shape.n, a8_shape.k);
        let capabilities = context.physical_device().capabilities;
        let Some(bits) =
            a8_shape.b_bits.filter(|_| VkMatmulPlan::tiles_weights(a8_shape, self.data_types, capabilities))
        else {
            return Ok((GemmBLayout::RowMajor, prepared_shape.signed_codes));
        };
        let bytes = u64::from(n) * u64::from(k) * u64::from(bits) / 8;
        let tiles = context.create_buffer_gpu(bytes as usize)?;
        let mut command_buffer = context.create_command_buffer(Some("prepare GEMM B layout"), None)?;
        let kernel = GemmPrepareBLayoutVulkanKernel::new(&context.arc())?;
        unsafe { kernel.encode((&*weights, 0..bytes), (&tiles, 0..bytes), k, n, bits / 2, &mut command_buffer) };
        drop(command_buffer.end_encoding().try_submit()?);
        *weights = tiles;
        Ok((GemmBLayout::Tiled, prepared_shape.signed_codes))
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
        let shape = MatmulShape::from_arguments(&arguments);
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
        command_buffer.declare_work(2 * u64::from(m) * u64::from(n) * u64::from(k));
        let (d, ops, row_stride) = (output.values, output.ops, output.row_stride.unwrap_or(n));
        let [weights_type, input_type, output_type] = self.data_types;

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
                return Err(VkKernelError::KernelVariant {
                    kernel: "TrellisGemv",
                    data_types: Box::new([output_type]),
                }
                .into());
            }
            let restart_columns = format.restart_columns.unwrap_or(0);
            let values = VkBuffer::resolve_ref(values);
            let codes = VkBuffer::resolve_ref(codes);
            if !ops.mask().is_empty()
                || !b_transpose
                || b_leading_dimension.is_some()
                || gather_indices.is_some()
                || row_stride < n
                || !k.is_multiple_of(TRELLIS_K_STEP)
                || (restart_columns != 0 && !k.is_multiple_of(restart_columns))
                || !(values.0.device_address() + values.1.start).is_multiple_of(4)
                || codes.1.end - codes.1.start < u64::from(n) * u64::from(format.code_row_bytes(k))
            {
                return Err(precondition(
                    "TrellisGemv",
                    "transposed B without leading dimension, gather or epilogue, row stride >= n, k % 64 == 0, \
                     whole restart blocks, A codes starting 4-byte aligned and n code rows of code_row_bytes(k)",
                ));
            }
            let kernel = cached(&mut self.trellis, format, || {
                TrellisGemvVulkanKernel::new(&context, format.vector_width, format.transition_bits, restart_columns)
            })?;
            unsafe {
                kernel.encode(
                    values,
                    VkBuffer::resolve_ref(column_group_sums),
                    VkBuffer::resolve_ref(scales),
                    codes,
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
        let (a_range, int8) = match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => (Some(VkBuffer::resolve_ref(values.subrange(offset * input_type.size_in_bytes()..))), None),
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums,
                quantization,
            } => (
                None,
                Some((
                    VkBuffer::resolve_ref(values),
                    VkBuffer::resolve_ref(scales),
                    group_sums.map(VkBuffer::resolve_ref),
                    quantization.scale_group_size(),
                    quantization.code_layout(),
                )),
            ),
            MatmulA::PackedHalf {
                ..
            }
            | MatmulA::Trellis {
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
        let gather = gather_indices.map(VkBuffer::resolve_ref);
        let d_range = VkBuffer::resolve_ref(d.as_ref());
        let physical_device = context.physical_device();
        let capabilities = physical_device.capabilities;

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
        let aligned = |(buffer, range): &(&std::sync::Arc<VkBuffer>, std::ops::Range<u64>)| {
            (buffer.device_address() + range.start).is_multiple_of(16)
        };
        if !aligned(&b_range)
            || !a_range.as_ref().is_none_or(aligned)
            || !int8.as_ref().is_none_or(|(codes, ..)| aligned(codes))
        {
            return Err(precondition("Matmul", "A, A8 codes and B ranges starting 16-byte aligned"));
        }
        let (scheme, mode, signed_codes, group_size) = prologue;
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
        let plan = VkMatmulPlan::baseline(&shape, b_layout, self.data_types, physical_device);
        plan.check(&shape, b_layout, self.data_types, physical_device)
            .map_err(|reason| precondition("Matmul", reason))?;
        let quantized_a = match (plan, &a_range) {
            (
                VkMatmulPlan::GemvInt8 {
                    ..
                },
                Some(values),
            ) => {
                let quantization = VkMatmulPlan::a8_quantization(&shape, self.data_types, capabilities)
                    .expect("the plan check took the A8 contract");
                let code_layout = quantization.code_layout();
                let sums = quantization.sum_group_size();
                let codes = command_buffer.allocate_scratch_for_shape(&[m * k], DataType::I8)?;
                let a_scales =
                    command_buffer.allocate_scratch_for_shape(&[m * k / ACTIVATION_SCALE_GROUP_SIZE], DataType::F32)?;
                let group_sums = sums
                    .map(|group| command_buffer.allocate_scratch_for_shape(&[m * k / group], DataType::I32))
                    .transpose()?;
                let key = (code_layout.is_grouped_by_nibble(), sums.unwrap_or(0));
                let kernel = cached(&mut self.quantize_a, key, || {
                    ActivationQuantizeVulkanKernel::new(&context, input_type, key.0, key.1)
                })?;
                unsafe {
                    kernel.encode(
                        values.clone(),
                        VkBuffer::resolve_ref(&codes),
                        VkBuffer::resolve_ref(&a_scales),
                        group_sums.as_ref().map(VkBuffer::resolve_ref),
                        m,
                        k,
                        command_buffer,
                    )
                };
                Some((codes, a_scales, group_sums, code_layout))
            },
            _ => None,
        };
        let int8 = int8.or_else(|| {
            quantized_a.as_ref().map(|(codes, a_scales, group_sums, code_layout)| {
                (
                    VkBuffer::resolve_ref(codes),
                    VkBuffer::resolve_ref(a_scales),
                    group_sums.as_ref().map(VkBuffer::resolve_ref),
                    ACTIVATION_SCALE_GROUP_SIZE,
                    *code_layout,
                )
            })
        });
        let rht = rht_factors.filter(|_| plan.fuses_output_rht()).map(VkBuffer::resolve_ref);
        let (has_bias, bias_range) = match rht {
            Some(_) => (bias.is_some(), bias.map(VkBuffer::resolve_ref)),
            None => (has_bias, bias_range),
        };

        if let Some((codes, a_scales, group_sums, scale_group_size, code_layout)) = int8 {
            if scale_group_size != ACTIVATION_SCALE_GROUP_SIZE
                || Some(code_layout) != Int8CodeLayout::for_right_bits(bits)
                || (scheme == GemmBPrologueKind::ScaleBiasDequant) != group_sums.is_some()
            {
                return Err(precondition(
                    "Matmul",
                    "A8 of 128-value activation scales, the weights' code layout and group sums exactly for ScaleBias",
                ));
            }
            let scales = scales.expect("quantized B has scales");
            let (scale_output, scale_group) = scale_strides.expect("quantized B has scale strides");
            match plan {
                VkMatmulPlan::GemvInt8 {
                    rows,
                    input_rows,
                } => {
                    let gemv_flags = [has_scale, has_bias, has_soft_cap, shape.gathered, rht.is_some()];
                    let kernel =
                        cached(&mut self.gemv_int8, (gemv_flags, prologue, rows, input_rows, b_layout), || {
                            GemvInt8VulkanKernel::new(
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
                                shape.gathered,
                                rht.is_some(),
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
                            gather,
                            rht.clone(),
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
                },
                VkMatmulPlan::GemmInt8 {
                    element_fold,
                    row_block,
                    simdgroups,
                    stage_depth,
                    prefetch,
                    splits,
                    k_slices,
                } => {
                    let partials = (splits > 1)
                        .then(|| command_buffer.allocate_scratch_for_shape(&[splits * m * n], DataType::U32))
                        .transpose()?;
                    let partials_range = partials.as_ref().map(VkBuffer::resolve_ref);
                    let phases = if splits == 1 {
                        0..1
                    } else {
                        1..3
                    };
                    let flags = [has_scale, has_bias, has_soft_cap, rht.is_some()];
                    macro_rules! gemm_int8 {
                    ($cache:expr, $kernel:ident, $phase:ident $(, $depth:expr, $prefetch:expr, $k_slices:expr)?) => {{
                        let kernel =
                            cached(&mut $cache, (flags, prologue, plan, $phase, b_layout), || {
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
                                    plan.int8_shared_words(),
                                    $phase,
                                    splits,
                                    b_layout,
                                    simdgroups,
                                    rht.is_some(),
                                    $($depth, $prefetch, $k_slices,)?
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
                                rht.clone(),
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
                            gemm_int8!(
                                self.gemm_int8_nv,
                                GemmInt8NvVulkanKernel,
                                phase,
                                stage_depth,
                                prefetch,
                                k_slices
                            );
                        } else if capabilities.contains(VkDeviceCapabilities::INT8_TENSOR_CORES_K32) {
                            gemm_int8!(self.gemm_int8, GemmInt8VulkanKernel, phase);
                        } else {
                            gemm_int8!(self.gemm_int8_k16, GemmInt8K16VulkanKernel, phase);
                        }
                    }
                },
                _ => unreachable!("the check gives A8 activations an A8 plan"),
            }
        } else {
            let a_range = a_range.expect("full-precision A outside A8");
            match plan {
                VkMatmulPlan::Gemv(tile) => {
                    let kernel = cached(
                        &mut self.gemv,
                        ([has_scale, has_bias, has_soft_cap, shape.gathered], prologue, tile, b_layout),
                        || {
                            GemvVulkanKernel::new(
                                &context,
                                input_type,
                                weights_type,
                                output_type,
                                has_scale,
                                has_bias,
                                has_soft_cap,
                                shape.gathered,
                                scheme,
                                mode,
                                signed_codes,
                                group_size,
                                tile.output_row_tile(),
                                tile.num_simdgroups,
                                tile.input_row_tile,
                                tile.group_lanes,
                                tile.input_aligned,
                                tile.full_tile,
                                tile.lane_layout,
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
                },
                VkMatmulPlan::GemmTensor(tiling) => {
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
                },
                VkMatmulPlan::Gemm(tiling) => {
                    let alignment = u32::from(m.is_multiple_of(tiling.block_m()))
                        | (u32::from(n.is_multiple_of(tiling.block_n())) * 2)
                        | (u32::from(k.is_multiple_of(tiling.block_k())) * 4);
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
                },
                _ => unreachable!("the check gives full-precision activations a full-precision plan"),
            }
        }

        if let Some(factors) = rht_factors.filter(|_| rht.is_none()) {
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
    VkKernelError::KernelPrecondition {
        kernel,
        condition,
    }
    .into()
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

fn wait<T: Send>(future: impl Future<Output = T> + Send) -> T {
    std::thread::scope(|scope| scope.spawn(|| block_on(future)).join().expect("a Vulkan wait panicked"))
}

#[cfg(test)]
#[path = "../../../../../unit/backends/vulkan/kernel/matmul/vk_matmul_kernel_test.rs"]
mod tests;
