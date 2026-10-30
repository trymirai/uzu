use std::{
    collections::{HashMap, hash_map::Entry},
    hash::Hash,
};

use futures::executor::block_on;
use half::f16;

use super::{GemmBPrologue, VkGemvTile, VkMatmulPlan};
use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, CpuBuffer, GlobalBufferCpuGpu,
            gpu_types::{
                gemm::{GemmBPrologueKind, GemmDTransform},
                hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
                quantization::QuantizationMode,
            },
            kernel::{
                ActivationQuantization, ActivationTransform,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{
                    ActivationFormat, Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel,
                    MatmulOutput, MatmulShape, TrellisFormat,
                },
            },
        },
        vulkan::{
            ActivationQuantizeVulkanKernel, Error, GemmInt8K16VulkanKernel, GemmInt8NvVulkanKernel,
            GemmInt8VulkanKernel, GemmPrepareBLayoutVulkanKernel, GemmTensorVulkanKernel, GemmVulkanKernel,
            GemvInt8VulkanKernel, GemvVulkanKernel, TrellisGemvVulkanKernel, VkBuffer, VkCommandBufferEncoding,
            VkContext, VkDeviceCapabilities, VkDeviceFacts, VkKernelError, Vulkan,
            gpu_types::{gemm::GemmTiling, gemm_b_layout::GemmBLayout},
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
    gemv_int8: HashMap<([bool; 5], GemmBPrologue, u32, u32, GemmBLayout), GemvInt8VulkanKernel>,
    /// INT8 GEMM pipelines by flags, prologue, plan, split phase and weight layout.
    gemm_int8: HashMap<([bool; 4], GemmBPrologue, VkMatmulPlan, u32, GemmBLayout), GemmInt8VulkanKernel>,
    gemm_int8_nv: HashMap<([bool; 4], GemmBPrologue, VkMatmulPlan, u32, GemmBLayout), GemmInt8NvVulkanKernel>,
    gemm_int8_k16: HashMap<([bool; 4], GemmBPrologue, VkMatmulPlan, u32, GemmBLayout), GemmInt8K16VulkanKernel>,
    trellis: HashMap<TrellisFormat, TrellisGemvVulkanKernel>,
    output_rht: HashMap<bool, ActivationTransform<Vulkan>>,
    /// Quantizers of full-precision A for GemvInt8, by code layout and group-sum size.
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
        if !data_types.iter().all(|data_type| matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32)) {
            return Err(VkKernelError::KernelVariant {
                kernel: "Matmul",
                data_types: Box::new(data_types),
            }
            .into());
        }
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
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

    /// A8 where the integer kernels take the shape.
    fn select_activation_quantization(
        &self,
        shape: &MatmulShape,
        context: &VkContext,
    ) -> Option<ActivationQuantization> {
        let capabilities = context.physical_device().capabilities;
        VkMatmulPlan::a8_quantization(shape, self.data_types(), capabilities)
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
        match VkMatmulPlan::activation_format(bf16_shape, weights_layout, self.data_types(), capabilities) {
            ActivationFormat::Int8(_) if matches!(prepared, ActivationFormat::Int8(_)) => prepared,
            _ => ActivationFormat::Bf16,
        }
    }

    /// Stores 8-bit codes signed where A8 applies, then packs codes into tiles for integer GEMM through one temporary
    /// copy; returns the layout and whether the codes are signed.
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
            // Device-local codes flip on the host through a copy.
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
            a8_shape.b_bits.filter(|_| VkMatmulPlan::tiles_weights(a8_shape, self.data_types(), capabilities))
        else {
            return Ok((GemmBLayout::RowMajor, prepared_shape.signed_codes));
        };
        let bytes = u64::from(n) * u64::from(k) * u64::from(bits) / 8;
        let rows = context.create_buffer_gpu(bytes as usize)?;
        let mut command_buffer = context.create_command_buffer(Some("prepare GEMM B layout"), None)?;
        command_buffer.encode_copy(weights, 0..bytes, &rows, 0)?;
        let kernel = GemmPrepareBLayoutVulkanKernel::new(&context.arc())?;
        // SAFETY: both ranges hold the n x k codes, which the copy above wrote to `rows`; the buffers do not alias.
        unsafe { kernel.encode((&rows, 0..bytes), (&*weights, 0..bytes), k, n, bits / 2, &mut command_buffer) };
        wait(command_buffer.end_encoding().submit().wait_until_completed())?;
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
        self.encode_plan(arguments, None, command_buffer)
    }
}

impl VkMatmulKernel {
    /// Whether the device's FP16 cooperative products keep FP32 precision: rows [128.125, -128.25, 0, ...] of A times
    /// rows [128.125, 128, 0, ...] of transposed B sum to exactly 1/64 from FP32 products, to 0 from FP16-rounded ones.
    pub fn fp16_products_exact(context: &VkContext) -> Result<bool, Error> {
        let rows = |row: [f32; 2], count: usize| {
            let values = row.into_iter().chain([0.0; 62]).map(f16::from_f32).cycle().take(64 * count);
            let values = values.collect::<Vec<_>>();
            let mut initializer = context.create_buffer_cpu_gpu(2 * values.len())?;
            initializer.copyin(&values);
            Ok::<_, Error>(initializer.handoff())
        };
        let (a, b) = (rows([128.125, -128.25], 16)?, rows([128.125, 128.0], 64)?);
        let mut d = context.create_buffer_gpu(2 * 16 * 64)?;
        let mut kernel = Self::new(context, DataType::F16, DataType::F16, DataType::F16)?;
        let mut command_buffer = context.create_command_buffer(Some("probe FP16 cooperative products"), None)?;
        let arguments = MatmulArguments {
            a: MatmulA::FullPrecision {
                values: &a,
                offset: 0,
            },
            b: MatmulB::FullPrecision {
                b: &b,
            },
            b_leading_dimension: None,
            b_transpose: true,
            output: MatmulOutput::new(&mut d, MatmulDOps::none()),
            gather_indices: None::<&<Vulkan as Backend>::GlobalGpuBuffer>,
            m: 16,
            n: 64,
            k: 64,
        };
        let plan = VkMatmulPlan::GemmTensor(GemmTiling::Tile16x64x64_Simdgroups1x4x2);
        kernel.encode_plan(arguments, Some(plan), &mut command_buffer)?;
        wait(command_buffer.end_encoding().submit().wait_until_completed())?;
        let output = wait(context.copyout::<f16>(&d))?;
        Ok(output.iter().all(|&value| value == f16::from_f32(1.0 / 64.0)))
    }

    /// Encodes the arguments by `plan`, or without one by the default, checked against them before any pipeline is
    /// created or command recorded. Trellis B has a single configuration and ignores the plan.
    fn encode_plan(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Vulkan,
            impl BufferRef<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
            impl BufferMut<Backend = Vulkan>,
            impl BufferRef<Backend = Vulkan>,
        >,
        plan: Option<VkMatmulPlan>,
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
            // SAFETY: the common contract makes callers pass Trellis A of [m, k] codes, [m, 4] column-group sums and m
            // scales, n checked code rows of code_row_bytes(k), n row scales, the 5-value codebook and BF16 D rows of
            // `row_stride`, D aliasing nothing.
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
        // The kernels' vector loads of A, A8 codes and B assume 16-byte aligned starts, as allocations have.
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
        let plan = match plan {
            Some(plan) => plan,
            None => self.baseline(&shape, b_layout, &context)?,
        };
        plan.check(&shape, b_layout, self.data_types(), VkDeviceFacts::new(physical_device))
            .map_err(|reason| precondition("Matmul", reason))?;
        // GemvInt8 of full-precision A first quantizes it to scratch A8 by the shape's A8 contract.
        let quantized_a = match (plan, &a_range) {
            (
                VkMatmulPlan::GemvInt8 {
                    ..
                },
                Some(values),
            ) => {
                let quantization = VkMatmulPlan::a8_quantization(&shape, self.data_types(), capabilities)
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
                    ActivationQuantizeVulkanKernel::new(&context, input_type, key.0, ACTIVATION_SCALE_GROUP_SIZE, key.1)
                })?;
                // SAFETY: A holds m rows of k values; the scratch holds their codes, scales and group sums.
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

        // SAFETY (every dispatch below): the common contract makes callers pass A of [m, k] elements from `offset` or
        // A8 codes of [m, k] with their [m, k / 128] scales and [m, k / group] group sums, B of n rows at the given
        // strides or n rows of k packed codes with their metadata planes at the given strides, D of [m, n], n bias
        // values and [m, n] gather indices below B's rows, D aliasing nothing; A8 split K's partials are fresh scratch
        // of their shape.
        if let Some((codes, a_scales, group_sums, scale_group_size, code_layout)) = int8 {
            // The plan check took the shape's A8 contract; the activations' own layout is checked here.
            if scale_group_size != ACTIVATION_SCALE_GROUP_SIZE
                || Some(code_layout) != Int8CodeLayout::for_right_bits(bits)
                || (scheme == GemmBPrologueKind::ScaleBiasDequant) != group_sums.is_some()
            {
                return Err(precondition(
                    "Matmul",
                    "A8 of 128-value activation scales, the weights' code layout and group sums exactly for ScaleBias",
                ));
            }
            let (scales, (scale_output, scale_group)) =
                (scales.expect("quantized B has scales"), scale_strides.expect("quantized B has scale strides"));
            if let VkMatmulPlan::GemvInt8 {
                rows,
                input_rows,
            } = plan
            {
                // The output transform, and the bias after it, finish in the epilogue where it fuses them.
                let rht = rht_factors.filter(|_| plan.fuses_output_rht()).map(VkBuffer::resolve_ref);
                let (has_bias, bias_range) = match rht {
                    Some(_) => (bias.is_some(), bias.map(VkBuffer::resolve_ref)),
                    None => (has_bias, bias_range),
                };
                let gemv_flags = [has_scale, has_bias, has_soft_cap, gathered, rht.is_some()];
                let kernel = cached(&mut self.gemv_int8, (gemv_flags, prologue, rows, input_rows, b_layout), || {
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
                        gathered,
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
                if rht.is_some() {
                    return Ok(());
                }
            } else {
                let VkMatmulPlan::GemmInt8 {
                    matrices,
                    row_block,
                    simdgroups,
                    stage_depth,
                    splits,
                } = plan
                else {
                    unreachable!("the check gives A8 activations an A8 plan");
                };
                // Split K's partial sums, the command buffer's scratch.
                let partials = (splits > 1)
                    .then(|| command_buffer.allocate_scratch_for_shape(&[splits * m * n], DataType::U32))
                    .transpose()?;
                let partials_range = partials.as_ref().map(VkBuffer::resolve_ref);
                // One whole-K phase, or split K's partial phase then its reduction, ordered by the `partials` barrier.
                let phases = if splits == 1 {
                    0..1
                } else {
                    1..3
                };
                // The output transform, and the bias after it, finish in the last phase's epilogue.
                let rht = rht_factors.map(VkBuffer::resolve_ref);
                let (has_bias, bias_range) = match rht {
                    Some(_) => (bias.is_some(), bias.map(VkBuffer::resolve_ref)),
                    None => (has_bias, bias_range),
                };
                let flags = [has_scale, has_bias, has_soft_cap, rht.is_some()];
                // The same arguments for every core and phase.
                macro_rules! gemm_int8 {
                    ($cache:expr, $kernel:ident, $phase:ident $(, $depth:expr)?) => {{
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
                                    $($depth,)?
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
                    if matrices.contains(VkDeviceCapabilities::INT8_ELEMENT_FOLD) {
                        gemm_int8!(self.gemm_int8_nv, GemmInt8NvVulkanKernel, phase, stage_depth);
                    } else if matrices == VkDeviceCapabilities::INT8_TENSOR_CORES_K16 {
                        gemm_int8!(self.gemm_int8_k16, GemmInt8K16VulkanKernel, phase);
                    } else {
                        gemm_int8!(self.gemm_int8, GemmInt8VulkanKernel, phase);
                    }
                }
                return Ok(());
            }
        } else {
            let a_range = a_range.expect("full-precision A outside A8");
            if let VkMatmulPlan::Gemv(tile) = plan {
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
            } else if let VkMatmulPlan::GemmTensor(tiling) = plan {
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
                let VkMatmulPlan::Gemm(tiling) = plan else {
                    unreachable!("the check gives full-precision activations a full-precision plan");
                };
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

    /// The default for `shape` from weights in `b_layout`, whose FP16 tensor cores count only where the context
    /// qualified them.
    fn baseline(
        &self,
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        context: &VkContext,
    ) -> Result<VkMatmulPlan, Error> {
        let device = VkDeviceFacts::new(context.physical_device());
        let eligible = context.qualified_float_tensor_cores(self.input_data_type)?;
        Ok(VkMatmulPlan::baseline(shape, b_layout, self.data_types(), device, eligible))
    }

    fn data_types(&self) -> [DataType; 3] {
        [self.weights_data_type, self.input_data_type, self.output_data_type]
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

/// `future`'s output, waited for on a thread of its own: weights load inside the caller's executor, which a nested
/// `block_on` on the same thread would reject.
fn wait<T: Send>(future: impl Future<Output = T> + Send) -> T {
    std::thread::scope(|scope| scope.spawn(|| block_on(future)).join().expect("a Vulkan wait panicked"))
}
