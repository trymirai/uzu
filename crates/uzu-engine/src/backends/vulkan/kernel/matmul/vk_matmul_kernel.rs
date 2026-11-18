use std::{
    collections::{HashMap, hash_map::Entry},
    hash::Hash,
    ops::Range,
    sync::Arc,
};

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
                matmul::{ActivationFormat, MatmulA, MatmulArguments, MatmulB, MatmulKernel, MatmulShape, QuantizedB},
            },
        },
        vulkan::{
            ActivationQuantizeVulkanKernel, Error, GemmInt8K16VulkanKernel, GemmInt8NvVulkanKernel,
            GemmInt8VulkanKernel, GemmPrepareBLayoutVulkanKernel, GemmTensorVulkanKernel, GemmVulkanKernel,
            GemvInt8VulkanKernel, GemvVulkanKernel, VkBuffer, VkCommandBufferEncoding, VkContext, VkDeviceCapabilities,
            VkKernelError, Vulkan, gpu_types::gemm_b_layout::GemmBLayout, kernel::matmul::VkMatmulPlan,
        },
    },
    data_type::DataType,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Specialization {
    plan: VkMatmulPlan,
    has_scale: bool,
    has_bias: bool,
    has_soft_cap: bool,
    has_rht: bool,
    gathered: bool,
    b_prologue: GemmBPrologueKind,
    quantization_mode: QuantizationMode,
    signed_codes: bool,
    group_size: u32,
    b_layout: GemmBLayout,
}

pub struct VkMatmulKernel {
    data_types: [DataType; 3],
    gemv: HashMap<Specialization, GemvVulkanKernel>,
    gemm: HashMap<(Specialization, u32), GemmVulkanKernel>,
    gemm_tensor: HashMap<Specialization, GemmTensorVulkanKernel>,
    gemv_int8: HashMap<Specialization, GemvInt8VulkanKernel>,
    gemm_int8: HashMap<(Specialization, u32), GemmInt8VulkanKernel>,
    gemm_int8_nv: HashMap<(Specialization, u32), GemmInt8NvVulkanKernel>,
    gemm_int8_k16: HashMap<(Specialization, u32), GemmInt8K16VulkanKernel>,
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
            output_rht: HashMap::new(),
            quantize_a: HashMap::new(),
        })
    }

    fn select_activation_quantization(
        &self,
        shape: &MatmulShape,
        context: &VkContext,
    ) -> Option<ActivationQuantization> {
        let capabilities = context.device().physical_device.capabilities;
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
        _weights_layout: GemmBLayout,
        prepared: ActivationFormat,
        context: &VkContext,
    ) -> ActivationFormat {
        let capabilities = context.device().physical_device.capabilities;
        match prepared {
            ActivationFormat::Int8(_) if VkMatmulPlan::a8_rows(bf16_shape.m, capabilities) => prepared,
            _ => ActivationFormat::Bf16,
        }
    }

    fn optimize_weights_layout(
        &self,
        a8_shape: &MatmulShape,
        weights: &mut <Vulkan as Backend>::GlobalGpuBuffer,
        context: &VkContext,
    ) -> Result<(GemmBLayout, bool), Error> {
        let capabilities = context.device().physical_device.capabilities;
        if VkMatmulPlan::a8_quantization(a8_shape, self.data_types, capabilities).is_none() {
            return Ok((GemmBLayout::RowMajor, a8_shape.signed_codes));
        }
        let signed_codes = a8_shape.signed_codes || a8_shape.b_bits == Some(8);
        if !a8_shape.signed_codes && signed_codes {
            let mut codes = VkContext::wait(context.copyout::<u8>(&*weights))?;
            let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(&mut codes);
            words.iter_mut().for_each(|word| *word ^= 0x8080_8080_8080_8080);
            prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= 0x80);
            let mut signed = context.create_buffer_cpu_gpu(codes.len())?;
            signed.copyin(&codes);
            *weights = signed.handoff()?;
        }
        let (n, k) = (a8_shape.n, a8_shape.k);
        let Some(bits) = a8_shape.b_bits.filter(|_| VkMatmulPlan::tiles_weights(n, capabilities)) else {
            return Ok((GemmBLayout::RowMajor, signed_codes));
        };
        let bytes = u64::from(n) * u64::from(k) * u64::from(bits) / 8;
        let tiles = context.create_buffer_gpu(bytes as usize)?;
        let mut command_buffer = context.create_command_buffer(Some("prepare GEMM B layout"), None)?;
        let kernel = GemmPrepareBLayoutVulkanKernel::new(context)?;
        kernel.encode((&*weights, 0..bytes), (&tiles, 0..bytes), k, n, bits / 2, &mut command_buffer);
        drop(command_buffer.end_encoding().try_submit()?);
        *weights = tiles;
        Ok((GemmBLayout::Tiled, signed_codes))
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
            output,
            gather,
            m,
            n,
            k,
            ..
        } = arguments;
        let context = command_buffer.context().arc();
        command_buffer.declare_work(2 * u64::from(m) * u64::from(n) * u64::from(k));
        let [weights_type, input_type, output_type] = self.data_types;
        let physical_device = &context.device().physical_device;
        let capabilities = physical_device.capabilities;
        let d_row_stride = output.checked_row_stride(n)?;
        let ops = output.ops;
        if ops.rht_factors.is_some() && !n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            return Err(precondition("n % 32 == 0 under output RHT"));
        }
        let (a_values, a8) = match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => (Some(VkBuffer::resolve_ref(values.subrange(offset..))), None),
            MatmulA::Int8Symmetric {
                values,
                scales,
                group_sums,
                ..
            } => {
                let a8 = (VkBuffer::resolve_ref(values), VkBuffer::resolve_ref(scales));
                (None, Some((a8.0, a8.1, group_sums.map(VkBuffer::resolve_ref))))
            },
            MatmulA::PackedHalf {
                ..
            }
            | MatmulA::Trellis {
                ..
            } => return Err(precondition("full-precision or A8 A with full-precision or quantized B")),
        };
        let b_layout = b.weights_layout();
        let (b_values, quantized) = match b {
            MatmulB::FullPrecision {
                b,
            } => (b, None),
            MatmulB::Quantized(quantized) => (quantized.codes, Some(quantized)),
            MatmulB::Trellis {
                ..
            } => return Err(precondition("full-precision or quantized B")),
        };
        let quantization_mode = match quantized.as_ref().map_or(QuantizationMode::U8, |quantized| quantized.mode) {
            QuantizationMode::I8 if shape.signed_codes => QuantizationMode::U8,
            QuantizationMode::I8 => return Err(precondition("I8 codes marked as signed")),
            mode => mode,
        };
        let b_range = VkBuffer::resolve_ref(b_values);
        let aligned =
            |(buffer, range): &(&Arc<VkBuffer>, Range<u64>)| (buffer.device_address() + range.start).is_multiple_of(16);
        if !aligned(&b_range)
            || !a_values.as_ref().is_none_or(aligned)
            || !a8.as_ref().is_none_or(|(codes, ..)| aligned(codes))
        {
            return Err(precondition("A, A8 codes and B ranges starting 16-byte aligned"));
        }
        let plan = VkMatmulPlan::baseline(&shape, b_layout, self.data_types, physical_device);
        plan.check(&shape, b_layout, self.data_types, physical_device).map_err(precondition)?;

        let scales = quantized.as_ref().map(|quantized| VkBuffer::resolve_ref(quantized.scales));
        let biases = quantized.as_ref().and_then(QuantizedB::biases).map(VkBuffer::resolve_ref);
        let zero_points = quantized.as_ref().and_then(QuantizedB::zero_points).map(VkBuffer::resolve_ref);
        let (scale_output, scale_group) = quantized
            .as_ref()
            .map(|quantized| quantized.params.scale_strides())
            .map(|strides| (strides.output_stride, strides.group_stride))
            .unzip();
        let (zero_point_output, zero_point_group) = quantized
            .as_ref()
            .filter(|quantized| quantized.zero_points().is_some())
            .map(QuantizedB::zero_point_strides)
            .map(|strides| (strides.output_stride, strides.group_stride))
            .unzip();
        let rht = ops.rht_factors.filter(|_| plan.fuses_output_rht());
        // An output RHT the kernel does not fuse runs afterwards and adds the bias itself.
        let bias = ops.bias.filter(|_| ops.rht_factors.is_none() || rht.is_some());
        let ab_scale = shape.d_transform.contains(GemmDTransform::SCALE).then_some(ops.ab_scale);
        let soft_cap = ops.soft_cap;
        let key = Specialization {
            plan,
            has_scale: ab_scale.is_some(),
            has_bias: bias.is_some(),
            has_soft_cap: soft_cap.is_some(),
            has_rht: rht.is_some(),
            gathered: shape.gathered,
            b_prologue: shape.b_prologue,
            quantization_mode,
            signed_codes: shape.signed_codes,
            group_size: shape.b_group_size.unwrap_or(0),
            b_layout,
        };
        let d = output.values;
        let d_range = VkBuffer::resolve_ref(d.as_ref());
        let bias_range = bias.map(VkBuffer::resolve_ref);
        let rht_range = rht.map(VkBuffer::resolve_ref);
        let gather_window = gather.as_ref().map(|gather| (gather.first_row, gather.row_count));
        let gather = gather.map(|gather| VkBuffer::resolve_ref(gather.indices));

        match plan {
            VkMatmulPlan::Gemv(tile) => {
                let kernel = cached(&mut self.gemv, key, || {
                    GemvVulkanKernel::new(
                        &context,
                        input_type,
                        weights_type,
                        output_type,
                        key.has_scale,
                        key.has_bias,
                        key.has_soft_cap,
                        key.gathered,
                        key.b_prologue,
                        key.quantization_mode,
                        key.signed_codes,
                        key.group_size,
                        tile.output_row_tile(),
                        tile.num_simdgroups,
                        tile.input_row_tile,
                        tile.group_lanes,
                        tile.input_aligned,
                        tile.full_tile,
                        tile.lane_layout,
                        key.b_layout,
                    )
                })?;
                kernel.encode(
                    b_range,
                    scales,
                    biases,
                    zero_points,
                    a_values.expect("the baseline plans full-precision kernels for full-precision A"),
                    d_range,
                    bias_range,
                    gather,
                    k,
                    n,
                    m,
                    d_row_stride,
                    scale_output,
                    scale_group,
                    zero_point_output,
                    zero_point_group,
                    ab_scale,
                    soft_cap,
                    gather_window.map(|(first_row, _)| first_row),
                    gather_window.map(|(_, row_count)| row_count),
                    command_buffer,
                );
            },
            VkMatmulPlan::GemmTensor(tiling) => {
                let kernel = cached(&mut self.gemm_tensor, key, || {
                    GemmTensorVulkanKernel::new(
                        &context,
                        weights_type,
                        output_type,
                        key.has_scale,
                        key.has_bias,
                        key.has_soft_cap,
                        key.b_prologue,
                        key.quantization_mode,
                        key.signed_codes,
                        key.group_size,
                        tiling,
                        tiling.simdgroups(),
                        tiling.tensor_staging(),
                    )
                })?;
                kernel.encode(
                    b_range,
                    scales,
                    biases,
                    zero_points,
                    a_values.expect("the baseline plans full-precision kernels for full-precision A"),
                    d_range,
                    bias_range,
                    k,
                    n,
                    m,
                    d_row_stride,
                    scale_output,
                    scale_group,
                    zero_point_output,
                    zero_point_group,
                    ab_scale,
                    soft_cap,
                    command_buffer,
                );
            },
            VkMatmulPlan::Gemm(tiling) => {
                let alignment = u32::from(m.is_multiple_of(tiling.block_m()))
                    | (u32::from(n.is_multiple_of(tiling.block_n())) * 2)
                    | (u32::from(k.is_multiple_of(tiling.block_k())) * 4);
                let kernel = cached(&mut self.gemm, (key, alignment), || {
                    GemmVulkanKernel::new(
                        &context,
                        input_type,
                        weights_type,
                        output_type,
                        key.has_scale,
                        key.has_bias,
                        key.has_soft_cap,
                        key.b_prologue,
                        key.quantization_mode,
                        key.signed_codes,
                        key.group_size,
                        tiling,
                        alignment,
                        tiling.simdgroups(),
                        tiling.scalar_staging(true, input_type.size_in_bytes() as u32),
                        tiling.scalar_staging(false, weights_type.size_in_bytes() as u32),
                    )
                })?;
                kernel.encode(
                    b_range,
                    scales,
                    biases,
                    zero_points,
                    a_values.expect("the baseline plans full-precision kernels for full-precision A"),
                    d_range,
                    bias_range,
                    k,
                    n,
                    m,
                    d_row_stride,
                    scale_output,
                    scale_group,
                    zero_point_output,
                    zero_point_group,
                    ab_scale,
                    soft_cap,
                    command_buffer,
                );
            },
            VkMatmulPlan::GemvInt8 {
                rows,
                input_rows,
            } => {
                let (Some(scales), Some(scale_output), Some(scale_group)) = (scales, scale_output, scale_group) else {
                    unreachable!("the check gives A8 plans quantized B");
                };
                let scratch;
                let (codes, a_scales, group_sums) = match a8 {
                    Some(a8) => a8,
                    None => {
                        let quantization = VkMatmulPlan::a8_quantization(&shape, self.data_types, capabilities)
                            .expect("the check gives A8 plans an A8 quantization");
                        let sums = quantization.sum_group_size();
                        scratch = (
                            command_buffer.allocate_scratch_for_shape(&[m * k], DataType::I8)?,
                            command_buffer
                                .allocate_scratch_for_shape(&[m * k / ACTIVATION_SCALE_GROUP_SIZE], DataType::F32)?,
                            sums.map(|group| {
                                command_buffer.allocate_scratch_for_shape(&[m * k / group], DataType::I32)
                            })
                            .transpose()?,
                        );
                        let a8 = (
                            VkBuffer::resolve_ref(&scratch.0),
                            VkBuffer::resolve_ref(&scratch.1),
                            scratch.2.as_ref().map(VkBuffer::resolve_ref),
                        );
                        let quantize_key = (quantization.code_layout().is_grouped_by_nibble(), sums.unwrap_or(0));
                        let kernel = cached(&mut self.quantize_a, quantize_key, || {
                            ActivationQuantizeVulkanKernel::new(&context, input_type, quantize_key.0, quantize_key.1)
                        })?;
                        kernel.encode(
                            a_values.expect("A without A8 is full-precision"),
                            a8.0.clone(),
                            a8.1.clone(),
                            a8.2.clone(),
                            m,
                            k,
                            command_buffer,
                        );
                        a8
                    },
                };
                let kernel = cached(&mut self.gemv_int8, key, || {
                    GemvInt8VulkanKernel::new(
                        &context,
                        weights_type,
                        output_type,
                        key.has_scale,
                        key.has_bias,
                        key.has_soft_cap,
                        key.b_prologue,
                        key.quantization_mode,
                        key.signed_codes,
                        key.group_size,
                        rows,
                        input_rows,
                        key.b_layout,
                        capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT),
                        key.gathered,
                        key.has_rht,
                    )
                })?;
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
                    rht_range,
                    k,
                    n,
                    m,
                    d_row_stride,
                    scale_output,
                    scale_group,
                    zero_point_output,
                    zero_point_group,
                    ab_scale,
                    soft_cap,
                    gather_window.map(|(first_row, _)| first_row),
                    gather_window.map(|(_, row_count)| row_count),
                    command_buffer,
                );
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
                let (Some(scales), Some(scale_output), Some(scale_group)) = (scales, scale_output, scale_group) else {
                    unreachable!("the check gives A8 plans quantized B");
                };
                let (codes, a_scales, group_sums) = a8.expect("the baseline plans GemmInt8 for A8 activations only");
                let partials = (splits > 1)
                    .then(|| command_buffer.allocate_scratch_for_shape(&[splits * m * n], DataType::U32))
                    .transpose()?;
                let partials_range = partials.as_ref().map(VkBuffer::resolve_ref);
                let phases = if splits == 1 {
                    0..1
                } else {
                    1..3
                };
                macro_rules! gemm_int8 {
                    ($cache:expr, $kernel:ident, $phase:ident $(, $depth:expr, $prefetch:expr, $k_slices:expr)?) => {{
                        let kernel = cached(&mut $cache, (key, $phase), || {
                            $kernel::new(
                                &context,
                                weights_type,
                                output_type,
                                key.has_scale,
                                key.has_bias,
                                key.has_soft_cap,
                                key.b_prologue,
                                key.quantization_mode,
                                key.signed_codes,
                                key.group_size,
                                row_block,
                                plan.int8_shared_words(),
                                $phase,
                                splits,
                                key.b_layout,
                                simdgroups,
                                key.has_rht,
                                $($depth, $prefetch, $k_slices,)?
                            )
                        })?;
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
                            rht_range.clone(),
                            k,
                            n,
                            m,
                            d_row_stride,
                            scale_output,
                            scale_group,
                            zero_point_output,
                            zero_point_group,
                            ab_scale,
                            soft_cap,
                            command_buffer,
                        )
                    }};
                }
                for phase in phases {
                    if element_fold {
                        gemm_int8!(self.gemm_int8_nv, GemmInt8NvVulkanKernel, phase, stage_depth, prefetch, k_slices);
                    } else if capabilities.contains(VkDeviceCapabilities::INT8_TENSOR_CORES_K32) {
                        gemm_int8!(self.gemm_int8, GemmInt8VulkanKernel, phase);
                    } else {
                        gemm_int8!(self.gemm_int8_k16, GemmInt8K16VulkanKernel, phase);
                    }
                }
            },
        }

        if let Some(factors) = ops.rht_factors.filter(|_| rht.is_none()) {
            let transform = cached(&mut self.output_rht, ops.bias.is_some(), || {
                ActivationTransform::output_rht(
                    context.as_ref(),
                    output_type,
                    ops.bias.map(|_| weights_type),
                    HADAMARD_TRANSFORM_BLOCK_SIZE,
                    true,
                )
            })?;
            transform.encode_fp_in_place(d, Some(factors), ops.bias, m, n, command_buffer);
        }
        Ok(())
    }
}

fn precondition(condition: &'static str) -> Error {
    VkKernelError::KernelPrecondition {
        kernel: "Matmul",
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
