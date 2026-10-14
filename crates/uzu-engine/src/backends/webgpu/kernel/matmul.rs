use std::collections::BTreeMap;

use super::{
    super::{
        WebGpu, WebGpuError,
        command_buffer::{Binding, WebGpuCommandBufferEncoding},
        context::WebGpuContext,
    },
    shader::{NamedBinding, Param, Shader},
};
use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, Kernels,
        kernel::{
            ActivationTransform, TensorAddBiasKernel, Unsupported,
            matmul::{MatmulA, MatmulArguments, MatmulB, MatmulKernel, QuantizedCorrection},
            radix_top_k_small::RadixTopKSmall,
        },
    },
    data_type::DataType,
};

pub struct WebGpuMatmul {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    subgroup_gemv: bool,
    subgroup32_gemv: bool,
    output_rht: ActivationTransform<WebGpu>,
    bias_add: <<WebGpu as Backend>::Kernels as Kernels>::TensorAddBiasKernel,
}

impl MatmulKernel for WebGpuMatmul {
    type Backend = WebGpu;

    fn new(
        context: &WebGpuContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, WebGpuError> {
        for dtype in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(dtype, DataType::F32 | DataType::F16 | DataType::BF16) {
                return Err(WebGpuError::Unsupported(format!("matmul data type {dtype:?}")));
            }
        }
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            subgroup_gemv: context.supports_subgroups() && context.subgroup_id_supported(),
            subgroup32_gemv: context.supports_subgroups()
                && context.subgroup_id_supported()
                && context.subgroup_size_range() == Some((32, 32)),
            output_rht: ActivationTransform::output_rht(context, output_data_type, None, true)?,
            bias_add: <<WebGpu as Backend>::Kernels as Kernels>::TensorAddBiasKernel::new(
                context,
                output_data_type,
                weights_data_type,
                true,
            )?,
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            WebGpu,
            impl BufferRef<Backend = WebGpu>,
            impl BufferRef<Backend = WebGpu>,
            impl BufferMut<Backend = WebGpu>,
            impl BufferRef<Backend = WebGpu>,
        >,
        command: &mut WebGpuCommandBufferEncoding,
    ) -> Result<(), WebGpuError> {
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
        let mut destination = output.values;
        let ops = output.ops;
        let row_stride = output.row_stride.unwrap_or(n);
        if row_stride < n {
            return Err(WebGpuError::Unsupported("matmul output stride is smaller than its width".into()));
        }
        if ops.rht_factors.is_some() && row_stride != n {
            return Err(WebGpuError::Unsupported("strided matmul output with Hadamard transform".into()));
        }
        let mut bindings = Vec::new();
        let mut specs = BTreeMap::new();
        let mut params = vec![
            ("m", Param::u32(m)),
            ("n", Param::u32(n)),
            ("k", Param::u32(k)),
            ("row_stride", Param::u32(row_stride)),
            ("output_scale", Param::f32(ops.ab_scale)),
        ];
        specs.insert("a_grouped", Param::bool(false));
        specs.insert("a_int8", Param::bool(false));
        specs.insert("b_quantized", Param::bool(false));
        specs.insert("bits", Param::u32(4));
        params.push(("a_group_size", Param::u32(1)));
        params.push(("group_size", Param::u32(1)));
        let mut input_bf16_words = false;
        let mut full_precision_input = false;
        match a {
            MatmulA::FullPrecision {
                values,
                offset,
            } => {
                let binding = Binding::read(values.subrange(offset * self.input_data_type.size_in_bytes()..));
                input_bf16_words = self.input_data_type == DataType::BF16 && binding.offset.is_multiple_of(4);
                full_precision_input = true;
                bindings.push(NamedBinding::new("a", binding, self.input_data_type));
            },
            MatmulA::Int8Symmetric {
                values,
                scales,
                scale_group_size,
                code_layout,
                ..
            } => {
                if scale_group_size == 0 || !k.is_multiple_of(scale_group_size) {
                    return Err(WebGpuError::Unsupported("unaligned int8 activation groups".into()));
                }
                if code_layout.is_grouped_by_nibble() && !k.is_multiple_of(8) {
                    return Err(WebGpuError::Unsupported(
                        "grouped int8 activation rows must be aligned to eight codes".into(),
                    ));
                }
                bindings.push(NamedBinding::new("a", Binding::read(values), DataType::I8));
                bindings.push(NamedBinding::new("a_scales", Binding::read(scales), DataType::F32));
                specs.insert("a_int8", Param::bool(true));
                specs.insert("a_grouped", Param::bool(code_layout.is_grouped_by_nibble()));
                params.push(("a_group_size", Param::u32(scale_group_size)));
            },
            MatmulA::Trellis {
                ..
            } => return Err(WebGpuError::Unsupported("Trellis activation format".into())),
        }
        let mut quantized_packing = None;
        let mut w4_group_size = None;
        match b {
            MatmulB::FullPrecision {
                b,
            } => {
                bindings.push(NamedBinding::new("b", Binding::read(b), self.weights_data_type));
                specs.insert("transpose", Param::bool(b_transpose));
                params.push((
                    "b_stride",
                    Param::u32(b_leading_dimension.unwrap_or(if b_transpose {
                        k
                    } else {
                        n
                    })),
                ));
            },
            MatmulB::Quantized(q) => {
                if !b_transpose || q.group_size == 0 || !k.is_multiple_of(q.group_size) {
                    return Err(WebGpuError::Unsupported(
                        "quantized matmul requires transposed, group-aligned weights".into(),
                    ));
                }
                let bits = q.bits();
                if !matches!(bits, 4 | 8) {
                    return Err(WebGpuError::Unsupported(format!("{bits}-bit quantized matmul")));
                }
                if q.group_size.is_multiple_of(32 / bits) {
                    quantized_packing = Some(32 / bits);
                }
                if bits == 4 && matches!(q.group_size, 32 | 64) {
                    w4_group_size = Some(q.group_size);
                }
                let strides = q.params.scale_strides();
                let zp_strides = q.zero_point_strides();
                bindings.push(NamedBinding::new("codes", Binding::read(q.codes), DataType::U32));
                bindings.push(NamedBinding::new("scales", Binding::read(q.scales), self.weights_data_type));
                specs.insert("b_quantized", Param::bool(true));
                specs.insert("bits", Param::u32(bits));
                specs.insert("signed_codes", Param::bool(q.signed_codes));
                params.extend([
                    ("group_size", Param::u32(q.group_size)),
                    ("scale_output_stride", Param::u32(strides.output_stride)),
                    ("scale_group_stride", Param::u32(strides.group_stride)),
                    ("zp_output_stride", Param::u32(zp_strides.output_stride)),
                    ("zp_group_stride", Param::u32(zp_strides.group_stride)),
                ]);
                let correction = match q.correction {
                    QuantizedCorrection::Symmetric => 0,
                    QuantizedCorrection::Biases(biases) => {
                        bindings.push(NamedBinding::new("quant_biases", Binding::read(biases), self.weights_data_type));
                        1
                    },
                    QuantizedCorrection::ZeroPoints(points) => {
                        bindings.push(NamedBinding::new("zero_points", Binding::read(points), DataType::U32));
                        2
                    },
                };
                specs.insert("correction", Param::u32(correction));
            },
            MatmulB::Trellis {
                ..
            } => return Err(WebGpuError::Unsupported("Trellis weights".into())),
        }
        let has_gather = gather_indices.is_some();
        if let Some(gather) = gather_indices {
            bindings.push(NamedBinding::new("gather", Binding::read(gather), DataType::U32));
        }
        let use_w4 = self.subgroup32_gemv && full_precision_input && w4_group_size.is_some();
        // Row reuse amortizes large projections even for short prompts. Smaller
        // projections retain the fused row kernel to avoid extra RHT dispatches.
        let row_parallel_w4 = use_w4 && (m <= 8 || (m <= 32 && (k < 2048 || n < 2048)) || has_gather);
        let fused_rht = row_parallel_w4 && ops.rht_factors.is_some() && n.is_multiple_of(32);
        if let Some(factors) = ops.rht_factors.filter(|_| fused_rht) {
            bindings.push(NamedBinding::new("rht_factors", Binding::read(factors), DataType::I32));
        }
        if let Some(bias) = ops.bias.filter(|_| ops.rht_factors.is_none() || fused_rht) {
            bindings.push(NamedBinding::new("bias", Binding::read(bias), self.weights_data_type));
        }
        bindings.push(NamedBinding::new("d", Binding::write(destination.reborrow()), self.output_data_type));
        specs.insert("gather", Param::bool(has_gather));
        specs.insert("has_bias", Param::bool(ops.bias.is_some() && (ops.rht_factors.is_none() || fused_rht)));
        specs.insert("has_soft_cap", Param::bool(ops.soft_cap.is_some()));
        params.push(("soft_cap", Param::f32(ops.soft_cap.unwrap_or(1.0))));
        if use_w4 {
            specs.insert("w4_group_size", Param::u32(w4_group_size.unwrap()));
            specs.insert("input_bf16_words", Param::bool(input_bf16_words));
        }
        let mut shader = Shader::new(bindings, specs, params);
        if row_parallel_w4 {
            // Small projections need more independent waves; wide projections
            // benefit from reusing each input slice across four output columns.
            let width = if fused_rht {
                32
            } else if n <= 8192 {
                2
            } else {
                4
            };
            let columns_per_group = if fused_rht {
                32
            } else {
                2 * width
            };
            let groups = n.div_ceil(columns_per_group);
            shader.groups = [groups.min(65535), m, groups.div_ceil(65535)];
            shader.kernel("matmul_w4", width);
        } else if use_w4 {
            // Keep the ordinary grid unchanged. Larger axes occupy pages in Z;
            // the shader discards padding in each page before indexing tensors.
            const AXIS_LIMIT: u32 = 65535;
            let columns = n.div_ceil(16);
            let rows = m.div_ceil(4);
            let column_pages = columns.div_ceil(AXIS_LIMIT);
            let row_pages = rows.div_ceil(AXIS_LIMIT);
            let slices = column_pages
                .checked_mul(row_pages)
                .filter(|&slices| slices <= AXIS_LIMIT)
                .ok_or_else(|| WebGpuError::Unsupported("matmul prefill grid exceeds 3D dispatch limits".into()))?;
            shader.specs.insert(
                "prefill_column_pages",
                Param::u32(if slices > 1 {
                    column_pages
                } else {
                    0
                }),
            );
            shader.groups = [columns.min(AXIS_LIMIT), rows.min(AXIS_LIMIT), slices];
            shader.kernel("matmul_w4_prefill", 0);
        } else if self.subgroup_gemv && quantized_packing.is_some() && (m <= 4 || has_gather) {
            let groups = n.div_ceil(4);
            shader.groups = [groups.min(65535), m, groups.div_ceil(65535)];
            shader.kernel("matmul_subgroup", 0);
        } else if m <= 4 || has_gather {
            shader.groups = [n.min(65535), m, n.div_ceil(65535)];
            shader.kernel("matmul_decode", 0);
        } else {
            shader.groups = [n.div_ceil(16), m.div_ceil(16), 1];
            shader.kernel("matmul_prefill", 0);
        }
        shader.dispatch(command);
        if let Some(factors) = ops.rht_factors.filter(|_| !fused_rht) {
            self.output_rht.encode_fp_in_place(destination.reborrow(), factors, None, m, n, command);
            if let Some(bias) = ops.bias {
                self.bias_add.encode(None::<&<WebGpu as Backend>::ScratchBuffer>, bias, destination, n, m * n, command);
            }
        }
        Ok(())
    }
}

impl RadixTopKSmall for Unsupported<WebGpu> {
    type Backend = WebGpu;
    fn new(
        _context: &WebGpuContext,
        _columns: u32,
    ) -> Result<Self, WebGpuError> {
        Err(WebGpuError::Unsupported("radix top-k for speculative decoding".into()))
    }
    fn encode(
        &self,
        _input: impl BufferRef<Backend = WebGpu>,
        _output_ids: impl BufferMut<Backend = WebGpu>,
        _output_scores: impl BufferMut<Backend = WebGpu>,
        _rows: u32,
        _k: u32,
        _command: &mut WebGpuCommandBufferEncoding,
    ) -> Result<(), WebGpuError> {
        Err(WebGpuError::Unsupported("radix top-k for speculative decoding".into()))
    }
}
