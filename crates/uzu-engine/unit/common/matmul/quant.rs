use std::mem::{size_of, size_of_val};

use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::{quantization::QuantizationMode, quantization_method::QuantizationMethod},
            kernel::{
                ActivationQuantization, ActivationTransform, Kernels,
                matmul::{
                    ActivationFormat, Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel,
                    MatmulOutput, MatmulShape, QuantParams, QuantParamsLayout, QuantizedB, QuantizedCorrection,
                    WeightsLayout,
                },
            },
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
};

#[derive(Clone)]
pub struct PreparedInt8A {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub group_sums: Vec<i32>,
    pub quantization: ActivationQuantization,
}

#[derive(Clone)]
pub struct QuantInput<T: DataTypeRepr + Float> {
    pub w_packed: Vec<u32>,
    pub scales: Vec<T>,
    pub zero_points: Option<Vec<u8>>,
    pub biases: Option<Vec<T>>,
    pub x: Vec<T>,
    pub ab_scale: f32,
    pub soft_cap: Option<f32>,
    pub k: u32,
    pub n: u32,
    pub m: u32,
    pub group_size: u32,
    pub quant_method: QuantizationMethod,
    pub mode: QuantizationMode,
    pub params_layout: QuantParamsLayout,
    pub signed_codes: bool,
    pub prepared_a: Option<PreparedInt8A>,
    /// Output random Hadamard factors, then an output bias after them, as RHT-wrapped linears finish.
    pub output_rht_factors: Option<Vec<i32>>,
    pub output_bias: Option<Vec<T>>,
}

fn mode_for_bits(bits: u32) -> QuantizationMode {
    match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::U8,
        _ => unreachable!("unsupported bits: {bits}"),
    }
}

fn pad<T: DataTypeRepr>(
    values: &[T],
    minimum_len: usize,
) -> Vec<T> {
    let mut padded = values.to_vec();
    padded.resize(values.len().max(minimum_len), T::zeroed());
    padded
}

pub fn transpose_metadata(
    plane: &mut [u8],
    columns: u32,
    groups: u32,
    bits: u32,
) {
    let data_type = match bits {
        4 => DataType::U4,
        8 => DataType::U8,
        16 => DataType::BF16,
        32 => DataType::F32,
        _ => unreachable!("unsupported metadata width: {bits}"),
    };
    let source_params = QuantParams::new(QuantParamsLayout::OutputGroup, columns, groups);
    let destination_params = QuantParams::new(QuantParamsLayout::GroupOutput, columns, groups);
    let plane_layout = |params: QuantParams| match data_type {
        DataType::U4 => {
            (params.zero_point_shape(QuantizationMode::U4), params.zero_point_strides(QuantizationMode::U4))
        },
        DataType::U8 => {
            (params.zero_point_shape(QuantizationMode::U8), params.zero_point_strides(QuantizationMode::U8))
        },
        _ => (params.scale_shape(), params.scale_strides()),
    };
    let (source_shape, source_strides) = plane_layout(source_params);
    let (_, destination_strides) = plane_layout(destination_params);
    let source_len = source_shape.into_iter().product::<u32>() as usize
        * if data_type == DataType::U4 {
            DataType::U8
        } else {
            data_type
        }
        .size_in_bytes();
    let source = plane[..source_len].to_vec();
    plane.fill(0);
    for output in 0..columns {
        for group in 0..groups {
            let source_index = (output * source_strides.output_stride + group * source_strides.group_stride) as usize;
            let destination_index =
                (output * destination_strides.output_stride + group * destination_strides.group_stride) as usize;
            if bits == 4 {
                let value = (source[source_index / 2] >> (source_index % 2 * 4)) & 0x0F;
                plane[destination_index / 2] |= value << (destination_index % 2 * 4);
            } else {
                let width = bits as usize / u8::BITS as usize;
                let source_offset = source_index * width;
                let destination_offset = destination_index * width;
                plane[destination_offset..destination_offset + width]
                    .copy_from_slice(&source[source_offset..source_offset + width]);
            }
        }
    }
}

impl<T: DataTypeRepr + Float> QuantInput<T> {
    pub fn new(
        m: u32,
        k: u32,
        n: u32,
        group_size: u32,
        bits: u32,
        quant_method: QuantizationMethod,
        seed: u64,
    ) -> Self {
        let num_groups_k = k.div_ceil(group_size);
        let mut rng = SmallRng::seed_from_u64(seed);

        let w_packed: Vec<u32> =
            (0..(n as usize * k as usize * bits as usize / 32)).map(|_| rng.random_range(0..u32::MAX)).collect();
        let scales: Vec<T> =
            (0..(n * num_groups_k) as usize).map(|_| T::from(rng.random_range(0.01f32..0.3f32)).unwrap()).collect();
        let x: Vec<T> = (0..(m * k) as usize).map(|_| T::from(rng.random_range(-0.3f32..0.3f32)).unwrap()).collect();

        let zp_stride = if bits == 4 {
            num_groups_k.div_ceil(2)
        } else {
            num_groups_k
        };
        let (zero_points, biases) = match quant_method {
            QuantizationMethod::ScaleBias => (
                None,
                Some(
                    (0..(n * num_groups_k) as usize)
                        .map(|_| T::from(rng.random_range(-0.03f32..0.03f32)).unwrap())
                        .collect(),
                ),
            ),
            QuantizationMethod::ScaleZeroPoint => {
                (Some((0..(n * zp_stride) as usize).map(|_| rng.random_range(0u8..u8::MAX)).collect()), None)
            },
            QuantizationMethod::ScaleSymmetric => (None, None),
        };

        Self {
            w_packed,
            scales,
            zero_points,
            biases,
            x,
            ab_scale: 1.0,
            soft_cap: None,
            k,
            n,
            m,
            group_size,
            quant_method,
            mode: mode_for_bits(bits),
            params_layout: QuantParamsLayout::OutputGroup,
            signed_codes: false,
            prepared_a: None,
            output_rht_factors: None,
            output_bias: None,
        }
    }

    /// Output RHT over n's 32-column blocks with sign factors, and with `bias` an output bias after it.
    pub fn with_output_rht(
        mut self,
        bias: bool,
    ) -> Self {
        assert!(self.n.is_multiple_of(32), "output RHT needs whole 32-column blocks");
        self.output_rht_factors = Some(
            (0..self.n)
                .map(|column| {
                    if column * 7 % 3 == 0 {
                        -1
                    } else {
                        1
                    }
                })
                .collect(),
        );
        self.output_bias =
            bias.then(|| (0..self.n).map(|column| T::from(0.3 + 0.05 * (column % 4) as f32).unwrap()).collect());
        self
    }

    pub fn with_signed_weight_codes(mut self) -> Self {
        self.signed_codes = true;
        self
    }

    pub fn with_group_output(mut self) -> Self {
        self.params_layout = QuantParamsLayout::GroupOutput;
        self
    }

    pub fn with_prepared_a(
        self,
        activation_scale_group_size: u32,
        sum_group_size: Option<u32>,
    ) -> Self {
        let code_layout = Int8CodeLayout::for_right_bits(DataType::from(self.mode).size_in_bits() as u32)
            .expect("W4/W8 quantization");
        self.with_prepared_a_layout(activation_scale_group_size, sum_group_size, code_layout)
    }

    pub fn with_prepared_a_and_reference(
        self,
        activation_scale_group_size: u32,
        sum_group_size: Option<u32>,
    ) -> (Self, Self) {
        let code_layout = Int8CodeLayout::for_right_bits(DataType::from(self.mode).size_in_bits() as u32)
            .expect("W4/W8 quantization");
        let reference = self.clone().with_prepared_a_layout(
            activation_scale_group_size,
            sum_group_size,
            Int8CodeLayout::Sequential,
        );
        let actual = self.with_prepared_a_layout(activation_scale_group_size, sum_group_size, code_layout);
        (actual, reference)
    }

    fn with_prepared_a_layout(
        mut self,
        activation_scale_group_size: u32,
        sum_group_size: Option<u32>,
        code_layout: Int8CodeLayout,
    ) -> Self {
        self.signed_codes = self.mode != QuantizationMode::U4;
        let rows = self.m;
        let columns = self.k;
        assert!(columns.is_multiple_of(activation_scale_group_size));
        if let Some(group_size) = sum_group_size {
            assert!(columns.is_multiple_of(group_size));
        }
        let context = block_on(<Cpu as Backend>::Context::new()).expect("CPU context");
        let input = create_buffer_with_data::<Cpu, T>(&context, &self.x);
        let factors = create_buffer_with_data::<Cpu, i32>(&context, &vec![1; columns as usize]);
        let element_count = rows * columns;
        let mut values = create_buffer::<Cpu, i8>(&context, element_count as usize);
        let mut scales = create_buffer::<Cpu, f32>(&context, (element_count / activation_scale_group_size) as usize);
        let mut group_sums =
            sum_group_size.map(|group_size| create_buffer::<Cpu, i32>(&context, (element_count / group_size) as usize));
        let quantization = ActivationQuantization::new(
            activation_scale_group_size,
            sum_group_size.unwrap_or(activation_scale_group_size),
            sum_group_size.is_some(),
            code_layout,
        )
        .expect("supported activation quantization");
        let transform =
            ActivationTransform::<Cpu>::quantize(&context, T::data_type(), ActivationFormat::Int8(quantization))
                .expect("CPU activation quantization transform");
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        transform.encode_quantize(
            &input,
            &mut values,
            &mut scales,
            group_sums.as_mut(),
            &factors,
            rows,
            columns,
            &mut command_buffer,
        );
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("CPU activation quantization");

        self.prepared_a = Some(PreparedInt8A {
            values: buffer_to_vec(&*context, &values),
            scales: buffer_to_vec(&*context, &scales),
            group_sums: group_sums.map_or_else(Vec::new, |sums| buffer_to_vec(&*context, &sums)),
            quantization,
        });
        self
    }

    pub fn weights_for_upload(&self) -> Vec<u32> {
        let mut words = self.w_packed.clone();
        let sign_flip_mask = self.signed_codes.then(|| self.mode.weight_codes_sign_flip_mask()).flatten();
        if let Some(mask) = sign_flip_mask {
            let broadcast_mask = u32::from(mask) * 0x0101_0101;
            words.iter_mut().for_each(|word| *word ^= broadcast_mask);
        }
        words
    }

    pub fn weight_buffer_bytes(&self) -> usize {
        size_of_val(self.w_packed.as_slice())
            + size_of_val(self.scales.as_slice())
            + self.biases.as_ref().map_or(0, |biases| size_of_val(biases.as_slice()))
            + self.zero_points.as_ref().map_or(0, |zero_points| size_of_val(zero_points.as_slice()))
    }
}

pub struct QuantBuffers<B: Backend, T: DataTypeRepr + Float> {
    pub w: B::GlobalGpuBuffer,
    pub scales: B::GlobalGpuBuffer,
    pub zp: Option<B::GlobalGpuBuffer>,
    pub bias: Option<B::GlobalGpuBuffer>,
    pub x: B::GlobalGpuBuffer,
    pub prepared_a: Option<B::GlobalGpuBuffer>,
    pub prepared_a_scales: Option<B::GlobalGpuBuffer>,
    pub prepared_a_group_sums: Option<B::GlobalGpuBuffer>,
    pub y: B::GlobalGpuBuffer,
    pub output_rht_factors: Option<B::GlobalGpuBuffer>,
    pub output_bias: Option<B::GlobalGpuBuffer>,
    pub weights_layout: WeightsLayout<B>,
    _t: std::marker::PhantomData<T>,
}

impl<B: Backend, T: DataTypeRepr + Float> QuantBuffers<B, T> {
    pub fn allocate(
        context: &B::Context,
        input: &QuantInput<T>,
    ) -> Self {
        let groups = input.k.div_ceil(input.group_size);
        let params_layout = input.params_layout;
        let params = QuantParams::new(params_layout, input.n, groups);
        let metadata_elements = params.scale_shape().into_iter().product::<u32>() as usize;
        let zero_point_bytes = params.zero_point_shape(input.mode).into_iter().product::<u32>() as usize;
        // Group-output metadata planes are transposed on the host before upload.
        let group_output = params_layout == QuantParamsLayout::GroupOutput;
        let metadata = |values: &[T], transposed: bool| {
            let mut plane = pad(values, metadata_elements);
            if group_output && transposed {
                let value_bits = size_of::<T>() as u32 * u8::BITS;
                transpose_metadata(bytemuck::cast_slice_mut(&mut plane), input.n, groups, value_bits);
            }
            plane
        };
        let (x, prepared_a, prepared_a_scales, prepared_a_group_sums) = activations::<B, T>(context, input);
        Self {
            w: create_buffer_with_data::<B, u32>(context, &input.weights_for_upload()),
            scales: create_buffer_with_data::<B, T>(context, &metadata(&input.scales, true)),
            zp: input.zero_points.as_ref().map(|zero_points| {
                let mut plane = pad(zero_points, zero_point_bytes);
                if group_output && input.quant_method == QuantizationMethod::ScaleZeroPoint {
                    let correction_bits = DataType::from(input.mode).size_in_bits() as u32;
                    transpose_metadata(&mut plane, input.n, groups, correction_bits);
                }
                create_buffer_with_data::<B, u8>(context, &plane)
            }),
            bias: input.biases.as_ref().map(|biases| {
                create_buffer_with_data::<B, T>(
                    context,
                    &metadata(biases, input.quant_method == QuantizationMethod::ScaleBias),
                )
            }),
            x,
            prepared_a,
            prepared_a_scales,
            prepared_a_group_sums,
            y: create_buffer::<B, T>(context, (input.m as usize) * (input.n as usize)),
            output_rht_factors: input
                .output_rht_factors
                .as_ref()
                .map(|factors| create_buffer_with_data::<B, i32>(context, factors)),
            output_bias: input.output_bias.as_ref().map(|bias| create_buffer_with_data::<B, T>(context, bias)),
            weights_layout: Default::default(),
            _t: std::marker::PhantomData,
        }
    }

    /// Replaces the activations, A8 included, with `input`'s, keeping the weights and their layout.
    pub fn upload_activations(
        &mut self,
        context: &B::Context,
        input: &QuantInput<T>,
    ) {
        (self.x, self.prepared_a, self.prepared_a_scales, self.prepared_a_group_sums) =
            activations::<B, T>(context, input);
    }

    pub fn matmul_b<'a>(
        &'a self,
        input: &QuantInput<T>,
    ) -> MatmulB<&'a B::GlobalGpuBuffer> {
        quant_b_variant(
            &self.w,
            &self.scales,
            self.zp.as_ref(),
            self.bias.as_ref(),
            input.params_layout,
            self.weights_layout,
            input,
        )
    }
}

/// `input`'s activations: full precision, then its prepared A8 codes, scales and group sums where it has them.
fn activations<B: Backend, T: DataTypeRepr + Float>(
    context: &B::Context,
    input: &QuantInput<T>,
) -> (B::GlobalGpuBuffer, Option<B::GlobalGpuBuffer>, Option<B::GlobalGpuBuffer>, Option<B::GlobalGpuBuffer>) {
    let prepared = input.prepared_a.as_ref();
    (
        create_buffer_with_data::<B, T>(context, &input.x),
        prepared.map(|prepared| create_buffer_with_data::<B, i8>(context, &prepared.values)),
        prepared.map(|prepared| create_buffer_with_data::<B, f32>(context, &prepared.scales)),
        prepared
            .filter(|prepared| !prepared.group_sums.is_empty())
            .map(|prepared| create_buffer_with_data::<B, i32>(context, &prepared.group_sums)),
    )
}

fn quant_b_variant<TB: BufferRef, T: DataTypeRepr + Float>(
    w: TB,
    scales: TB,
    zero_points: Option<TB>,
    biases: Option<TB>,
    params_layout: QuantParamsLayout,
    weights_layout: WeightsLayout<TB::Backend>,
    input: &QuantInput<T>,
) -> MatmulB<TB> {
    let params = QuantParams::new(params_layout, input.n, input.k.div_ceil(input.group_size));
    let correction = match input.quant_method {
        QuantizationMethod::ScaleBias => QuantizedCorrection::Biases(biases.expect("bias buffer")),
        QuantizationMethod::ScaleZeroPoint => QuantizedCorrection::ZeroPoints(zero_points.expect("zp buffer")),
        QuantizationMethod::ScaleSymmetric => QuantizedCorrection::Symmetric,
    };
    MatmulB::Quantized(QuantizedB {
        codes: w,
        scales,
        correction,
        params,
        mode: input.mode,
        group_size: input.group_size,
        signed_codes: input.signed_codes,
        weights_layout,
    })
}

pub fn quant_arguments<'a, B: Backend, T: DataTypeRepr + Float>(
    buffers: &'a mut QuantBuffers<B, T>,
    input: &QuantInput<T>,
) -> MatmulArguments<
    'a,
    B,
    &'a B::GlobalGpuBuffer,
    &'a B::GlobalGpuBuffer,
    &'a mut B::GlobalGpuBuffer,
    &'a B::GlobalGpuBuffer,
> {
    let QuantBuffers {
        w,
        scales,
        zp,
        bias,
        x,
        prepared_a,
        prepared_a_scales,
        prepared_a_group_sums,
        y,
        output_rht_factors,
        output_bias,
        weights_layout,
        ..
    } = buffers;
    let b = quant_b_variant(&*w, &*scales, zp.as_ref(), bias.as_ref(), input.params_layout, *weights_layout, input);
    let a = match &input.prepared_a {
        Some(prepared) => MatmulA::Int8Symmetric {
            values: prepared_a.as_ref().expect("prepared activation buffer"),
            scales: prepared_a_scales.as_ref().expect("prepared activation scales"),
            group_sums: prepared_a_group_sums.as_ref(),
            quantization: prepared.quantization,
        },
        None => MatmulA::full_precision(&*x),
    };
    MatmulArguments {
        a,
        b,
        b_leading_dimension: None,
        b_transpose: true,
        output: MatmulOutput::new(
            y,
            MatmulDOps {
                ab_scale: input.ab_scale,
                bias: output_bias.as_ref(),
                rht_factors: output_rht_factors.as_ref(),
                soft_cap: input.soft_cap,
            },
        ),
        gather_indices: None,
        m: input.m,
        n: input.n,
        k: input.k,
    }
}

type BackendMatmul<B> = <<B as Backend>::Kernels as Kernels>::MatmulKernel;

/// `input` as a linear loaded with A8 storage runs it, with its buffers: U8 codes signed, and the activations
/// `select_activation_format` picks for the weights' layout, A8 prepared as the backend quantizes. `None` where the
/// backend gives the weights no A8 storage, so they stay row-major with BF16 activations.
pub fn production_quant_input<B: Backend, T: DataTypeRepr + Float>(
    context: &B::Context,
    matmul: &BackendMatmul<B>,
    input: QuantInput<T>,
) -> Option<(QuantInput<T>, ActivationFormat, QuantBuffers<B, T>)> {
    let mut input = match input.mode {
        QuantizationMode::U4 => input,
        _ => input.with_signed_weight_codes(),
    };
    let mut buffers = production_quant_buffers::<B, T>(context, matmul, &input);
    let shape = MatmulShape::from_arguments(&quant_arguments(&mut buffers, &input));
    matmul.select_activation_quantization(&shape, context)?;
    let prepared = matmul.activation_preparation(&shape, buffers.weights_layout, context);
    let format = matmul.select_activation_format(&shape, buffers.weights_layout, prepared, context);
    if let ActivationFormat::Int8(quantization) = format {
        input = input.with_prepared_a(quantization.scale_group_size(), quantization.sum_group_size());
        buffers.upload_activations(context, &input);
    }
    Some((input, format, buffers))
}

/// Buffers of `input` with its weights laid out by the backend's `optimize_weights_layout` where it takes A8, as a
/// loaded linear's.
pub fn production_quant_buffers<B: Backend, T: DataTypeRepr + Float>(
    context: &B::Context,
    matmul: &BackendMatmul<B>,
    input: &QuantInput<T>,
) -> QuantBuffers<B, T> {
    let mut buffers = QuantBuffers::<B, T>::allocate(context, input);
    let shape = MatmulShape::from_arguments(&quant_arguments(&mut buffers, input));
    if matmul.select_activation_quantization(&shape, context).is_some() {
        buffers.weights_layout =
            matmul.optimize_weights_layout(&shape, &mut buffers.w, context).expect("weights layout").0;
    }
    buffers
}

/// A quantized product's (relative, absolute) tolerance against the CPU's product of the same codes. GEMM stages
/// dequantized B as BF16, as Metal does: over 256 U8 levels it errs with the outputs' size.
pub fn quant_tolerance(
    format: ActivationFormat,
    bits: u32,
    expected: &[bf16],
) -> (f64, f64) {
    let abs_tol = match (format, bits) {
        (ActivationFormat::Bf16, 8) => expected.iter().fold(0.4f32, |tol, value| tol.max(0.02 * value.to_f32().abs())),
        _ => 0.4,
    };
    (0.05, f64::from(abs_tol))
}

pub fn run_quant_cpu<T: DataTypeRepr + Float>(input: &QuantInput<T>) -> Vec<T> {
    let context = block_on(<Cpu as Backend>::Context::new()).expect("Cpu context");
    let mut buffers = QuantBuffers::<Cpu, T>::allocate(&context, input);
    run_quant_with(&*context, &mut buffers, input)
}

/// Runs `input` on `buffers`, whose weights a test may have laid out for the backend first.
pub fn run_quant_with<B: Backend, T: DataTypeRepr + Float>(
    context: &B::Context,
    buffers: &mut QuantBuffers<B, T>,
    input: &QuantInput<T>,
) -> Vec<T> {
    let mut matmul =
        <B::Kernels as Kernels>::MatmulKernel::new(context, T::data_type(), T::data_type(), T::data_type())
            .expect("MatmulKernel");
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    matmul.encode(quant_arguments(buffers, input), &mut command_buffer).expect("encode quant");
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, T>(context, &buffers.y)
}
