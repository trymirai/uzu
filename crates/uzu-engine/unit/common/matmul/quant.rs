use std::mem::{size_of, size_of_val};

use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

#[cfg(backend = "metal")]
use super::harness::TestDispatch;
#[cfg(backend = "metal")]
use crate::backends::metal::{Metal, MetalContext};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                ActivationQuantization, ActivationTransform, Kernels,
                matmul::{
                    Int8CodeLayout, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, QuantParams,
                    QuantParamsLayout, QuantizedB, QuantizedCorrection,
                },
            },
        },
        cpu::Cpu,
    },
    data_type::DataType,
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
pub struct QuantInput<T: ArrayElement + Float> {
    pub w_packed: Vec<u32>,
    pub scales: Vec<T>,
    pub zero_points: Option<Vec<u8>>,
    pub biases: Option<Vec<T>>,
    pub x: Vec<T>,
    pub k: u32,
    pub n: u32,
    pub m: u32,
    pub group_size: u32,
    pub quant_method: QuantizationMethod,
    pub mode: QuantizationMode,
    pub params_layout: QuantParamsLayout,
    pub signed_codes: bool,
    pub prepared_a: Option<PreparedInt8A>,
}

fn mode_for_bits(bits: u32) -> QuantizationMode {
    match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::U8,
        _ => unreachable!("unsupported bits: {bits}"),
    }
}

fn pad<T: ArrayElement>(
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

impl<T: ArrayElement + Float> QuantInput<T> {
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
            k,
            n,
            m,
            group_size,
            quant_method,
            mode: mode_for_bits(bits),
            params_layout: QuantParamsLayout::OutputGroup,
            signed_codes: false,
            prepared_a: None,
        }
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
        let context = <Cpu as Backend>::Context::new().expect("CPU context");
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
        let transform = ActivationTransform::<Cpu>::quantize(&context, T::data_type(), quantization)
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
        command_buffer.end_encoding().submit().wait_until_completed().expect("CPU activation quantization");

        self.prepared_a = Some(PreparedInt8A {
            values: buffer_to_vec(&values),
            scales: buffer_to_vec(&scales),
            group_sums: group_sums.map_or_else(Vec::new, |sums| buffer_to_vec(&sums)),
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

pub struct QuantBuffers<B: Backend, T: ArrayElement + Float> {
    pub w: B::GlobalBuffer,
    pub scales: B::GlobalBuffer,
    pub zp: Option<B::GlobalBuffer>,
    pub bias: Option<B::GlobalBuffer>,
    pub x: B::GlobalBuffer,
    pub prepared_a: Option<B::GlobalBuffer>,
    pub prepared_a_scales: Option<B::GlobalBuffer>,
    pub prepared_a_group_sums: Option<B::GlobalBuffer>,
    pub y: B::GlobalBuffer,
    _t: std::marker::PhantomData<T>,
}

impl<B: Backend, T: ArrayElement + Float> QuantBuffers<B, T> {
    pub fn allocate(
        context: &B::Context,
        input: &QuantInput<T>,
    ) -> Self {
        let groups = input.k.div_ceil(input.group_size);
        let params_layout = input.params_layout;
        let params = QuantParams::new(params_layout, input.n, groups);
        let metadata_elements = params.scale_shape().into_iter().product::<u32>() as usize;
        let zero_point_bytes = params.zero_point_shape(input.mode).into_iter().product::<u32>() as usize;
        let mut buffers = Self {
            w: create_buffer_with_data::<B, u32>(context, &input.weights_for_upload()),
            scales: create_buffer_with_data::<B, T>(context, &pad(&input.scales, metadata_elements)),
            zp: input
                .zero_points
                .as_ref()
                .map(|zero_points| create_buffer_with_data::<B, u8>(context, &pad(zero_points, zero_point_bytes))),
            bias: input
                .biases
                .as_ref()
                .map(|biases| create_buffer_with_data::<B, T>(context, &pad(biases, metadata_elements))),
            x: create_buffer_with_data::<B, T>(context, &input.x),
            prepared_a: input
                .prepared_a
                .as_ref()
                .map(|prepared| create_buffer_with_data::<B, i8>(context, &prepared.values)),
            prepared_a_scales: input
                .prepared_a
                .as_ref()
                .map(|prepared| create_buffer_with_data::<B, f32>(context, &prepared.scales)),
            prepared_a_group_sums: input
                .prepared_a
                .as_ref()
                .filter(|prepared| !prepared.group_sums.is_empty())
                .map(|prepared| create_buffer_with_data::<B, i32>(context, &prepared.group_sums)),
            y: create_buffer::<B, T>(context, (input.m as usize) * (input.n as usize)),
            _t: std::marker::PhantomData,
        };
        if params_layout == QuantParamsLayout::GroupOutput {
            buffers.transpose_quant_params(input);
        }
        buffers
    }

    pub fn matmul_b<'a>(
        &'a self,
        input: &QuantInput<T>,
    ) -> MatmulB<&'a B::GlobalBuffer> {
        quant_b_variant(&self.w, &self.scales, self.zp.as_ref(), self.bias.as_ref(), input.params_layout, input)
    }

    fn transpose_quant_params(
        &mut self,
        input: &QuantInput<T>,
    ) {
        let columns = input.n;
        let groups = input.k.div_ceil(input.group_size);
        let value_bits = size_of::<T>() as u32 * u8::BITS;
        transpose_metadata(self.scales.as_slice_mut(), columns, groups, value_bits);
        match input.quant_method {
            QuantizationMethod::ScaleBias => {
                transpose_metadata(
                    self.bias.as_mut().expect("bias buffer").as_slice_mut(),
                    columns,
                    groups,
                    value_bits,
                );
            },
            QuantizationMethod::ScaleZeroPoint => {
                let correction_bits = DataType::from(input.mode).size_in_bits() as u32;
                transpose_metadata(
                    self.zp.as_mut().expect("zp buffer").as_slice_mut(),
                    columns,
                    groups,
                    correction_bits,
                );
            },
            QuantizationMethod::ScaleSymmetric => {},
        }
    }
}

fn quant_b_variant<TB: BufferRef, T: ArrayElement + Float>(
    w: TB,
    scales: TB,
    zero_points: Option<TB>,
    biases: Option<TB>,
    params_layout: QuantParamsLayout,
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
    })
}

pub fn quant_arguments<'a, B: Backend, T: ArrayElement + Float>(
    buffers: &'a mut QuantBuffers<B, T>,
    input: &QuantInput<T>,
) -> MatmulArguments<'a, B, &'a B::GlobalBuffer, &'a B::GlobalBuffer, &'a mut B::GlobalBuffer, &'a B::GlobalBuffer> {
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
        ..
    } = buffers;
    let b = quant_b_variant(&*w, &*scales, zp.as_ref(), bias.as_ref(), input.params_layout, input);
    let a = match &input.prepared_a {
        Some(prepared) => MatmulA::Int8Symmetric {
            values: prepared_a.as_ref().expect("prepared activation buffer"),
            scales: prepared_a_scales.as_ref().expect("prepared activation scales"),
            // Symmetric weights carry no correction term, so the GEMM never reads these.
            group_sums: (input.quant_method != QuantizationMethod::ScaleSymmetric)
                .then(|| prepared_a_group_sums.as_ref().expect("prepared activation row sums")),
            scale_group_size: prepared.quantization.scale_group_size(),
            code_layout: prepared.quantization.code_layout(),
        },
        None => MatmulA::FullPrecision {
            values: &*x,
            offset: 0,
        },
    };
    MatmulArguments {
        a,
        b,
        b_leading_dimension: None,
        b_transpose: true,
        d: y,
        d_transform: MatmulDOps::none(),
        gather_indices: None,
        m: input.m,
        n: input.n,
        k: input.k,
    }
}

pub fn run_quant_cpu<T: ArrayElement + Float>(input: &QuantInput<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("Cpu context");
    let mut buffers = QuantBuffers::<Cpu, T>::allocate(&context, input);
    let mut matmul = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        &context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulCpuKernel");
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    matmul.encode(quant_arguments(&mut buffers, input), &mut command_buffer).expect("encode cpu quant");
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<Cpu, T>(&buffers.y)
}

#[cfg(backend = "metal")]
pub fn run_quant_metal<T: ArrayElement + Float>(
    context: &MetalContext,
    input: &QuantInput<T>,
    dispatch: TestDispatch,
) -> Vec<T> {
    let mut buffers = QuantBuffers::<Metal, T>::allocate(context, input);
    let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context,
        T::data_type(),
        T::data_type(),
        T::data_type(),
    )
    .expect("MatmulMetalKernel");
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let args = quant_arguments(&mut buffers, input);
    if let Some(engine) = dispatch {
        matmul.gemm.encode_with_engine(args, engine, &mut command_buffer).expect("forced GEMM engine encode failed");
    } else {
        matmul.encode(args, &mut command_buffer).expect("matmul encode failed");
    }
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<Metal, T>(&buffers.y)
}
