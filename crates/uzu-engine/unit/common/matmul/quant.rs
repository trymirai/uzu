mod prepared_int8_a;
mod quant_buffers;
mod quant_input;

use num_traits::Float;
pub use prepared_int8_a::PreparedInt8A;
pub use quant_buffers::QuantBuffers;
pub use quant_input::QuantInput;

#[cfg(backend = "metal")]
use super::harness::TestDispatch;
#[cfg(backend = "metal")]
use crate::backends::metal::{Metal, MetalContext};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::{
                Kernels,
                matmul::{
                    MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput, QuantParams,
                    QuantParamsLayout, QuantizedB, QuantizedCorrection,
                },
            },
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::buffer_to_vec,
};

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
        output: MatmulOutput::new(y, MatmulDOps::none()),
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
        matmul.encode_with_gemm_engine(args, engine, &mut command_buffer).expect("forced GEMM engine encode failed");
    } else {
        matmul.encode(args, &mut command_buffer).expect("matmul encode failed");
    }
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<Metal, T>(&buffers.y)
}
