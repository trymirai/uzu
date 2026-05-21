use half::{bf16, f16};
use num_traits::Float;

use crate::{
    ArrayElement, DataType,
    backends::{
        common::{
            AsBufferRangeMut, AsBufferRangeRef, Encoder,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::quant_matmul::{QuantizedMatmulArguments, QuantizedMatmulConfiguration},
        },
        cpu::{Cpu, kernel::matmul::gemm::qmm_transposed::qmm_transposed},
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub(crate) fn encode_quantized_gemm(
    encoder: &mut Encoder<Cpu>,
    arguments: QuantizedMatmulArguments<Cpu>,
    configuration: &QuantizedMatmulConfiguration,
) {
    let QuantizedMatmulArguments {
        a,
        a_offset,
        b,
        scales,
        zero_points_or_biases,
        output,
        hadamard_factors: _,
        batch_dim,
    } = arguments;

    let bits: usize = match configuration.mode {
        QuantizationMode::U4 => 4,
        QuantizationMode::I8 | QuantizationMode::U8 => 8,
    };
    let quant_method = configuration.quantization_method;
    let group_size = configuration.group_size;
    let in_vec_size = configuration.input_dim;
    let out_vec_size = configuration.output_dim;
    let batch_size = batch_dim;
    let data_type = configuration.data_type;

    let a_buf = a.as_buffer_range_ref();
    let b_buf = b.as_buffer_range_ref();
    let scales_buf = scales.as_buffer_range_ref();
    let zp_buf = zero_points_or_biases.as_buffer_range_ref();
    let d_buf = output.as_buffer_range_mut();

    let a_byte_off = a_buf.range().start + a_offset * data_type.size_in_bytes();
    let b_byte_off = b_buf.range().start;
    let scales_byte_off = scales_buf.range().start;
    let zp_byte_off = zp_buf.range().start;
    let d_byte_off = d_buf.range().start;

    let a_ptr = SendPtr(unsafe { &*a_buf.buffer().get() }.as_ptr().wrapping_byte_add(a_byte_off));
    let b_ptr = SendPtr(unsafe { &*b_buf.buffer().get() }.as_ptr().wrapping_byte_add(b_byte_off));
    let scales_ptr = SendPtr(unsafe { &*scales_buf.buffer().get() }.as_ptr().wrapping_byte_add(scales_byte_off));
    let zp_ptr = SendPtr(unsafe { &*zp_buf.buffer().get() }.as_ptr().wrapping_byte_add(zp_byte_off));
    let d_ptr = SendPtrMut(unsafe { (&*d_buf.buffer().get()).as_ptr().wrapping_byte_add(d_byte_off) as *mut u8 });

    let command_buffer = encoder.as_command_buffer_mut();
    command_buffer.push_command(move || match data_type {
        DataType::F32 => run::<f32>(
            a_ptr,
            b_ptr,
            scales_ptr,
            zp_ptr,
            d_ptr,
            in_vec_size,
            out_vec_size,
            batch_size,
            quant_method,
            group_size,
            bits,
        ),
        DataType::F16 => run::<f16>(
            a_ptr,
            b_ptr,
            scales_ptr,
            zp_ptr,
            d_ptr,
            in_vec_size,
            out_vec_size,
            batch_size,
            quant_method,
            group_size,
            bits,
        ),
        DataType::BF16 => run::<bf16>(
            a_ptr,
            b_ptr,
            scales_ptr,
            zp_ptr,
            d_ptr,
            in_vec_size,
            out_vec_size,
            batch_size,
            quant_method,
            group_size,
            bits,
        ),
        _ => unreachable!(),
    });
}

fn run<T: ArrayElement + Float>(
    a_ptr: SendPtr<u8>,
    b_ptr: SendPtr<u8>,
    scales_ptr: SendPtr<u8>,
    zp_or_bias_ptr: SendPtr<u8>,
    d_ptr: SendPtrMut<u8>,
    in_vec_size: usize,
    out_vec_size: usize,
    batch_size: usize,
    quant_method: QuantizationMethod,
    group_size: usize,
    bits: usize,
) {
    let (zp, bias) = match quant_method {
        QuantizationMethod::ScaleZeroPoint => (Some(zp_or_bias_ptr.as_ptr() as *const u8), None),
        QuantizationMethod::ScaleBias => (None, Some(zp_or_bias_ptr.as_ptr() as *const T)),
    };
    qmm_transposed::<T>(
        b_ptr.as_ptr() as *const u32,
        scales_ptr.as_ptr() as *const T,
        zp,
        bias,
        a_ptr.as_ptr() as *const T,
        d_ptr.as_ptr() as *mut T,
        in_vec_size,
        out_vec_size,
        batch_size,
        quant_method,
        group_size,
        bits,
    );
}
