use half::{bf16, f16};

use crate::{
    backends::{
        common::{AsBufferRangeRef, Buffer, gpu_types::QuantizationMode, kernel::matmul::MatmulB},
        cpu::{Cpu, buffer::BufferDowncastExt},
    },
    data_type::DataType,
    utils::pointers::SendPtr,
};

pub(super) enum WeightData {
    FullPrecision {
        ptr: SendPtr<u8>,
        leading_dimension: usize,
        transpose: bool,
    },
    Quantized {
        weights: SendPtr<u8>,
        scales: SendPtr<u8>,
        dequantization: QuantizedDequantization,
        bits: usize,
        group_size: usize,
    },
}

pub(super) enum QuantizedDequantization {
    ScaleBias {
        biases: SendPtr<u8>,
    },
    ScaleZeroPoint {
        zero_points: SendPtr<u8>,
    },
    ScaleSymmetric,
    LloydMax {
        codebook: SendPtr<u8>,
        bias_indices: SendPtr<u8>,
        bias_codebook: SendPtr<u8>,
    },
}

impl WeightData {
    pub(super) fn from_b<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Cpu>>>(
        b: MatmulB<'_, Cpu, TB>,
        b_offset: usize,
        b_leading_dimension: Option<u32>,
        b_transpose: bool,
        weights_data_type: DataType,
        k: usize,
        n: usize,
    ) -> Self {
        let alloc_ptr = |a: &crate::backends::common::Allocation<Cpu>| {
            let r = a.as_buffer_range_ref();
            SendPtr(unsafe { &*r.buffer().get() }.as_ptr().wrapping_byte_add(r.range().start))
        };
        let bits_of = |mode| match mode {
            QuantizationMode::U4 => 4usize,
            _ => 8usize,
        };
        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                let leading_dimension = b_leading_dimension.map(|ld| ld as usize).unwrap_or(if b_transpose {
                    k
                } else {
                    n
                });
                let r = weights.as_buffer_range_ref();
                let byte_off = r.range().start + b_offset * weights_data_type.size_in_bytes();
                WeightData::FullPrecision {
                    ptr: SendPtr(unsafe { &*r.buffer().downcast().get() }.as_ptr().wrapping_byte_add(byte_off)),
                    leading_dimension,
                    transpose: b_transpose,
                }
            },
            MatmulB::ScaleBiasDequant {
                b: weights,
                scales,
                biases,
                mode,
                group_size,
            } => WeightData::Quantized {
                weights: alloc_ptr(weights),
                scales: alloc_ptr(scales),
                dequantization: QuantizedDequantization::ScaleBias {
                    biases: alloc_ptr(biases),
                },
                bits: bits_of(mode),
                group_size: group_size as usize,
            },
            MatmulB::ScaleZeroPointDequant {
                b: weights,
                scales,
                zero_points,
                mode,
                group_size,
            } => WeightData::Quantized {
                weights: alloc_ptr(weights),
                scales: alloc_ptr(scales),
                dequantization: QuantizedDequantization::ScaleZeroPoint {
                    zero_points: alloc_ptr(zero_points),
                },
                bits: bits_of(mode),
                group_size: group_size as usize,
            },
            MatmulB::ScaleSymmetricDequant {
                b: weights,
                scales,
                mode,
                group_size,
            } => WeightData::Quantized {
                weights: alloc_ptr(weights),
                scales: alloc_ptr(scales),
                dequantization: QuantizedDequantization::ScaleSymmetric,
                bits: bits_of(mode),
                group_size: group_size as usize,
            },
            MatmulB::LloydMaxDequant {
                b: weights,
                scales,
                codebook,
                bias_indices,
                bias_codebook,
                mode,
                group_size,
            } => WeightData::Quantized {
                weights: alloc_ptr(weights),
                scales: alloc_ptr(scales),
                dequantization: QuantizedDequantization::LloydMax {
                    codebook: alloc_ptr(codebook),
                    bias_indices: alloc_ptr(bias_indices),
                    bias_codebook: alloc_ptr(bias_codebook),
                },
                bits: bits_of(mode),
                group_size: group_size as usize,
            },
        }
    }
}

#[inline]
pub(super) unsafe fn read_f32(
    base: *const u8,
    data_type: DataType,
    index: usize,
) -> f32 {
    unsafe {
        match data_type {
            DataType::F32 => *(base as *const f32).add(index),
            DataType::F16 => (*(base as *const f16).add(index)).to_f32(),
            DataType::BF16 => (*(base as *const bf16).add(index)).to_f32(),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub(super) unsafe fn write_f32(
    base: *mut u8,
    data_type: DataType,
    index: usize,
    value: f32,
) {
    unsafe {
        match data_type {
            DataType::F32 => *(base as *mut f32).add(index) = value,
            DataType::F16 => *(base as *mut f16).add(index) = f16::from_f32(value),
            DataType::BF16 => *(base as *mut bf16).add(index) = bf16::from_f32(value),
            _ => unreachable!(),
        }
    }
}
