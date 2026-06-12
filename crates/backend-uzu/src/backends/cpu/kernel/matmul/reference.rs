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
        zero_points: Option<SendPtr<u8>>,
        biases: Option<SendPtr<u8>>,
        bits: usize,
        group_size: usize,
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
                zero_points: None,
                biases: Some(alloc_ptr(biases)),
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
                zero_points: Some(alloc_ptr(zero_points)),
                biases: None,
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
                zero_points: None,
                biases: None,
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
