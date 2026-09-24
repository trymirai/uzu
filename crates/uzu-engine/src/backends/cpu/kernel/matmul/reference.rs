use half::{bf16, f16};

use crate::{
    backends::{
        common::{
            AsBufferRangeRef, BufferArg,
            kernel::matmul::{MatmulB, MatmulError, QuantParamsStrides},
        },
        cpu::Cpu,
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
        zero_points: Option<(SendPtr<u8>, QuantParamsStrides)>,
        biases: Option<SendPtr<u8>>,
        scale_strides: QuantParamsStrides,
        bits: usize,
        group_size: usize,
        signed_codes: bool,
    },
}

impl WeightData {
    pub(super) fn from_b<'a, TB: BufferArg<'a, Cpu>>(
        b: MatmulB<'a, Cpu, TB>,
        b_leading_dimension: Option<u32>,
        b_transpose: bool,
        k: usize,
        n: usize,
    ) -> Result<Self, MatmulError<Cpu>> {
        let alloc_ptr = |a: &crate::backends::common::Allocation<Cpu>| {
            let r = a.as_buffer_range_ref();
            SendPtr(unsafe { &*r.buffer().get() }.as_ptr().wrapping_byte_add(r.range().start))
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
                let (buffer, byte_off, _) = weights.into_parts();
                Ok(WeightData::FullPrecision {
                    ptr: SendPtr(unsafe { &*buffer.downcast().get() }.as_ptr().wrapping_byte_add(byte_off)),
                    leading_dimension,
                    transpose: b_transpose,
                })
            },
            MatmulB::Quantized(quantized) => Ok(WeightData::Quantized {
                weights: alloc_ptr(quantized.codes),
                scales: alloc_ptr(quantized.scales),
                zero_points: quantized.zero_points().map(|values| (alloc_ptr(values), quantized.zero_point_strides())),
                biases: quantized.biases().map(alloc_ptr),
                scale_strides: quantized.params.scale_strides(),
                bits: quantized.bits() as usize,
                group_size: quantized.group_size as usize,
                signed_codes: quantized.signed_codes,
            }),
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
