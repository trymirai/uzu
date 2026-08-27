use half::{bf16, f16};

use crate::{
    backends::{
        common::{
            BufferCpuAccessible, BufferRef,
            kernel::matmul::{MatmulB, MatmulError, QuantParamsStrides},
        },
        cpu::{Cpu, buffer::CpuBufferExt},
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
    pub(super) fn from_b(
        b: MatmulB<impl BufferRef<Backend = Cpu>>,
        b_leading_dimension: Option<u32>,
        b_transpose: bool,
        k: usize,
        n: usize,
    ) -> Result<Self, MatmulError<Cpu>> {
        fn buffer_ptr(view: impl BufferRef<Backend = Cpu>) -> SendPtr<u8> {
            let (buffer, range) = view.parts();
            SendPtr(buffer.downcast().cpu_ptr().as_ptr().cast::<u8>().cast_const().wrapping_byte_add(range.start))
        }
        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                let leading_dimension = b_leading_dimension.map(|ld| ld as usize).unwrap_or(if b_transpose {
                    k
                } else {
                    n
                });
                Ok(WeightData::FullPrecision {
                    ptr: buffer_ptr(weights),
                    leading_dimension,
                    transpose: b_transpose,
                })
            },
            MatmulB::Quantized(quantized) => Ok(WeightData::Quantized {
                weights: buffer_ptr(quantized.codes),
                scales: buffer_ptr(quantized.scales),
                zero_points: quantized.zero_points().map(|values| (buffer_ptr(values), quantized.zero_point_strides())),
                biases: quantized.biases().map(buffer_ptr),
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
