use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
    mem::discriminant,
};

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Allocation, AsBufferRangeRef, Backend, Buffer, Encoder,
        gpu_types::{QuantizationMode, gemm::GemmDTransform},
        kernel::ManualKernels,
    },
};

#[derive(Debug, Error)]
pub enum MatmulError<B: Backend> {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
    #[error("Hadamard not supported for this kernel configuration")]
    UnsupportedHadamard,
    #[error("Unsupported D-transform op {bit:?} on path {path}")]
    UnsupportedDOp {
        bit: GemmDTransform,
        path: &'static str,
    },
    #[error("Unsupported B layout on path {path}")]
    UnsupportedLayout {
        path: &'static str,
    },
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}

pub enum MatmulB<'a, B: Backend, TB: AsBufferRangeRef = Allocation<B>> {
    FullPrecision {
        b: &'a TB,
    },
    ScaleBiasDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        biases: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
    ScaleZeroPointDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        zero_points: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
}

pub enum MatmulDOp<'a, B: Backend> {
    Scale {
        ab_scale: f32,
    },
    Accumulate,
    Bias {
        bias: &'a Allocation<B>,
    },
    Rht {
        factors: &'a Allocation<B>,
    },
}

impl<'a, B: Backend> MatmulDOp<'a, B> {
    pub fn bit(&self) -> GemmDTransform {
        match self {
            MatmulDOp::Scale {
                ..
            } => GemmDTransform::SCALE,
            MatmulDOp::Accumulate => GemmDTransform::ACCUMULATE,
            MatmulDOp::Bias {
                ..
            } => GemmDTransform::BIAS,
            MatmulDOp::Rht {
                ..
            } => GemmDTransform::RHT,
        }
    }

    pub fn mask(set: &HashSet<Self>) -> GemmDTransform {
        set.iter().fold(GemmDTransform::empty(), |m, op| m | op.bit())
    }

    pub fn as_scale(&self) -> Option<f32> {
        match self {
            MatmulDOp::Scale {
                ab_scale,
            } => Some(*ab_scale),
            _ => None,
        }
    }

    pub fn as_bias(&self) -> Option<&'a Allocation<B>> {
        match self {
            MatmulDOp::Bias {
                bias,
            } => Some(*bias),
            _ => None,
        }
    }

    pub fn as_rht(&self) -> Option<&'a Allocation<B>> {
        match self {
            MatmulDOp::Rht {
                factors,
            } => Some(*factors),
            _ => None,
        }
    }
}

impl<B: Backend> Hash for MatmulDOp<'_, B> {
    fn hash<H: Hasher>(
        &self,
        state: &mut H,
    ) {
        discriminant(self).hash(state);
    }
}

impl<B: Backend> PartialEq for MatmulDOp<'_, B> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        discriminant(self) == discriminant(other)
    }
}

impl<B: Backend> Eq for MatmulDOp<'_, B> {}

pub struct MatmulArguments<'a, B: Backend, TB: AsBufferRangeRef = Allocation<B>> {
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    pub b: MatmulB<'a, B, TB>,
    pub b_offset: usize,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub d: &'a mut Allocation<B>,
    pub d_transform: HashSet<MatmulDOp<'a, B>>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub trait MatmulKernel: Sized {
    type Backend: Backend<Kernels: ManualKernels<MatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        data_type: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Self::Backend>>>(
        &mut self,
        arguments: MatmulArguments<Self::Backend, TB>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
