pub mod gemm;

use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Allocation, AsBufferRangeRef, Backend, Buffer, Encoder,
        gpu_types::{QuantizationMethod, QuantizationMode, gemm::GemmDTransform},
        kernel::ManualKernels,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct MatmulQuantCombo {
    pub method: QuantizationMethod,
    pub mode: QuantizationMode,
    pub group_size: u32,
}

#[derive(Debug, Error)]
pub enum MatmulError<B: Backend> {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
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

pub struct MatmulDOps<'a, B: Backend> {
    pub ab_scale: f32,
    pub accumulate: bool,
    pub bias: Option<&'a Allocation<B>>,
    pub rht_factors: Option<&'a Allocation<B>>,
}

impl<'a, B: Backend> MatmulDOps<'a, B> {
    pub fn none() -> Self {
        Self {
            ab_scale: 1.0,
            accumulate: false,
            bias: None,
            rht_factors: None,
        }
    }

    pub fn mask(&self) -> GemmDTransform {
        let mut m = GemmDTransform::empty();
        if self.ab_scale != 1.0 {
            m |= GemmDTransform::SCALE;
        }
        if self.accumulate {
            m |= GemmDTransform::ACCUMULATE;
        }
        if self.bias.is_some() {
            m |= GemmDTransform::BIAS;
        }
        if self.rht_factors.is_some() {
            m |= GemmDTransform::RHT;
        }
        m
    }

    pub fn without(
        self,
        bits: GemmDTransform,
    ) -> Self {
        Self {
            ab_scale: if bits.contains(GemmDTransform::SCALE) {
                1.0
            } else {
                self.ab_scale
            },
            accumulate: if bits.contains(GemmDTransform::ACCUMULATE) {
                false
            } else {
                self.accumulate
            },
            bias: if bits.contains(GemmDTransform::BIAS) {
                None
            } else {
                self.bias
            },
            rht_factors: if bits.contains(GemmDTransform::RHT) {
                None
            } else {
                self.rht_factors
            },
        }
    }
}

pub struct MatmulArguments<'a, B: Backend, TB: AsBufferRangeRef = Allocation<B>> {
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    pub b: MatmulB<'a, B, TB>,
    pub b_offset: usize,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub d: &'a mut Allocation<B>,
    pub d_transform: MatmulDOps<'a, B>,
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

    fn preheat_quant_combo(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _combo: MatmulQuantCombo,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        Ok(())
    }
}
