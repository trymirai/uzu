pub mod gemm;

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

/// Output-side post-ops for a matmul: `d = op(a @ b)`.
///
/// Constructed via [`MatmulDOps::new`] so the [`GemmDTransform`] mask and the
/// optional data fields can never disagree. Use [`MatmulDOps::none`] for an
/// op-free transform.
pub struct MatmulDOps<'a, B: Backend> {
    mask: GemmDTransform,
    pub ab_scale: Option<f32>,
    pub accumulate: bool,
    pub bias: Option<&'a Allocation<B>>,
    pub rht_factors: Option<&'a Allocation<B>>,
}

impl<'a, B: Backend> MatmulDOps<'a, B> {
    pub fn new(
        ab_scale: Option<f32>,
        accumulate: bool,
        bias: Option<&'a Allocation<B>>,
        rht_factors: Option<&'a Allocation<B>>,
    ) -> Self {
        let mut mask = GemmDTransform::empty();
        if ab_scale.is_some() {
            mask |= GemmDTransform::SCALE;
        }
        if accumulate {
            mask |= GemmDTransform::ACCUMULATE;
        }
        if bias.is_some() {
            mask |= GemmDTransform::BIAS;
        }
        if rht_factors.is_some() {
            mask |= GemmDTransform::RHT;
        }
        Self {
            mask,
            ab_scale,
            accumulate,
            bias,
            rht_factors,
        }
    }

    pub fn none() -> Self {
        Self::new(None, false, None, None)
    }

    pub fn mask(&self) -> GemmDTransform {
        self.mask
    }

    /// Return a copy of `self` with the ops named by `bits` cleared.
    pub fn without(
        self,
        bits: GemmDTransform,
    ) -> Self {
        Self::new(
            if bits.contains(GemmDTransform::SCALE) {
                None
            } else {
                self.ab_scale
            },
            if bits.contains(GemmDTransform::ACCUMULATE) {
                false
            } else {
                self.accumulate
            },
            if bits.contains(GemmDTransform::BIAS) {
                None
            } else {
                self.bias
            },
            if bits.contains(GemmDTransform::RHT) {
                None
            } else {
                self.rht_factors
            },
        )
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
}
