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

/// Describes the B operand encoding. Memory layout fields
/// (`b_leading_dimension`, `b_transpose`) are common to all encodings and live
/// on [`MatmulArguments`].
///
/// `TB` is the buffer type for the full-precision B operand; it defaults to
/// [`Allocation<B>`] but can be specialized to KV-cache buffer types
/// (sparse/dense) as long as `&TB: AsBufferRangeRef`. Quantized variants always
/// use [`Allocation<B>`] because quant weights are never KV-cache buffers.
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

/// Op applied to D after the matmul core. `Hash`/`Eq` are implemented on the
/// variant discriminant so that a `HashSet<MatmulDOp>` enforces at most one of
/// each variant. Each variant exposes its corresponding [`GemmDTransform`] bit
/// via [`MatmulDOp::bit`]. The kernel applies set members in canonical order
/// scale → accumulate → bias → rht regardless of insertion order.
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
    /// The bit-flag corresponding to this op's variant.
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

    /// OR-fold the bits of every op in a set.
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

/// D = transform( A @ op(B) ) where op(B) = B^T when b_transpose else B.
/// For quantized B variants, B is packed and dequantized internally. Input-side
/// transforms (e.g. hadamard on A) are the caller's responsibility before invocation.
pub struct MatmulArguments<'a, B: Backend, TB: AsBufferRangeRef = Allocation<B>> {
    /// A: [M, K]
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    /// B operand: encoding + variant-specific aux buffers.
    pub b: MatmulB<'a, B, TB>,
    pub b_offset: usize,
    /// Leading dimension of B (uniform across encodings).
    pub b_leading_dimension: Option<u32>,
    /// Whether B is transposed (uniform across encodings).
    pub b_transpose: bool,
    /// D: [M, N]
    pub d: &'a mut Allocation<B>,
    /// D-side transform ops. Empty = store, no transform.
    pub d_transform: HashSet<MatmulDOp<'a, B>>,
    /// M dimension (rows of A and D).
    pub m: u32,
    /// N dimension (cols of B and D).
    pub n: u32,
    /// K dimension (inner contraction).
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
