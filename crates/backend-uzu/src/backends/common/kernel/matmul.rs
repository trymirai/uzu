use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{
            QuantizationMode,
            gemm::{GemmAPrologue, GemmDTransform},
        },
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
    #[error("Unsupported A-prologue op {bit:?} on path {path}")]
    UnsupportedAOp {
        bit: GemmAPrologue,
        path: &'static str,
    },
    #[error("Unsupported D-transform op {bit:?} on path {path}")]
    UnsupportedDOp {
        bit: GemmDTransform,
        path: &'static str,
    },
    #[error("Unsupported B layout on path {path}")]
    UnsupportedLayout {
        path: &'static str,
    },
    #[error("Duplicate A-prologue op {0:?}")]
    DuplicateAOp(GemmAPrologue),
    #[error("Duplicate D-transform op {0:?}")]
    DuplicateDOp(GemmDTransform),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}

/// Op applied to A before the matmul core. A SET of these in
/// [`MatmulArguments::a_prologue`] describes the A-side prologue; empty slice
/// means no prologue (pass A through unchanged).
pub enum MatmulAOp<'a, B: Backend> {
    Rht {
        factors: &'a Allocation<B>,
    },
}

/// Describes the B operand encoding. Memory layout fields
/// (`b_leading_dimension`, `b_transpose`) are common to all encodings and live
/// on [`MatmulArguments`].
pub enum MatmulB<'a, B: Backend> {
    FullPrecision {
        b: &'a Allocation<B>,
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

/// Op applied to D after the matmul core. A SET of these in
/// [`MatmulArguments::d_transform`] describes the D-side transform; empty slice
/// means store-without-transform.
///
/// Canonical application order is scale → accumulate → bias → rht regardless
/// of caller order. Duplicate ops within a single call return
/// [`MatmulError::DuplicateDOp`].
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

/// D = transform( (prologue(A) @ op(B)) ) where op(B) = B^T when b_transpose
/// else B. For quantized B variants, B is packed and dequantized internally.
pub struct MatmulArguments<'a, B: Backend> {
    /// A: [M, K]
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    /// A-side prologue ops. Empty = no prologue.
    pub a_prologue: &'a [MatmulAOp<'a, B>],
    /// B operand: encoding + variant-specific aux buffers.
    pub b: MatmulB<'a, B>,
    pub b_offset: usize,
    /// Leading dimension of B (uniform across encodings).
    pub b_leading_dimension: Option<u32>,
    /// Whether B is transposed (uniform across encodings).
    pub b_transpose: bool,
    /// D: [M, N]
    pub d: &'a mut Allocation<B>,
    /// D-side transform ops. Empty = store, no transform.
    pub d_transform: &'a [MatmulDOp<'a, B>],
    /// M dimension (rows of A and D).
    pub m: u32,
    /// N dimension (cols of B and D).
    pub n: u32,
    /// K dimension (inner contraction).
    pub k: u32,
}

/// Bitmask + associated data resolved from a `&[MatmulAOp]` slice.
#[derive(Clone, Copy)]
pub struct ResolvedAPrologue<'a, B: Backend> {
    pub mask: GemmAPrologue,
    pub rht: Option<&'a Allocation<B>>,
}

/// Bitmask + associated data resolved from a `&[MatmulDOp]` slice.
#[derive(Clone, Copy)]
pub struct ResolvedDTransform<'a, B: Backend> {
    pub mask: GemmDTransform,
    pub ab_scale: f32,
    pub bias: Option<&'a Allocation<B>>,
    pub rht: Option<&'a Allocation<B>>,
}

pub fn resolve_a<'a, B: Backend>(
    ops: &'a [MatmulAOp<'a, B>]
) -> Result<ResolvedAPrologue<'a, B>, MatmulError<B>> {
    let mut mask = GemmAPrologue::empty();
    let mut rht = None;
    for op in ops {
        match op {
            MatmulAOp::Rht {
                factors,
            } => {
                if mask.contains(GemmAPrologue::RHT) {
                    return Err(MatmulError::DuplicateAOp(GemmAPrologue::RHT));
                }
                mask.insert(GemmAPrologue::RHT);
                rht = Some(*factors);
            },
        }
    }
    Ok(ResolvedAPrologue {
        mask,
        rht,
    })
}

pub fn resolve_d<'a, B: Backend>(
    ops: &'a [MatmulDOp<'a, B>]
) -> Result<ResolvedDTransform<'a, B>, MatmulError<B>> {
    let mut mask = GemmDTransform::empty();
    let mut ab_scale = 1.0;
    let mut bias = None;
    let mut rht = None;
    for op in ops {
        match op {
            MatmulDOp::Scale {
                ab_scale: s,
            } => {
                if mask.contains(GemmDTransform::SCALE) {
                    return Err(MatmulError::DuplicateDOp(GemmDTransform::SCALE));
                }
                mask.insert(GemmDTransform::SCALE);
                ab_scale = *s;
            },
            MatmulDOp::Accumulate => {
                if mask.contains(GemmDTransform::ACCUMULATE) {
                    return Err(MatmulError::DuplicateDOp(GemmDTransform::ACCUMULATE));
                }
                mask.insert(GemmDTransform::ACCUMULATE);
            },
            MatmulDOp::Bias {
                bias: b,
            } => {
                if mask.contains(GemmDTransform::BIAS) {
                    return Err(MatmulError::DuplicateDOp(GemmDTransform::BIAS));
                }
                mask.insert(GemmDTransform::BIAS);
                bias = Some(*b);
            },
            MatmulDOp::Rht {
                factors,
            } => {
                if mask.contains(GemmDTransform::RHT) {
                    return Err(MatmulError::DuplicateDOp(GemmDTransform::RHT));
                }
                mask.insert(GemmDTransform::RHT);
                rht = Some(*factors);
            },
        }
    }
    Ok(ResolvedDTransform {
        mask,
        ab_scale,
        bias,
        rht,
    })
}

pub trait MatmulKernel: Sized {
    type Backend: Backend<Kernels: ManualKernels<MatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Self::Backend>>;

    fn encode(
        &mut self,
        arguments: MatmulArguments<Self::Backend>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), MatmulError<Self::Backend>>;
}
