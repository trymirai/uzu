use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
    mem::discriminant,
};

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
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}

/// Op applied to A before the matmul core. `Hash`/`Eq` are implemented on the
/// variant discriminant so that a `HashSet<MatmulAOp>` enforces at most one of
/// each variant — duplicate inserts simply return `false` from `insert` and
/// are discarded.
pub enum MatmulAOp<'a, B: Backend> {
    Rht {
        factors: &'a Allocation<B>,
    },
}

impl<B: Backend> Hash for MatmulAOp<'_, B> {
    fn hash<H: Hasher>(
        &self,
        state: &mut H,
    ) {
        discriminant(self).hash(state);
    }
}

impl<B: Backend> PartialEq for MatmulAOp<'_, B> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        discriminant(self) == discriminant(other)
    }
}

impl<B: Backend> Eq for MatmulAOp<'_, B> {}

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

/// Op applied to D after the matmul core. `Hash`/`Eq` are implemented on the
/// variant discriminant so that a `HashSet<MatmulDOp>` enforces at most one of
/// each variant. The kernel applies set members in canonical order
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

/// D = transform( (prologue(A) @ op(B)) ) where op(B) = B^T when b_transpose
/// else B. For quantized B variants, B is packed and dequantized internally.
pub struct MatmulArguments<'a, B: Backend> {
    /// A: [M, K]
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    /// A-side prologue ops. Empty = no prologue.
    pub a_prologue: HashSet<MatmulAOp<'a, B>>,
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
    pub d_transform: HashSet<MatmulDOp<'a, B>>,
    /// M dimension (rows of A and D).
    pub m: u32,
    /// N dimension (cols of B and D).
    pub n: u32,
    /// K dimension (inner contraction).
    pub k: u32,
}

/// Bitmask + associated data resolved from a [`HashSet<MatmulAOp>`].
#[derive(Clone, Copy)]
pub struct ResolvedAPrologue<'a, B: Backend> {
    pub mask: GemmAPrologue,
    pub rht: Option<&'a Allocation<B>>,
}

/// Bitmask + associated data resolved from a [`HashSet<MatmulDOp>`].
#[derive(Clone, Copy)]
pub struct ResolvedDTransform<'a, B: Backend> {
    pub mask: GemmDTransform,
    pub ab_scale: f32,
    pub bias: Option<&'a Allocation<B>>,
    pub rht: Option<&'a Allocation<B>>,
}

pub fn resolve_a<'a, B: Backend>(ops: &HashSet<MatmulAOp<'a, B>>) -> ResolvedAPrologue<'a, B> {
    let mut mask = GemmAPrologue::empty();
    let mut rht = None;
    for op in ops {
        match op {
            MatmulAOp::Rht {
                factors,
            } => {
                mask.insert(GemmAPrologue::RHT);
                rht = Some(*factors);
            },
        }
    }
    ResolvedAPrologue {
        mask,
        rht,
    }
}

pub fn resolve_d<'a, B: Backend>(ops: &HashSet<MatmulDOp<'a, B>>) -> ResolvedDTransform<'a, B> {
    let mut mask = GemmDTransform::empty();
    let mut ab_scale = 1.0;
    let mut bias = None;
    let mut rht = None;
    for op in ops {
        match op {
            MatmulDOp::Scale {
                ab_scale: s,
            } => {
                mask.insert(GemmDTransform::SCALE);
                ab_scale = *s;
            },
            MatmulDOp::Accumulate => {
                mask.insert(GemmDTransform::ACCUMULATE);
            },
            MatmulDOp::Bias {
                bias: b,
            } => {
                mask.insert(GemmDTransform::BIAS);
                bias = Some(*b);
            },
            MatmulDOp::Rht {
                factors,
            } => {
                mask.insert(GemmDTransform::RHT);
                rht = Some(*factors);
            },
        }
    }
    ResolvedDTransform {
        mask,
        ab_scale,
        bias,
        rht,
    }
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
