use bitflags::bitflags;
use derive_more::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmComputeKind {
    SimdgroupMma,
    MxuMma,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmInputPrologueKind {
    FullPrecision,
    ExternalRht,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmOutputTransformKind {
    Store,
    Scale,
    Accumulate,
    Bias,
    Rht,
    ScaleAccumulate,
    ScaleAccumulateBias,
    ScaleAccumulateBiasRht,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmWeightPrologueKind {
    FullPrecision,
    ScaleBiasDequant,
    ScaleZeroPointDequant,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmTilingConfig {
    pub threadgroup_m: u32,
    pub threadgroup_n: u32,
    pub threadgroup_k: u32,
    pub simdgroups_m: u32,
    pub simdgroups_n: u32,
}

bitflags! {
    /// Which GEMM axes are evenly divisible by the tile shape. Passed to the
    /// kernel through a `uint32_t` function-constant slot whose bits the
    /// Metal side tests directly — there is no corresponding GPU struct.
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GemmAlignment: u32 {
        const M = 1 << 0;
        const N = 1 << 1;
        const K = 1 << 2;
    }
}

impl GemmAlignment {
    pub fn from_axes(
        m: bool,
        n: bool,
        k: bool,
    ) -> Self {
        let mut bits = Self::empty();
        bits.set(Self::M, m);
        bits.set(Self::N, n);
        bits.set(Self::K, k);
        bits
    }
}
