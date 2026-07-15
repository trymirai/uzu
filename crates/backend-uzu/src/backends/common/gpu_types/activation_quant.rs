use derive_more::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmActivationQuant {
    Disabled,
    Int8Symmetric,
    Int8Asymmetric,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationScaleStat {
    AbsMax,
    Rms,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationScaleGranularity {
    GroupWise,
    TokenWise,
}
