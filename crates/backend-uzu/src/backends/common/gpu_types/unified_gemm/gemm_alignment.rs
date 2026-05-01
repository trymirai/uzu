use debug_display::Display;

#[repr(C, align(4))]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmAlignment {
    pub m_aligned: bool,
    pub n_aligned: bool,
    pub k_aligned: bool,
}
