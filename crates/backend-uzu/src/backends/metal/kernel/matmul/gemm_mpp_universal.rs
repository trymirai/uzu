#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmMppUniversalSpecialization {
    pub apply_ab_scale: bool,
    pub is_accumulate: bool,
}

impl GemmMppUniversalSpecialization {
    pub const fn select(
        apply_ab_scale: bool,
        is_accumulate: bool,
    ) -> Self {
        Self {
            apply_ab_scale,
            is_accumulate,
        }
    }
}
