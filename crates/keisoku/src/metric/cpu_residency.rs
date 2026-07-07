#[derive(Default)]
pub struct CpuResidency {
    pub(crate) ecpu: Vec<(u32, f32)>,
    pub(crate) pcpu: Vec<(u32, f32)>,
}
