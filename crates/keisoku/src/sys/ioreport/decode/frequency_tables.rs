/// Borrowed SoC frequency tables — the only context CPU/GPU usage metrics need.
#[derive(Default, Clone, Copy)]
pub struct FrequencyTables<'a> {
    pub(crate) ecpu: &'a [u32],
    pub(crate) pcpu: &'a [u32],
    pub(crate) gpu: &'a [u32],
    pub(crate) ecpu_cores: u8,
    pub(crate) pcpu_cores: u8,
}
