#[derive(Default)]
pub struct DramBandwidth {
    pub(crate) read_bytes: f64,
    pub(crate) write_bytes: f64,
    pub(crate) read_histogram: f32,
    pub(crate) write_histogram: f32,
}
