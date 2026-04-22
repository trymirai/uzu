use serde::{Deserialize, Serialize};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcmBatch {
    pub samples: Vec<f64>,
    pub sample_rate: u32,
    pub channels: u32,
    pub lengths: Vec<u32>,
}

impl PcmBatch {
    pub fn batch_size(&self) -> u32 {
        self.lengths.len() as u32
    }

    pub fn total_frames(&self) -> u32 {
        self.lengths.iter().sum::<_>()
    }

    pub fn into_parts(self) -> (Box<[f64]>, u32, u32, Box<[u32]>) {
        (self.samples.into_boxed_slice(), self.sample_rate, self.channels, self.lengths.into_boxed_slice())
    }
}
