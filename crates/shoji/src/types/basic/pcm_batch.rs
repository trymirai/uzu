use hound::{SampleFormat, WavSpec, WavWriter};
use serde::{Deserialize, Serialize};

#[bindings::export(Error)]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PcmBatchError {
    #[error("Unable to save as wav: {message}")]
    UnableToSaveAsWav {
        message: String,
    },
}

#[bindings::export(Class)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcmBatch {
    pub samples: Vec<f64>,
    pub sample_rate: u32,
    pub channels: u32,
    pub lengths: Vec<u32>,
}

#[bindings::export(Implementation)]
impl PcmBatch {
    #[bindings::export(Method)]
    pub fn batch_size(&self) -> u32 {
        self.lengths.len() as u32
    }

    #[bindings::export(Method)]
    pub fn total_frames(&self) -> u32 {
        self.lengths.iter().sum::<_>()
    }
}

impl PcmBatch {
    pub fn into_parts(self) -> (Box<[f64]>, u32, u32, Box<[u32]>) {
        (self.samples.into_boxed_slice(), self.sample_rate, self.channels, self.lengths.into_boxed_slice())
    }
}

#[bindings::export(Implementation)]
impl PcmBatch {
    #[bindings::export(Method)]
    pub fn save_as_wav(
        &self,
        path: String,
    ) -> Result<(), PcmBatchError> {
        let path = std::path::Path::new(&path);
        if path.exists() {
            std::fs::remove_file(path).map_err(|error| PcmBatchError::UnableToSaveAsWav {
                message: error.to_string(),
            })?;
        }

        let spec = WavSpec {
            channels: self.channels as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec).map_err(|error| PcmBatchError::UnableToSaveAsWav {
            message: error.to_string(),
        })?;
        for sample in self.samples.iter() {
            writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(|error| {
                PcmBatchError::UnableToSaveAsWav {
                    message: error.to_string(),
                }
            })?;
        }
        writer.finalize().map_err(|error| PcmBatchError::UnableToSaveAsWav {
            message: error.to_string(),
        })?;

        Ok(())
    }
}
