use hound::{SampleFormat, WavSpec, WavWriter};

use crate::{extensions::Error, types::session::text_to_speech::PcmBatch};

impl PcmBatch {
    pub fn save_as_wav(
        &self,
        path: String,
    ) -> Result<(), Error> {
        let path = std::path::Path::new(&path);
        if path.exists() {
            std::fs::remove_file(path).map_err(|error| Error::UnableToSaveAsWav {
                message: error.to_string(),
            })?;
        }

        let spec = WavSpec {
            channels: self.channels as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec).map_err(|error| Error::UnableToSaveAsWav {
            message: error.to_string(),
        })?;
        for sample in self.samples.iter() {
            writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(|error| {
                Error::UnableToSaveAsWav {
                    message: error.to_string(),
                }
            })?;
        }
        writer.finalize().map_err(|error| Error::UnableToSaveAsWav {
            message: error.to_string(),
        })?;

        Ok(())
    }
}
