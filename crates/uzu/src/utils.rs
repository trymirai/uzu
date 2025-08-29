use std::{fs::File, io::BufReader, path::Path};

use crate::config::{DecoderConfig, ModelMetadata};

pub fn load_decoder_config(model_path: &Path) -> Result<DecoderConfig, String> {
    let config_path = model_path.join("config.json");

    let config_file = File::open(&config_path)
        .map_err(|e| format!("Failed to load config: {}", e))?;

    let model_metadata: ModelMetadata =
        serde_json::from_reader(BufReader::new(config_file))
            .map_err(|e| format!("Failed to parse config: {}", e))?;

    return Ok(model_metadata.model_config.decoder_config.clone());
}

pub fn open_weights_file(model_path: &Path) -> Result<File, String> {
    let weights_path = model_path.join("model.safetensors");

    File::open(&weights_path)
        .map_err(|e| format!("Failed to open weights: {}", e))
}
