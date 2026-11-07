use std::{fmt, path::Path};

use shoji::types::model::{Model, ModelIdentifier};

/// A model whose weights, with room for activations and pipelines, exceed the memory free on the device, and the
/// largest variant of it that fits, where the catalog lists one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryShortfall {
    pub model: ModelIdentifier,
    pub required_bytes: u64,
    pub available_bytes: u64,
    pub smaller_model: Option<ModelIdentifier>,
}

impl MemoryShortfall {
    /// `None` when `model`'s weights and their margin fit in `available_bytes`; `variants` are catalog models with
    /// their download sizes.
    pub fn new<'a>(
        model: &Model,
        weight_bytes: u64,
        available_bytes: u64,
        variants: impl IntoIterator<Item = (&'a Model, u64)>,
    ) -> Option<Self> {
        let required_bytes = with_margin(weight_bytes);
        if required_bytes <= available_bytes {
            return None;
        }
        let smaller_model = variants
            .into_iter()
            .filter(|(variant, bytes)| {
                model.family.is_some()
                    && variant.identifier != model.identifier
                    && variant.family == model.family
                    && variant.properties == model.properties
                    && *bytes > 0
                    && with_margin(*bytes) <= available_bytes
            })
            .max_by_key(|(_, bytes)| *bytes)
            .map(|(variant, _)| variant.identifier.clone());
        Some(Self {
            model: model.identifier.clone(),
            required_bytes,
            available_bytes,
            smaller_model,
        })
    }
}

impl fmt::Display for MemoryShortfall {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let gigabytes = |bytes: u64| bytes as f64 / 1e9;
        write!(
            formatter,
            "{} needs about {:.1} GB of device memory and {:.1} GB is free; ",
            self.model,
            gigabytes(self.required_bytes),
            gigabytes(self.available_bytes)
        )?;
        match &self.smaller_model {
            Some(model) => write!(formatter, "use {model} instead"),
            None => write!(formatter, "no smaller variant of it fits"),
        }
    }
}

/// A model's weights and a tenth more for activations and pipelines; the KV cache is not counted.
fn with_margin(weight_bytes: u64) -> u64 {
    weight_bytes + weight_bytes / 10
}

/// The bytes of every file under `directory`.
pub async fn directory_bytes(directory: &Path) -> std::io::Result<u64> {
    let mut bytes = 0;
    let mut pending = vec![directory.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in kiban::fs::asyn::read_dir(&directory).await? {
            if kiban::fs::asyn::is_dir(&entry).await {
                pending.push(entry);
            } else {
                bytes += kiban::fs::asyn::file_length(&entry).await?;
            }
        }
    }
    Ok(bytes)
}
