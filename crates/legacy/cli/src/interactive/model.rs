use shoji::types::model::{Model, ModelAccessibility, ModelSource};
use sysinfo::System;
use uzu::engine::{Engine, EngineError};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ModelResolutionError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error("model `{model}` has checkpoints, but none fit in {memory_total} bytes of total memory")]
    InsufficientMemory {
        model: String,
        memory_total: u64,
    },
}

pub async fn resolve_model_id(
    engine: &Engine,
    model: String,
) -> Result<Option<String>, ModelResolutionError> {
    let models = engine.models().await?;
    let mut system = System::new();
    system.refresh_memory();
    let resolved = if let Some(model) = models.iter().find(|candidate| candidate.identifier == model) {
        model.identifier.clone()
    } else if let Some(model) = resolve_model_shorthand(&models, &model, system.total_memory())? {
        model.identifier.clone()
    } else {
        model
    };
    Ok(Some(resolved))
}

fn resolve_model_shorthand<'a>(
    models: &'a [Model],
    requested: &str,
    memory_total: u64,
) -> Result<Option<&'a Model>, ModelResolutionError> {
    let mut candidates = models.iter().filter(|model| model_shorthand_matches(model, requested)).collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(None);
    }

    candidates.retain(|model| checkpoint_size_bytes(model).is_some_and(|size| size <= memory_total));
    if candidates.is_empty() {
        return Err(ModelResolutionError::InsufficientMemory {
            model: requested.to_string(),
            memory_total,
        });
    }

    candidates.sort_by(|left, right| {
        quantization_priority(left)
            .cmp(&quantization_priority(right))
            .then_with(|| model_parameter_count(right).cmp(&model_parameter_count(left)))
            .then_with(|| quantization_bits(right).cmp(&quantization_bits(left)))
            .then_with(|| left.identifier.cmp(&right.identifier))
    });
    Ok(candidates.into_iter().next())
}

fn model_shorthand_matches(
    model: &Model,
    requested: &str,
) -> bool {
    let Some(family) = &model.family else {
        return false;
    };
    let Some(properties) = &model.properties else {
        return false;
    };

    let vendor = family.vendor.identifier.as_str();
    let family_id = family.identifier.strip_prefix(&format!("{vendor}:")).unwrap_or(&family.identifier);
    let size = properties.identifier.as_str();
    let bits = model.quantization.as_ref().map(|quantization| quantization.bits_per_weight.to_string());

    for include_vendor in [false, true] {
        for include_size in [false, true] {
            for include_quantization_vendor in [false, true] {
                for include_quantization in [false, true] {
                    for include_bits in [false, true] {
                        let mut parts = Vec::with_capacity(6);
                        if include_vendor {
                            parts.push(vendor);
                        }
                        parts.push(family_id);
                        if include_size {
                            parts.push(size);
                        }

                        if include_quantization_vendor || include_quantization || include_bits {
                            let Some(quantization) = &model.quantization else {
                                continue;
                            };
                            if include_quantization_vendor {
                                parts.push(quantization.vendor.identifier.as_str());
                            }
                            if include_quantization {
                                parts.push(quantization.method.as_str());
                            }
                            if include_bits {
                                parts.push(bits.as_deref().expect("quantization bits are available"));
                            }
                        }

                        if parts.join(":") == requested {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

fn checkpoint_size_bytes(model: &Model) -> Option<u64> {
    if let ModelAccessibility::OnDevice {
        source: ModelSource::Registry {
            files,
            ..
        },
    } = &model.accessibility
        && !files.is_empty()
    {
        return files.iter().try_fold(0_u64, |total, file| {
            let size = u64::try_from(file.size).ok()?;
            total.checked_add(size)
        });
    }

    let parameters = u64::try_from(model.properties.as_ref()?.size).ok()?;
    let bits = u64::from(model.quantization.as_ref().map_or(16, |quantization| quantization.bits_per_weight));
    Some(parameters.checked_mul(bits)?.div_ceil(8))
}

fn model_parameter_count(model: &Model) -> i64 {
    model.properties.as_ref().map_or(0, |properties| properties.size)
}

fn quantization_priority(model: &Model) -> u8 {
    match model.quantization.as_ref().map(|quantization| quantization.method.as_str()) {
        Some(method) if method.eq_ignore_ascii_case("mirai-m") => 0,
        Some(method) if method.eq_ignore_ascii_case("mirai-s") => 1,
        _ => 2,
    }
}

fn quantization_bits(model: &Model) -> u32 {
    model.quantization.as_ref().map_or(16, |quantization| quantization.bits_per_weight)
}

#[cfg(test)]
#[path = "../../unit/interactive/model_test.rs"]
mod tests;
