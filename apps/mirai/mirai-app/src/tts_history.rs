use std::{
    fs,
    path::{Path, PathBuf},
};

use hound::WavReader;
use serde::{Deserialize, Serialize};
use uzu::types::{basic::PcmBatch, model::Model};

use crate::{
    data_ops::tts_audio_dir,
    persistence::{self, now_ms},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct TtsHistoryEntry {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub vendor: String,
    pub text: String,
    pub created_at: u64,
}

fn history_path() -> PathBuf {
    persistence::mirai_data_dir().join("tts-history.json")
}

pub fn list() -> Vec<TtsHistoryEntry> {
    let Ok(bytes) = fs::read(history_path()) else {
        return Vec::new();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn write_all(entries: &[TtsHistoryEntry]) {
    let dir = persistence::mirai_data_dir();
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(json) = serde_json::to_string_pretty(entries) {
        let _ = fs::write(history_path(), json);
    }
}

fn merge_batches(batches: &[PcmBatch]) -> Option<PcmBatch> {
    let first = batches.first()?;
    Some(PcmBatch {
        samples: batches.iter().flat_map(|b| b.samples.iter().copied()).collect(),
        sample_rate: first.sample_rate,
        channels: first.channels,
        lengths: vec![batches.iter().flat_map(|b| b.lengths.iter().copied()).sum()],
    })
}

pub fn save_generation(
    model: &Model,
    vendor: &str,
    text: &str,
    batches: &[PcmBatch],
) -> Option<TtsHistoryEntry> {
    let merged = merge_batches(batches)?;
    if merged.samples.is_empty() {
        return None;
    }
    let id = format!("tts-{}", now_ms());
    let dir = tts_audio_dir();
    fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("{id}.wav"));
    merged.save_as_wav(path.to_string_lossy().into_owned()).ok()?;
    let entry = TtsHistoryEntry {
        id,
        model_id: model.identifier.clone(),
        model_name: model.name(),
        vendor: vendor.to_string(),
        text: text.to_string(),
        created_at: now_ms(),
    };
    let mut entries = list();
    entries.insert(0, entry.clone());
    write_all(&entries);
    Some(entry)
}

pub fn delete(id: &str) {
    let mut entries = list();
    entries.retain(|e| e.id != id);
    write_all(&entries);
    let _ = fs::remove_file(tts_audio_dir().join(format!("{id}.wav")));
}

pub fn load_pcm(id: &str) -> Option<PcmBatch> {
    pcm_from_wav(&tts_audio_dir().join(format!("{id}.wav")))
}

pub fn clear_all() {
    write_all(&[]);
}

fn pcm_from_wav(path: &Path) -> Option<PcmBatch> {
    let mut reader = WavReader::open(path).ok()?;
    let spec = reader.spec();
    let channels = spec.channels as u32;
    let sample_rate = spec.sample_rate;
    let samples: Vec<f64> =
        reader.samples::<i16>().map(|s| s.map(|v| v as f64 / 32767.0)).collect::<Result<_, _>>().ok()?;
    let frames = samples.len() as u32 / channels.max(1);
    Some(PcmBatch {
        samples,
        sample_rate,
        channels,
        lengths: vec![frames],
    })
}
