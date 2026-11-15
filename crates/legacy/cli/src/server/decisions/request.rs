use base64::{Engine as _, engine::general_purpose::STANDARD};
use indexmap::IndexMap;
use serde::Deserialize;
use shoji::types::session::decision::DecisionInput;
use uzu_engine::d1_omni::{D1Instruction, D1MediaInput, D1Options, D1Question, ScoreReadout};

use crate::common::{audio::decode_audio_bytes_16khz_mono, decision::DecisionRequestQuestion};

#[derive(Deserialize, Default)]
pub struct DecisionRequestOptions {
    #[serde(default)]
    pub score_readout: Option<String>,
    #[serde(default)]
    pub set_size: Option<usize>,
    #[serde(default)]
    pub abstain_below: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    // Accepted and not used: replies name the served model.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    #[serde(default)]
    pub state: Option<serde_json::Value>,
    // Raw values, so a malformed question is reported under its id.
    pub questions: IndexMap<String, serde_json::Value>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub videos: Vec<Vec<String>>,
    #[serde(default)]
    pub audio: Vec<String>,
    #[serde(default)]
    pub media: Vec<serde_json::Value>,
    #[serde(default)]
    pub options: Option<DecisionRequestOptions>,
}

pub struct InvalidQuestion {
    pub id: String,
    pub message: String,
}

fn decode_media_bytes(raw: &str) -> Result<Vec<u8>, String> {
    let b64 = if let Some(rest) = raw.strip_prefix("data:") {
        let (_, data) = rest.split_once(',').ok_or_else(|| "invalid data URL".to_string())?;
        data
    } else {
        raw
    };
    STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("invalid base64 media payload: {e}"))
}

fn decode_image_item(raw: &str) -> Result<D1MediaInput, String> {
    let bytes = decode_media_bytes(raw)?;
    let img = image::load_from_memory(&bytes).map_err(|e| format!("invalid image: {e}"))?;
    Ok(D1MediaInput::Image(img))
}

fn decode_audio_item(raw: &str) -> Result<D1MediaInput, String> {
    let bytes = decode_media_bytes(raw)?;
    let waveform = decode_audio_bytes_16khz_mono(&bytes, None, None)
        .map_err(|e| format!("invalid audio: {e}"))?;
    Ok(D1MediaInput::Audio {
        waveform,
    })
}

fn decode_video_frames(frames: &[String]) -> Result<D1MediaInput, String> {
    if frames.is_empty() {
        return Err("video frame list must not be empty".to_string());
    }
    let mut imgs = Vec::with_capacity(frames.len());
    for raw in frames {
        let bytes = decode_media_bytes(raw)?;
        let img = image::load_from_memory(&bytes).map_err(|e| format!("invalid video frame: {e}"))?;
        imgs.push(img);
    }
    Ok(D1MediaInput::Video(imgs))
}

fn resolve_spoken_instruction(q: &mut D1Question) -> Result<(), String> {
    if let D1Instruction::Spoken {
        audio,
        data,
        path,
        waveform,
        ..
    } = q.instructions_mut()
    {
        if waveform.is_some() {
            return Ok(());
        }
        let raw_opt = data
            .as_deref()
            .or_else(|| audio.as_ref().and_then(|v| v.as_str()))
            .or_else(|| audio.as_ref().and_then(|v| v.get("data").and_then(|s| s.as_str())));
        let bytes = if let Some(raw) = raw_opt {
            decode_media_bytes(raw)?
        } else if let Some(p) = path.as_deref() {
            std::fs::read(p).map_err(|e| format!("failed to read spoken audio `{p}`: {e}"))?
        } else {
            return Err("spoken instruction missing `data`, `audio`, or `path`".to_string());
        };
        let samples = decode_audio_bytes_16khz_mono(&bytes, None, None)
            .map_err(|e| format!("invalid spoken instruction audio: {e}"))?;
        *waveform = Some(samples);
    }
    Ok(())
}

impl DecisionRequest {
    // The question ids in request order, and the input that decides them in the same order.
    pub fn into_input(self) -> Result<(Vec<String>, DecisionInput), InvalidQuestion> {
        if !self.images.is_empty() || !self.videos.is_empty() || !self.audio.is_empty() || !self.media.is_empty() {
            return Err(InvalidQuestion {
                id: "media".to_string(),
                message: "this decision model only accepts text `state`".to_string(),
            });
        }
        let state_str = match self.state {
            None | Some(serde_json::Value::Null) => String::new(),
            Some(serde_json::Value::String(s)) => s,
            Some(_) => {
                return Err(InvalidQuestion {
                    id: "state".to_string(),
                    message: "this decision model requires `state` to be a string".to_string(),
                });
            },
        };
        let mut ids = Vec::with_capacity(self.questions.len());
        let mut questions = Vec::with_capacity(self.questions.len());
        for (id, question) in self.questions {
            let question =
                serde_json::from_value::<DecisionRequestQuestion>(question).map_err(|error| InvalidQuestion {
                    id: id.clone(),
                    message: error.to_string(),
                })?;
            ids.push(id);
            questions.push(question.into());
        }
        let input = DecisionInput {
            state: state_str,
            questions,
        };
        Ok((ids, input))
    }

    pub fn into_d1_input(
        self,
    ) -> Result<(Option<serde_json::Value>, Vec<(String, D1Question)>, Vec<D1MediaInput>, D1Options), InvalidQuestion>
    {
        let mut questions = Vec::with_capacity(self.questions.len());
        for (id, q) in self.questions {
            let mut parsed = D1Question::from_json(q).map_err(|message| InvalidQuestion {
                id: id.clone(),
                message,
            })?;
            resolve_spoken_instruction(&mut parsed).map_err(|message| InvalidQuestion {
                id: id.clone(),
                message,
            })?;
            questions.push((id, parsed));
        }

        let mut media = Vec::new();
        for img in &self.images {
            media.push(decode_image_item(img).map_err(|message| InvalidQuestion {
                id: "images".to_string(),
                message,
            })?);
        }
        for vid in &self.videos {
            media.push(decode_video_frames(vid).map_err(|message| InvalidQuestion {
                id: "videos".to_string(),
                message,
            })?);
        }
        for aud in &self.audio {
            media.push(decode_audio_item(aud).map_err(|message| InvalidQuestion {
                id: "audio".to_string(),
                message,
            })?);
        }
        for item in &self.media {
            let kind = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if kind == "video" {
                let frame_vals = item
                    .get("frames")
                    .or_else(|| item.get("images"))
                    .or_else(|| item.get("video"))
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| InvalidQuestion {
                        id: "media".to_string(),
                        message: "video media item missing `frames` (or `images`/`video`) array".to_string(),
                    })?;
                let frames: Vec<String> = frame_vals
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_str()
                            .map(str::to_string)
                            .ok_or_else(|| InvalidQuestion {
                                id: "media".to_string(),
                                message: format!("video frame {i} must be a base64 string"),
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                media.push(decode_video_frames(&frames).map_err(|message| InvalidQuestion {
                    id: "media".to_string(),
                    message,
                })?);
                continue;
            }
            let payload = item
                .get("data")
                .or_else(|| item.get(kind))
                .and_then(|v| v.as_str())
                .ok_or_else(|| InvalidQuestion {
                    id: "media".to_string(),
                    message: format!("media item missing `{kind}` or `data` base64 string"),
                })?;
            match kind {
                "image" => media.push(decode_image_item(payload).map_err(|message| InvalidQuestion {
                    id: "media".to_string(),
                    message,
                })?),
                "audio" => media.push(decode_audio_item(payload).map_err(|message| InvalidQuestion {
                    id: "media".to_string(),
                    message,
                })?),
                other => {
                    return Err(InvalidQuestion {
                        id: "media".to_string(),
                        message: format!("unsupported media type `{other}`"),
                    });
                },
            }
        }

        let opts = self.options.unwrap_or_default();
        let score_readout = match opts.score_readout.as_deref() {
            None | Some("median") => ScoreReadout::Median,
            Some("mean") => ScoreReadout::Mean,
            Some("argmax") => ScoreReadout::Argmax,
            Some(other) => {
                return Err(InvalidQuestion {
                    id: "options.score_readout".to_string(),
                    message: format!("unknown score_readout `{other}`"),
                });
            },
        };
        let d1_opts = D1Options {
            score_readout,
            set_size: opts.set_size,
            abstain_below: opts.abstain_below,
        };
        let state = self.state.filter(|v| !v.is_null());
        Ok((state, questions, media, d1_opts))
    }
}
