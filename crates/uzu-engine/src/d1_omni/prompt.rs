use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::config::{Calibration, question_bucket};

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const BOS: &str = "<|startoftext|>";
pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const EDGE_NAMES: [&str; 4] = ["top", "bottom", "left", "right"];
pub const INSTANCES: usize = 16;
pub const MAX_INSTANCES: usize = 64;
pub const DRAWS: usize = 512;

#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub bos: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub vision_start: u32,
    pub vision_end: u32,
}

impl SpecialTokens {
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self, String> {
        let req = |tok: &str| {
            tokenizer
                .token_to_id(tok)
                .ok_or_else(|| format!("tokenizer missing required special token {tok}"))
        };
        Ok(Self {
            bos: req(BOS)?,
            im_start: req(IM_START)?,
            im_end: req(IM_END)?,
            vision_start: req(VISION_START)?,
            vision_end: req(VISION_END)?,
        })
    }
}

pub use crate::utils::tokenizer::build_plain_tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum D1Instruction {
    Text(String),
    Spoken {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        audio: Option<serde_json::Value>,
        #[serde(default)]
        data: Option<String>,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        waveform: Option<Vec<f32>>,
    },
}

impl D1Instruction {
    pub fn is_spoken(&self) -> bool {
        matches!(self, Self::Spoken { .. })
    }

    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            Self::Spoken {
                ..
            } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum D1Question {
    Noul {
        instructions: D1Instruction,
        criteria: Option<BTreeMap<String, String>>,
    },
    Choice {
        instructions: D1Instruction,
        criteria: Vec<(String, Option<String>)>,
        multiple: bool,
    },
    Score {
        instructions: D1Instruction,
        criteria: Vec<String>,
    },
    Read {
        instructions: D1Instruction,
        kind: String, // "text" | "number" | "date"
        unit: Option<String>,
    },
    Estimate {
        instructions: D1Instruction,
        low: f64,
        high: f64,
        step: Option<f64>,
        unit: Option<String>,
    },
    Box {
        instructions: D1Instruction,
    },
    Boxes {
        instructions: D1Instruction,
        max_boxes: Option<usize>,
    },
}

#[derive(Debug, Deserialize)]
struct RawQuestion {
    #[serde(rename = "type")]
    kind: String,
    instructions: D1Instruction,
    #[serde(default)]
    criteria: Option<serde_json::Value>,
    #[serde(default)]
    multiple: bool,
    #[serde(default)]
    range: Option<[f64; 2]>,
    #[serde(default)]
    step: Option<f64>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    max: Option<usize>,
}

impl D1Question {
    pub fn from_json(value: serde_json::Value) -> Result<Self, String> {
        let raw: RawQuestion = serde_json::from_value(value).map_err(|e| e.to_string())?;
        match raw.kind.as_str() {
            "noul" | "bool" => {
                let criteria = match raw.criteria {
                    Some(serde_json::Value::Object(map)) => Some(
                        map.into_iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
                            .collect(),
                    ),
                    _ => None,
                };
                Ok(Self::Noul {
                    instructions: raw.instructions,
                    criteria,
                })
            },
            "choice" => {
                let map = match raw.criteria {
                    Some(serde_json::Value::Object(m)) => m,
                    _ => return Err("choice requires object criteria".to_string()),
                };
                if map.is_empty() {
                    return Err("choice with no options".to_string());
                }
                let criteria = map
                    .into_iter()
                    .map(|(k, v)| {
                        let desc = match v {
                            serde_json::Value::Null => None,
                            serde_json::Value::String(s) => Some(s),
                            other => Some(other.to_string()),
                        };
                        (k, desc)
                    })
                    .collect();
                Ok(Self::Choice {
                    instructions: raw.instructions,
                    criteria,
                    multiple: raw.multiple,
                })
            },
            "score" => {
                let arr = match raw.criteria {
                    Some(serde_json::Value::Array(a)) => a,
                    _ => return Err("score requires array criteria".to_string()),
                };
                if !(2..=10).contains(&arr.len()) {
                    return Err("score requires 2..=10 levels".to_string());
                }
                let criteria = arr
                    .into_iter()
                    .map(|v| match v {
                        serde_json::Value::String(s) => s,
                        other => other.to_string(),
                    })
                    .collect();
                Ok(Self::Score {
                    instructions: raw.instructions,
                    criteria,
                })
            },
            "text" | "date" => Ok(Self::Read {
                instructions: raw.instructions,
                kind: raw.kind,
                unit: raw.unit,
            }),
            "number" => {
                if let Some([low, high]) = raw.range {
                    if !(low.is_finite() && high.is_finite() && low < high) {
                        return Err("number range must have low < high".to_string());
                    }
                    Ok(Self::Estimate {
                        instructions: raw.instructions,
                        low,
                        high,
                        step: raw.step,
                        unit: raw.unit,
                    })
                } else if raw.step.is_some() {
                    Err("\"step\" needs a \"range\"".to_string())
                } else {
                    Ok(Self::Read {
                        instructions: raw.instructions,
                        kind: "number".to_string(),
                        unit: raw.unit,
                    })
                }
            },
            "box" => Ok(Self::Box {
                instructions: raw.instructions,
            }),
            "boxes" => Ok(Self::Boxes {
                instructions: raw.instructions,
                max_boxes: raw.max,
            }),
            other => Err(format!("unknown question type `{other}`")),
        }
    }

    pub fn instructions(&self) -> &D1Instruction {
        match self {
            Self::Noul {
                instructions,
                ..
            }
            | Self::Choice {
                instructions,
                ..
            }
            | Self::Score {
                instructions,
                ..
            }
            | Self::Read {
                instructions,
                ..
            }
            | Self::Estimate {
                instructions,
                ..
            }
            | Self::Box {
                instructions,
            }
            | Self::Boxes {
                instructions,
                ..
            } => instructions,
        }
    }

    pub fn instructions_mut(&mut self) -> &mut D1Instruction {
        match self {
            Self::Noul {
                instructions,
                ..
            }
            | Self::Choice {
                instructions,
                ..
            }
            | Self::Score {
                instructions,
                ..
            }
            | Self::Read {
                instructions,
                ..
            }
            | Self::Estimate {
                instructions,
                ..
            }
            | Self::Box {
                instructions,
            }
            | Self::Boxes {
                instructions,
                ..
            } => instructions,
        }
    }

    pub fn is_typed(&self) -> bool {
        matches!(
            self,
            Self::Read { .. }
                | Self::Estimate { .. }
                | Self::Box { .. }
                | Self::Boxes { .. }
                | Self::Choice {
                    multiple: true,
                    ..
                }
        )
    }
}

#[derive(Debug, Clone)]
pub struct Grid {
    pub low: f64,
    pub high: f64,
    pub size: usize,
}

impl Grid {
    pub fn new(
        low: f64,
        high: f64,
        size: usize,
    ) -> Self {
        Self {
            low,
            high,
            size,
        }
    }

    pub fn step(&self) -> f64 {
        (self.high - self.low) / (self.size - 1) as f64
    }

    pub fn values(&self) -> Vec<f64> {
        let step = self.step();
        (0..self.size).map(|i| self.low + i as f64 * step).collect()
    }

    pub fn median(
        &self,
        probs: &[f64],
    ) -> f64 {
        let total: f64 = probs.iter().sum::<f64>().max(1e-12);
        let mut mass = 0.0;
        let vals = self.values();
        for (&v, &p) in vals.iter().zip(probs.iter()) {
            mass += p / total;
            if mass >= 0.5 {
                return v;
            }
        }
        self.high
    }

    pub fn crossing(
        &self,
        probs: &[f64],
    ) -> f64 {
        let total: f64 = probs.iter().sum::<f64>().max(1e-12);
        let step = self.step();
        let vals = self.values();
        let mut mass = 0.0;
        for (&v, &p) in vals.iter().zip(probs.iter()) {
            let share = p / total;
            if share > 0.0 && mass + share >= 0.5 {
                let x = v - step / 2.0 + (0.5 - mass) / share * step;
                return x.clamp(self.low, self.high);
            }
            mass += share;
        }
        self.high
    }

    pub fn quantile(
        &self,
        probs: &[f64],
        q: f64,
    ) -> f64 {
        let total: f64 = probs.iter().sum::<f64>().max(1e-12);
        let vals = self.values();
        let mut mass = 0.0;
        for (&v, &p) in vals.iter().zip(probs.iter()) {
            mass += p / total;
            if mass >= q {
                return v;
            }
        }
        self.high
    }
}

pub fn format_py_g(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let s = format!("{v:.6}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

pub fn estimate_grid(
    low: f64,
    high: f64,
    step: Option<f64>,
) -> Grid {
    let size = match step {
        None => 128,
        Some(s) => 128.min(((high - low) / s).round() as usize + 1).max(2),
    };
    Grid::new(low, high, size)
}

pub fn option_codes(
    labels: &[String],
    scheme: u32,
) -> Vec<String> {
    let labs: Vec<String> = labels.iter().map(|s| s.trim().to_string()).collect();
    if !labs.is_empty() && labs.iter().all(|k| k.chars().count() == 1 && k.chars().all(|c| c.is_alphabetic())) {
        return labs;
    }
    if labs.len() <= 26 {
        return (0..labs.len()).map(|i| ((b'A' + i as u8) as char).to_string()).collect();
    }
    if scheme == 2 {
        let mut out = Vec::with_capacity(labs.len());
        for c in b'A'..=b'Z' {
            out.push((c as char).to_string());
        }
        for a in b'A'..=b'Z' {
            for b in b'A'..=b'Z' {
                if out.len() >= labs.len() {
                    return out;
                }
                out.push(format!("{}{}", a as char, b as char));
            }
        }
        return out;
    }
    (0..labs.len()).map(|i| format!("{i:02}")).collect()
}

fn encode_no_special(
    tokenizer: &Tokenizer,
    text: &str,
) -> Vec<u32> {
    tokenizer
        .encode(text, false)
        .map(|enc| enc.get_ids().to_vec())
        .unwrap_or_default()
}

pub fn aliases(
    tokenizer: &Tokenizer,
    labels: &[String],
    scheme: u32,
) -> Result<Vec<(String, u32)>, String> {
    let codes = option_codes(labels, scheme);
    let mut used = HashSet::new();
    let mut out = Vec::with_capacity(codes.len());

    let mut fallback: Vec<String> = (b'A'..=b'Z').map(|c| (c as char).to_string()).collect();
    fallback.extend((0..100).map(|i| format!("{i:02}")));
    fallback.extend((b'a'..=b'z').map(|c| (c as char).to_string()));
    fallback.extend((0..200).map(|i| format!("#{i}")));
    for a in b'A'..=b'Z' {
        for b in b'A'..=b'Z' {
            fallback.push(format!("{}{}", a as char, b as char));
        }
    }

    for code in codes {
        let enc = encode_no_special(tokenizer, &code);
        if enc.len() == 1 && used.insert(enc[0]) {
            out.push((code, enc[0]));
            continue;
        }
        let mut found = false;
        for raw in &fallback {
            let e = encode_no_special(tokenizer, raw);
            if e.len() == 1 && used.insert(e[0]) {
                out.push((raw.clone(), e[0]));
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!("no single-token alias left for {} options", labels.len()));
        }
    }
    Ok(out)
}

pub fn number_ids(
    tokenizer: &Tokenizer,
    grid: &Grid,
    scale: i64,
) -> Option<Vec<Vec<u32>>> {
    let vals = grid.values();
    let mut texts = Vec::with_capacity(vals.len());
    for &v in &vals {
        let scaled = v * scale as f64;
        let rounded = scaled.round();
        if (scaled - rounded).abs() > 1e-6 {
            return None;
        }
        texts.push(format!("{}", rounded as i64));
    }
    let mut groups = Vec::with_capacity(texts.len());
    for t in &texts {
        let plain_enc = encode_no_special(tokenizer, t);
        if plain_enc.len() != 1 {
            return None;
        }
        let mut ids = vec![plain_enc[0]];
        let space_enc = encode_no_special(tokenizer, &format!(" {t}"));
        if space_enc.len() == 1 && !ids.contains(&space_enc[0]) {
            ids.push(space_enc[0]);
        }
        groups.push(ids);
    }
    Some(groups)
}

#[derive(Debug, Clone)]
pub enum SlotSpec {
    Noul,
    Choice {
        labels: Vec<String>,
    },
    Score {
        levels: usize,
    },
    Number {
        grid: Grid,
        scale: i64,
    },
    Position,
}

impl SlotSpec {
    pub fn bucket_kind(&self) -> (&'static str, usize) {
        match self {
            Self::Noul => ("noul", 2),
            Self::Score {
                levels,
            } => ("score", *levels),
            Self::Choice {
                labels,
            } => ("choice", labels.len()),
            Self::Number {
                grid,
                ..
            } => ("choice", grid.size),
            Self::Position => ("position", 0),
        }
    }

    pub fn readout_ids(
        &self,
        tokenizer: &Tokenizer,
        d1_codes: u32,
    ) -> Result<Vec<Vec<u32>>, String> {
        match self {
            Self::Noul => {
                let mut yes = Vec::new();
                for s in ["yes", "Yes", "YES"] {
                    let e = encode_no_special(tokenizer, s);
                    if e.len() == 1 && !yes.contains(&e[0]) {
                        yes.push(e[0]);
                    }
                }
                let mut no = Vec::new();
                for s in ["no", "No", "NO"] {
                    let e = encode_no_special(tokenizer, s);
                    if e.len() == 1 && !no.contains(&e[0]) {
                        no.push(e[0]);
                    }
                }
                if yes.is_empty() || no.is_empty() {
                    return Err("tokenizer has no single-token yes/no".to_string());
                }
                Ok(vec![yes, no])
            },
            Self::Score {
                levels,
            } => {
                let mut groups = Vec::with_capacity(*levels);
                for i in 0..*levels {
                    let e = encode_no_special(tokenizer, &i.to_string());
                    if e.len() != 1 {
                        return Err(format!("score level {i} is not single-token"));
                    }
                    groups.push(e);
                }
                Ok(groups)
            },
            Self::Choice {
                labels,
            } => {
                let al = aliases(tokenizer, labels, d1_codes)?;
                let mut groups = Vec::with_capacity(al.len());
                for (code, tid) in al {
                    let mut g = vec![tid];
                    let extra = encode_no_special(tokenizer, &format!(" {code}"));
                    if extra.len() == 1 && extra[0] != tid {
                        g.push(extra[0]);
                    }
                    groups.push(g);
                }
                Ok(groups)
            },
            Self::Number {
                grid,
                scale,
            } => number_ids(tokenizer, grid, *scale).ok_or_else(|| "number grid has non-single-token value".to_string()),
            Self::Position => Err("position uses pointer readout".to_string()),
        }
    }
}

pub fn edge_grid(
    tokenizer: &Tokenizer,
    d1_numbers: bool,
) -> (Grid, i64) {
    let thousandths = Grid::new(0.0, 0.999, 1000);
    if d1_numbers && number_ids(tokenizer, &thousandths, 1000).is_some() {
        (thousandths, 1000)
    } else {
        (Grid::new(0.0, 1.0, 128), 1)
    }
}

pub fn grid_slot(
    tokenizer: &Tokenizer,
    grid: Grid,
    scale: i64,
    d1_numbers: bool,
) -> SlotSpec {
    if d1_numbers && number_ids(tokenizer, &grid, scale).is_some() {
        return SlotSpec::Number {
            grid,
            scale,
        };
    }
    if (2..=10).contains(&grid.size) {
        return SlotSpec::Score {
            levels: grid.size,
        };
    }
    let labels: Vec<String> = (0..grid.size).map(|i| format!("v{i}")).collect();
    SlotSpec::Choice {
        labels,
    }
}

pub fn question_slots(
    q: &D1Question,
    tokenizer: &Tokenizer,
    d1_numbers: bool,
) -> Vec<SlotSpec> {
    match q {
        D1Question::Noul {
            ..
        } => vec![SlotSpec::Noul],
        D1Question::Choice {
            criteria,
            multiple,
            ..
        } => {
            if *multiple {
                vec![SlotSpec::Noul; criteria.len()]
            } else {
                vec![SlotSpec::Choice {
                    labels: criteria.iter().map(|(k, _)| k.clone()).collect(),
                }]
            }
        },
        D1Question::Score {
            criteria,
            ..
        } => vec![SlotSpec::Score {
            levels: criteria.len(),
        }],
        D1Question::Read {
            ..
        } => vec![SlotSpec::Position, SlotSpec::Position],
        D1Question::Estimate {
            low,
            high,
            step,
            ..
        } => {
            let g = estimate_grid(*low, *high, *step);
            vec![grid_slot(tokenizer, g, 1, d1_numbers)]
        },
        D1Question::Box {
            ..
        } => {
            let (g, scale) = edge_grid(tokenizer, d1_numbers);
            let mut out = vec![SlotSpec::Noul];
            for _ in 0..4 {
                out.push(grid_slot(tokenizer, g.clone(), scale, d1_numbers));
            }
            out
        },
        D1Question::Boxes {
            max_boxes,
            ..
        } => {
            let blocks = max_boxes.unwrap_or(INSTANCES).min(MAX_INSTANCES);
            let (g, scale) = edge_grid(tokenizer, d1_numbers);
            let mut out = Vec::with_capacity(blocks * 5);
            for _ in 0..blocks {
                out.push(SlotSpec::Noul);
                for _ in 0..4 {
                    out.push(grid_slot(tokenizer, g.clone(), scale, d1_numbers));
                }
            }
            out
        },
    }
}

pub fn state_block(state: &serde_json::Value) -> String {
    if let Some(s) = state.as_str() {
        format!("{s}\n\n")
    } else {
        let pretty = serde_json::to_string_pretty(state).unwrap_or_else(|_| state.to_string());
        format!("{pretty}\n\n")
    }
}

pub fn state_text(state: &serde_json::Value) -> String {
    let b = state_block(state);
    b[..b.len() - 2].to_string()
}

pub fn question_header(state: Option<&serde_json::Value>) -> &'static str {
    if state.is_some() {
        "QUESTION:\n"
    } else {
        ""
    }
}

pub fn build_prefix_ids(
    plain_tokenizer: &Tokenizer,
    special: &SpecialTokens,
    state: Option<&serde_json::Value>,
    state_media_tokens: &[u32],
) -> Result<(Vec<u32>, Option<StatePositions>), String> {
    let mut ids = vec![special.bos, special.im_start];
    if state_media_tokens.is_empty() {
        if let Some(s) = state {
            let raw_text = state_text(s);
            let combined = format!("user\n{}\n{}", state_block(s), question_header(Some(s)));
            let user_len = "user\n".len();
            let state_end = user_len + raw_text.len();
            let enc = plain_tokenizer.encode(combined.as_str(), false).map_err(|e| e.to_string())?;
            let base_tok = ids.len();
            ids.extend_from_slice(enc.get_ids());
            let mut offsets = vec![None; base_tok];
            for &(a, b) in enc.get_offsets() {
                let mut span = None;
                let lo = a.max(user_len);
                let hi = b.min(state_end);
                if lo < hi && combined[a..lo].trim().is_empty() && combined[hi..b].trim().is_empty() {
                    let left = lo - user_len;
                    let right = hi - user_len;
                    if left < right && right <= raw_text.len() && !raw_text[left..right].trim().is_empty() {
                        span = Some((left, right));
                    }
                }
                offsets.push(span);
            }
            if offsets.is_empty() || offsets[0].is_some() {
                return Err("position 0 must be a template token for nothing readout".to_string());
            }
            return Ok((
                ids,
                Some(StatePositions {
                    text: raw_text,
                    offsets,
                }),
            ));
        } else {
            ids.extend(encode_no_special(plain_tokenizer, "user\n"));
            let offsets = vec![None; ids.len()];
            return Ok((
                ids,
                Some(StatePositions {
                    text: String::new(),
                    offsets,
                }),
            ));
        }
    }

    ids.extend(encode_no_special(plain_tokenizer, "user\n"));
    ids.extend_from_slice(state_media_tokens);
    if let Some(s) = state {
        let raw_text = state_text(s);
        let state_plus_header = format!("{}\n{}", state_block(s), question_header(Some(s)));
        let enc = plain_tokenizer
            .encode(state_plus_header.as_str(), false)
            .map_err(|e| e.to_string())?;
        let base_tok = ids.len();
        ids.extend_from_slice(enc.get_ids());
        let mut offsets = vec![None; base_tok];
        for &(a, b) in enc.get_offsets() {
            let mut span = None;
            let lo = a;
            let hi = b.min(raw_text.len());
            if lo < hi && state_plus_header[hi..b].trim().is_empty()
                && !raw_text[lo..hi].trim().is_empty() {
                    span = Some((lo, hi));
                }
            offsets.push(span);
        }
        if offsets.is_empty() || offsets[0].is_some() {
            return Err("position 0 must be a template token for nothing readout".to_string());
        }
        Ok((
            ids,
            Some(StatePositions {
                text: raw_text,
                offsets,
            }),
        ))
    } else {
        let offsets = vec![None; ids.len()];
        Ok((
            ids,
            Some(StatePositions {
                text: String::new(),
                offsets,
            }),
        ))
    }
}

fn option_line(
    code: &str,
    label: &str,
    desc: Option<&str>,
) -> String {
    let text = desc.map(|s| s.to_string()).unwrap_or_else(|| label.replace('_', " "));
    format!("{code} {text}")
}

fn edge_legend(
    tokenizer: &Tokenizer,
    d1_numbers: bool,
) -> &'static str {
    let (_, scale) = edge_grid(tokenizer, d1_numbers);
    if scale == 1000 {
        "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, left and right from its left edge."
    } else {
        "Edges are codes from 0 (top or left) to 1 (bottom or right). Values are equally spaced."
    }
}

pub fn question_body_tail(
    tokenizer: &Tokenizer,
    q: &D1Question,
    d1_codes: u32,
    d1_numbers: bool,
    d1_prompt: u32,
) -> Result<String, String> {
    match q {
        D1Question::Box {
            ..
        } => {
            let slots = question_slots(q, tokenizer, d1_numbers);
            let one = if matches!(slots.last(), Some(SlotSpec::Number { .. })) {
                "one number"
            } else {
                "one code"
            };
            let leg = edge_legend(tokenizer, d1_numbers);
            Ok(format!(
                "\n\nPresence is yes or no. {leg} Reply with yes or no at present, then {one} at top, bottom, left and right."
            ))
        },
        D1Question::Boxes {
            ..
        } => {
            let most = if d1_prompt == 1 {
                format!("{INSTANCES} at most, ")
            } else {
                String::new()
            };
            let leg = edge_legend(tokenizer, d1_numbers);
            Ok(format!(
                "\n\nEvery one of them, {most}in reading order (left edge, then top). Present k is yes when there are at least k. {leg} Reply with yes or no at each present, then its top, bottom, left and right."
            ))
        },
        D1Question::Estimate {
            low,
            high,
            step,
            unit,
            ..
        } => {
            let g = estimate_grid(*low, *high, *step);
            let slots = question_slots(q, tokenizer, d1_numbers);
            let u = unit.as_deref().map(|u| format!(" {u}")).unwrap_or_default();
            if matches!(slots.first(), Some(SlotSpec::Number { .. })) {
                Ok(format!(
                    "\n\nEstimate from {} to {}{u}, in steps of {}. Reply with the number only.",
                    format_py_g(g.low),
                    format_py_g(g.high),
                    format_py_g(g.step())
                ))
            } else {
                let codes = match slots.first() {
                    Some(SlotSpec::Score {
                        levels,
                    }) => (0..*levels).map(|i| i.to_string()).collect::<Vec<_>>(),
                    Some(SlotSpec::Choice {
                        labels,
                    }) => aliases(tokenizer, labels, d1_codes)?.into_iter().map(|(c, _)| c).collect(),
                    _ => unreachable!(),
                };
                Ok(format!(
                    "\n\nEstimate from {} to {}{u} on {} equally spaced values, step {}: {} is {}, {} is {}. Reply with the code only.",
                    format_py_g(g.low),
                    format_py_g(g.high),
                    g.size,
                    format_py_g(g.step()),
                    codes.first().map(String::as_str).unwrap_or(""),
                    format_py_g(g.low),
                    codes.last().map(String::as_str).unwrap_or(""),
                    format_py_g(g.high),
                ))
            }
        },
        D1Question::Read {
            kind,
            unit,
            ..
        } => {
            let u = if kind == "number" {
                unit.as_deref().filter(|s| !s.is_empty()).map(|s| format!(" (unit: {s})")).unwrap_or_default()
            } else {
                String::new()
            };
            Ok(format!("\n\nRead the {kind}{u} from the state, or nothing if it is absent."))
        },
        D1Question::Choice {
            criteria,
            multiple,
            ..
        } => {
            let labels: Vec<String> = criteria.iter().map(|(k, _)| k.clone()).collect();
            if *multiple {
                let lines: Vec<String> = criteria
                    .iter()
                    .map(|(name, desc)| {
                        let desc = desc.as_deref().unwrap_or(name.as_str());
                        let j = serde_json::to_string(name).unwrap_or_else(|_| format!("\"{name}\""));
                        format!("{j}: {desc}")
                    })
                    .collect();
                Ok(format!("\n\nOptions:\n{}\n\nReply with yes or no for each option.", lines.join("\n")))
            } else {
                let al = aliases(tokenizer, &labels, d1_codes)?;
                let lines: Vec<String> = al
                    .iter()
                    .zip(criteria.iter())
                    .map(|((code, _), (lab, desc))| option_line(code, lab, desc.as_deref()))
                    .collect();
                Ok(format!("\n\nOptions:\n{}\n\nReply with the option code only.", lines.join("\n")))
            }
        },
        D1Question::Noul {
            criteria,
            ..
        } => {
            let extra = match criteria {
                Some(c) => format!(
                    "\nYes: {}\nNo: {}",
                    c.get("true").map(String::as_str).unwrap_or("None"),
                    c.get("false").map(String::as_str).unwrap_or("None")
                ),
                None => String::new(),
            };
            Ok(format!("{extra}\n\nReply with yes or no only."))
        },
        D1Question::Score {
            criteria,
            ..
        } => {
            let legend: Vec<String> = criteria.iter().enumerate().map(|(i, name)| format!("{i} {name}")).collect();
            Ok(format!(
                "\n\n{}\n\nReply with a single digit 0-{} only.",
                legend.join("\n"),
                criteria.len() - 1
            ))
        },
    }
}

fn marked_assistant_tail(
    tokenizer: &Tokenizer,
    plain_tokenizer: &Tokenizer,
    names: &[String],
    lead: &str,
    after: &str,
) -> Result<(Vec<u32>, Vec<isize>), String> {
    let prefix_ids = encode_no_special(tokenizer, &format!("assistant\n{lead}"));
    let mut text = String::new();
    let mut ends = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            text.push('\n');
        }
        text.push_str(name);
        text.push(':');
        ends.push(text.len());
    }
    if !after.is_empty() {
        let test_enc = encode_no_special(plain_tokenizer, &format!("{text}{after}"));
        if let Some(&last) = test_enc.last() {
            let decoded = plain_tokenizer.decode(&[last], false).unwrap_or_default();
            if decoded.trim_end().ends_with(':') {
                text.push_str(after);
            }
        }
    }
    let enc = plain_tokenizer.encode(text.as_str(), false).map_err(|e| e.to_string())?;
    let ids = enc.get_ids();
    let offsets = enc.get_offsets();
    let mut slots = Vec::with_capacity(ends.len());
    for chars in ends {
        let mut found = None;
        for (i, &(start, stop)) in offsets.iter().enumerate() {
            if start < chars && chars - 1 < stop {
                if found.is_some() {
                    return Err("ambiguous slot marker token".to_string());
                }
                found = Some((i, stop));
            }
        }
        let (end, stop) = found.ok_or_else(|| "slot marker has no exact colon token".to_string())?;
        if !text[chars..stop].trim().is_empty() {
            return Err("slot marker colon shares token with next marker".to_string());
        }
        let decoded = plain_tokenizer.decode(&[ids[end]], false).unwrap_or_default();
        if !decoded.trim_end().ends_with(':') {
            return Err("slot marker token does not end with colon".to_string());
        }
        slots.push(end as isize - ids.len() as isize + 1);
    }
    let mut out = Vec::with_capacity(prefix_ids.len() + ids.len());
    out.extend_from_slice(&prefix_ids);
    out.extend_from_slice(ids);
    Ok((out, slots))
}

pub fn suffix_layout_parts(
    tokenizer: &Tokenizer,
    plain_tokenizer: &Tokenizer,
    q: &D1Question,
    lead: &str,
    d1_codes: u32,
    d1_numbers: bool,
    d1_prompt: u32,
) -> Result<(String, Vec<u32>, Vec<isize>), String> {
    let tail = question_body_tail(plain_tokenizer, q, d1_codes, d1_numbers, d1_prompt)?;
    let (assistant_ids, slots) = match q {
        D1Question::Read {
            ..
        } => marked_assistant_tail(tokenizer, plain_tokenizer, &["start".to_string(), "end".to_string()], lead, "")?,
        D1Question::Box {
            ..
        } => {
            let names: Vec<String> = std::iter::once("present")
                .chain(EDGE_NAMES.iter().copied())
                .map(str::to_string)
                .collect();
            marked_assistant_tail(tokenizer, plain_tokenizer, &names, lead, "")?
        },
        D1Question::Boxes {
            max_boxes,
            ..
        } => {
            let blocks = max_boxes.unwrap_or(INSTANCES).min(MAX_INSTANCES);
            let mut names = Vec::with_capacity(blocks * 5);
            for k in 1..=blocks {
                names.push(format!("present {k}"));
                for edge in EDGE_NAMES {
                    names.push(format!("{edge} {k}"));
                }
            }
            let after = if blocks < INSTANCES || d1_prompt > 1 { "\n" } else { "" };
            marked_assistant_tail(tokenizer, plain_tokenizer, &names, lead, after)?
        },
        D1Question::Choice {
            criteria,
            multiple: true,
            ..
        } => {
            let names: Vec<String> = criteria
                .iter()
                .map(|(k, _)| serde_json::to_string(k).unwrap_or_else(|_| format!("\"{k}\"")))
                .collect();
            marked_assistant_tail(tokenizer, plain_tokenizer, &names, lead, "")?
        },
        _ => (encode_no_special(tokenizer, &format!("assistant\n{lead}")), vec![0]),
    };
    Ok((tail, assistant_ids, slots))
}

pub fn pack_set_ids(
    plain_tokenizer: &Tokenizer,
    special: &SpecialTokens,
    prefix_ids: &[u32],
    group_qs: &[&D1Question],
    group_tails: &[String],
    group_assistant_ids: &[Vec<u32>],
    group_offsets: &[Vec<isize>],
    group_spoken_tokens: &[Vec<u32>],
    header: &str,
) -> Result<(Vec<u32>, usize, Vec<i32>, Vec<Vec<usize>>), String> {
    if group_qs.is_empty() {
        return Err("question set cannot be empty".to_string());
    }
    let newline_ids = encode_no_special(plain_tokenizer, "\n");
    let state_len = prefix_ids.len();
    let mut ids = prefix_ids.to_vec();
    let mut question = vec![-1i32; state_len];
    let mut ends = Vec::with_capacity(group_qs.len());

    for (idx, q) in group_qs.iter().enumerate() {
        let q_start = ids.len();
        let q_prefix = if idx > 0 { header } else { "" };
        match q.instructions().text() {
            Some(said) => {
                let user_body = format!("{q_prefix}{said}{}", group_tails[idx]);
                ids.extend(encode_no_special(plain_tokenizer, &user_body));
            },
            None => {
                if !q_prefix.is_empty() {
                    ids.extend(encode_no_special(plain_tokenizer, q_prefix));
                }
                ids.extend_from_slice(&group_spoken_tokens[idx]);
                ids.extend(encode_no_special(plain_tokenizer, &group_tails[idx]));
            },
        }
        ids.push(special.im_end);
        ids.extend_from_slice(&newline_ids);
        ids.push(special.im_start);
        ids.extend_from_slice(&group_assistant_ids[idx]);

        let q_len = ids.len() - q_start;
        let q_id = (idx + 1) as i32;
        question.resize(ids.len(), q_id);
        let end = ids.len() - 1;
        ends.push(
            group_offsets[idx]
                .iter()
                .map(|&o| {
                    let pos = end as isize + o;
                    assert!(pos >= state_len as isize && (q_len as isize + o) >= 0);
                    pos as usize
                })
                .collect(),
        );
    }
    Ok((ids, state_len, question, ends))
}

pub fn probabilities(
    slot: &SlotSpec,
    scores: &[f64],
    calibration: Option<&Calibration>,
) -> Vec<f64> {
    let (kind, k) = slot.bucket_kind();
    let t = calibration.map(|c| c.temperature(&question_bucket(kind, k))).unwrap_or(1.0);
    let scaled: Vec<f64> = if (t - 1.0).abs() < 1e-12 {
        scores.to_vec()
    } else {
        scores.iter().map(|&s| s / t).collect()
    };
    let m = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&s| (s - m).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

pub struct StatePositions {
    pub text: String,
    pub offsets: Vec<Option<(usize, usize)>>,
}

impl StatePositions {
    pub fn valid_len(&self) -> usize {
        self.offsets
            .iter()
            .enumerate()
            .rev()
            .find(|&(i, o)| i == 0 || o.is_some())
            .map(|(i, _)| i + 1)
            .unwrap_or(1)
    }

    pub fn is_valid(
        &self,
        idx: usize,
    ) -> bool {
        idx == 0 || self.offsets.get(idx).is_some_and(Option::is_some)
    }

    pub fn read_span(
        &self,
        first: usize,
        last: usize,
    ) -> Option<(String, [usize; 2])> {
        if first == 0 && last == 0 {
            return None;
        }
        let a = self.offsets.get(first).copied().flatten()?;
        let b = self.offsets.get(last).copied().flatten()?;
        let mut s = a.0;
        let mut e = b.1;
        while s < e && self.text.as_bytes()[s].is_ascii_whitespace() {
            s += 1;
        }
        while e > s && self.text.as_bytes()[e - 1].is_ascii_whitespace() {
            e -= 1;
        }
        Some((self.text[s..e].to_string(), [s, e]))
    }
}

pub fn pointer_span(
    start_hidden: &[f32],
    end_hidden: &[f32],
    state_hidden: &[f32], // [T, D]
    dim: usize,
    positions: &StatePositions,
    max_tokens: usize,
) -> (usize, usize, f64) {
    let t = positions.valid_len();
    let dot_row = |q: &[f32]| -> Vec<f32> {
        (0..t)
            .map(|i| {
                if !positions.is_valid(i) {
                    0.0
                } else {
                    let row = &state_hidden[i * dim..(i + 1) * dim];
                    let dot: f64 = q.iter().zip(row.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
                    dot as f32
                }
            })
            .collect()
    };
    let start_dots = dot_row(start_hidden);
    let end_dots = dot_row(end_hidden);
    pointer_span_from_logits(&start_dots, &end_dots, dim, positions, max_tokens)
}

pub fn pointer_span_from_logits(
    start_dots: &[f32],
    end_dots: &[f32],
    dim: usize,
    positions: &StatePositions,
    max_tokens: usize,
) -> (usize, usize, f64) {
    let t = positions.valid_len();
    let scale = 1.0 / (dim as f64).sqrt();
    let score_row = |dots: &[f32]| -> Vec<f64> {
        let mut logits = Vec::with_capacity(t);
        for i in 0..t {
            if !positions.is_valid(i) {
                logits.push(f64::NEG_INFINITY);
            } else {
                logits.push((dots[i] as f64) * scale);
            }
        }
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + logits.iter().map(|&x| (x - m).exp()).sum::<f64>().ln();
        logits.into_iter().map(|x| x - lse).collect()
    };

    let log_start = score_row(start_dots);
    let log_end = score_row(end_dots);
    let found = -log_start[0].exp_m1();
    if t <= 1 || found < 0.5 {
        return (0, 0, found);
    }
    let cap = max_tokens.min(t - 1);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_first = 1usize;
    let mut best_last = 1usize;
    for first in 1..t {
        if !log_start[first].is_finite() {
            continue;
        }
        let limit = (first + cap).min(t);
        for last in first..limit {
            if !log_end[last].is_finite() {
                continue;
            }
            let score = log_start[first] + log_end[last];
            if score > best_score {
                best_score = score;
                best_first = first;
                best_last = last;
            }
        }
    }
    (best_first, best_last, found)
}

fn ordered_edges(
    a: f64,
    b: f64,
    grid: &Grid,
) -> (f64, f64) {
    if a < b {
        return (a, b);
    }
    let half = grid.step() / 2.0;
    let middle = ((a + b) / 2.0).clamp(half, 1.0 - half);
    (middle - half, middle + half)
}

fn rectangle(
    edges: &[f64; 4],
    grid: &Grid,
) -> [f64; 4] {
    let (top, bottom) = ordered_edges(edges[0], edges[1], grid);
    let (left, right) = ordered_edges(edges[2], edges[3], grid);
    [top, bottom, left, right]
}

fn box_iou(
    a: &[f64; 4],
    b: &[f64; 4],
) -> f64 {
    let h = (a[1].min(b[1]) - a[0].max(b[0])).max(0.0);
    let w = (a[3].min(b[3]) - a[2].max(b[2])).max(0.0);
    let inter = h * w;
    let area_a = (a[1] - a[0]) * (a[3] - a[2]);
    let area_b = (b[1] - b[0]) * (b[3] - b[2]);
    inter / (area_a + area_b - inter)
}

fn radical_inverse(
    mut i: usize,
    base: usize,
) -> f64 {
    let mut value = 0.0;
    let mut scale = 1.0;
    while i > 0 {
        let digit = i % base;
        i /= base;
        scale /= base as f64;
        value += digit as f64 * scale;
    }
    value
}

pub fn box_confidence(
    edge_probs: &[Vec<f64>],
    rect: &[f64; 4],
) -> f64 {
    let grid = if edge_probs[0].len() == 1000 {
        Grid::new(0.0, 0.999, 1000)
    } else {
        Grid::new(0.0, 1.0, 128)
    };
    let values = grid.values();
    let cdfs: Vec<Vec<f64>> = edge_probs
        .iter()
        .map(|row| {
            let total: f64 = row.iter().sum::<f64>().max(1e-12);
            let mut acc = 0.0;
            row.iter()
                .map(|&p| {
                    acc += p / total;
                    acc
                })
                .collect()
        })
        .collect();
    let mut hits = 0usize;
    for i in 1..=DRAWS {
        let bases = [2, 3, 5, 7];
        let mut draw = [0.0; 4];
        for (k, &base) in bases.iter().enumerate() {
            let u = radical_inverse(i, base);
            let idx = cdfs[k].partition_point(|&c| c < u).min(grid.size - 1);
            draw[k] = values[idx];
        }
        let r = rectangle(&draw, &grid);
        if box_iou(&r, rect) >= 0.5 {
            hits += 1;
        }
    }
    hits as f64 / DRAWS as f64
}

pub fn box_medians(
    edge_probs: &[Vec<f64>],
    edge_mode: &str,
) -> [f64; 4] {
    let grid = if edge_probs[0].len() == 1000 {
        Grid::new(0.0, 0.999, 1000)
    } else {
        Grid::new(0.0, 1.0, 128)
    };
    let mut raw = [0.0; 4];
    for i in 0..4 {
        raw[i] = if edge_mode == "interp" {
            grid.crossing(&edge_probs[i])
        } else {
            grid.median(&edge_probs[i])
        };
    }
    rectangle(&raw, &grid)
}
