use std::path::PathBuf;

use serde::Serialize;
use serde_json::Value;

use crate::model::{Coefficients, Work};

#[derive(Debug, Serialize)]
pub struct Prediction {
    pub chip: String,
    pub model_dim: u64,
    pub hidden_dim: u64,
    pub num_layers: u64,
    pub vocab_size: u64,
    pub weight_bytes: f64,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub decode_energy_per_token_j: f64,
    pub decode_avg_power_w: f64,
    pub decode_tokens_per_second: f64,
    pub prefill_energy_j: f64,
    pub time_to_first_token_s: f64,
    pub total_energy_j: f64,
}

pub fn run(args: &[String]) {
    let mut model_path: Option<String> = None;
    let mut coeffs_path: Option<String> = None;
    let mut out_path = String::from("keisoku-prediction.json");
    let mut prompt_tokens = 512u64;
    let mut output_tokens = 128u64;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model_path = iter.next().cloned(),
            "--coeffs" => coeffs_path = iter.next().cloned(),
            "--out" => out_path = iter.next().cloned().unwrap_or(out_path),
            "--prompt-tokens" => prompt_tokens = next_u64(&mut iter, "--prompt-tokens"),
            "--tokens" => output_tokens = next_u64(&mut iter, "--tokens"),
            other => {
                eprintln!("predict: unknown argument {other}");
                std::process::exit(2);
            },
        }
    }

    let model_path = model_path.unwrap_or_else(|| fail("predict requires --model <dir-or-config.json>"));
    let coeffs_path = coeffs_path.unwrap_or_else(|| fail("predict requires --coeffs <coeffs.json>"));

    let coefficients: Coefficients =
        serde_json::from_str(&std::fs::read_to_string(&coeffs_path).expect("failed to read coefficients"))
            .expect("failed to parse coefficients");
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(resolve_config(&model_path)).expect("read config"))
            .expect("failed to parse config.json");

    let decoder = resolve_decoder(&config);
    let transformer = &decoder["transformer_config"];
    let model_dim = number(transformer, "model_dim");
    let hidden_dim = number(transformer, "hidden_dim");
    let vocab_size = number(decoder, "vocab_size");
    let num_layers = transformer["layer_configs"].as_array().map(|layers| layers.len() as u64).unwrap_or(0);
    let weight_bytes = weight_bytes(&config["quantization"]);

    let weight_elements = num_layers as f64 * (4.0 * sq(model_dim) + 3.0 * model_dim as f64 * hidden_dim as f64)
        + vocab_size as f64 * model_dim as f64;
    let decode = Work {
        bytes: weight_elements * weight_bytes,
        flops: 2.0 * weight_elements,
    };

    let decode_energy = coefficients.energy_joules(decode);
    let decode_time = safe_div(decode.bytes, coefficients.peak_bandwidth_bytes_per_s);
    let prefill = Work {
        bytes: decode.bytes,
        flops: decode.flops * prompt_tokens as f64,
    };
    let prefill_energy = coefficients.energy_joules(prefill);
    let prefill_time = safe_div(prefill.flops, coefficients.peak_flop_rate_per_s);

    let prediction = Prediction {
        chip: coefficients.chip.clone(),
        model_dim,
        hidden_dim,
        num_layers,
        vocab_size,
        weight_bytes,
        prompt_tokens,
        output_tokens,
        decode_energy_per_token_j: decode_energy,
        decode_avg_power_w: safe_div(decode_energy, decode_time),
        decode_tokens_per_second: safe_div(1.0, decode_time),
        prefill_energy_j: prefill_energy,
        time_to_first_token_s: prefill_time,
        total_energy_j: prefill_energy + output_tokens as f64 * decode_energy,
    };

    let json = serde_json::to_string_pretty(&prediction).expect("failed to serialize prediction");
    std::fs::write(&out_path, json).expect("failed to write prediction");
    eprintln!(
        "keisoku: decode {:.3} W ({:.1} tok/s, {:.1} mJ/tok), TTFT {:.3} s, total {:.1} J -> {out_path}",
        prediction.decode_avg_power_w,
        prediction.decode_tokens_per_second,
        prediction.decode_energy_per_token_j * 1000.0,
        prediction.time_to_first_token_s,
        prediction.total_energy_j,
    );
}

fn resolve_config(path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_dir() {
        path.join("config.json")
    } else {
        path
    }
}

fn resolve_decoder(config: &Value) -> &Value {
    let nested = &config["model_config"]["model_config"];
    if nested.get("transformer_config").is_some() {
        return nested;
    }
    let top = &config["decoder_config"];
    if top.get("transformer_config").is_some() {
        return top;
    }
    fail("could not locate transformer_config in config.json (unknown schema)")
}

fn number(
    value: &Value,
    key: &str,
) -> u64 {
    value[key].as_u64().unwrap_or_else(|| fail(&format!("config missing numeric field {key}")) as u64)
}

fn weight_bytes(quantization: &Value) -> f64 {
    match quantization.as_str() {
        Some(name) => {
            let name = name.to_lowercase();
            if name.contains('4') {
                0.5
            } else if name.contains('8') {
                1.0
            } else {
                2.0
            }
        },
        None => 2.0,
    }
}

fn sq(value: u64) -> f64 {
    value as f64 * value as f64
}

fn safe_div(
    numerator: f64,
    denominator: f64,
) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn next_u64(
    iter: &mut std::slice::Iter<String>,
    flag: &str,
) -> u64 {
    iter.next().and_then(|value| value.parse().ok()).unwrap_or_else(|| fail(&format!("{flag} requires an integer")))
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}
