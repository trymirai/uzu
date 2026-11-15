//! `cli complete`: raw autoregressive completion against the resident uzu
//! server, bypassing every chat-template layer. Thin HTTP client — the server
//! owns the loaded weights, so repeated probes cost no load/unload cycles.
//!
//! If no server is listening, one is started idle (this binary re-spawns
//! itself with `server`), and `--model` triggers a load + ready-wait first.
//! The reply is the model's raw continuation of the prompt bytes: no chat
//! template, no reasoning/tool parsing. Use it to observe model behavior
//! directly (template contracts, tokenizer edges, sampling probes).

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::json;

#[derive(Parser)]
pub struct CompleteArgs {
    /// Prompt text, prefilled verbatim (no chat template). Omit to use --tokens.
    #[arg(value_name = "PROMPT")]
    pub prompt: Option<String>,
    /// Literal token ids to prefill instead of text (tokenizer bypass).
    #[arg(long, value_name = "IDS", value_delimiter = ',')]
    pub tokens: Option<Vec<u64>>,
    /// Ensure this model is resident before completing (models-dir directory
    /// name). Omit to use whatever is resident.
    #[arg(long, value_name = "MODEL")]
    pub model: Option<String>,
    #[arg(long, default_value_t = 8124)]
    pub port: u16,
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,
    #[arg(long)]
    pub temperature: Option<f64>,
    #[arg(long)]
    pub top_p: Option<f64>,
    #[arg(long)]
    pub top_k: Option<i64>,
    /// Sampling seed (matches the global --seed type; negatives allowed).
    #[arg(long, allow_negative_numbers = true)]
    pub seed: Option<i64>,
    /// Stop sequence; generation ends before it (repeatable).
    #[arg(long, value_name = "SEQ")]
    pub stop: Vec<String>,
    /// Stream raw text deltas as they decode.
    #[arg(long, short = 's')]
    pub stream: bool,
    /// Models dir used when starting a server (optional; defaults to HF cache & ~/.cache/mirai/models).
    #[arg(long, value_name = "DIR")]
    pub models_dir: Option<String>,
}

struct ServerHandle {
    port: u16,
}

impl ServerHandle {
    fn base(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn status(&self) -> Option<serde_json::Value> {
        let response =
            reqwest::blocking::get(format!("{}/v1/models/status", self.base())).ok()?;
        if !response.status().is_success() {
            return None;
        }
        response.json::<serde_json::Value>().ok()
    }

    fn load(&self, model: &str) -> Result<()> {
        let response = reqwest::blocking::Client::new()
            .post(format!("{}/v1/models/load", self.base()))
            .json(&json!({ "model": model }))
            .send()
            .context("load request failed")?;
        anyhow::ensure!(response.status().is_success(), "load rejected: HTTP {}", response.status());
        Ok(())
    }

    fn wait_ready(&self, want_model: Option<&str>, timeout_secs: u64) -> Result<String> {
        let started = std::time::Instant::now();
        loop {
            if let Some(status) = self.status() {
                let state = status["status"].as_str().unwrap_or_default().to_string();
                let resident = status["model"].as_str().unwrap_or_default().to_string();
                let model_matches = want_model.is_none_or(|want| resident == want);
                if state == "ready" && model_matches {
                    return Ok(resident);
                }
                if state == "failed" && want_model.is_none_or(|want| resident == want) {
                    let error = status["last_error"]["error"].as_str().unwrap_or("unknown error");
                    bail!("model '{resident}' failed to load: {error}");
                }
            }
            anyhow::ensure!(
                started.elapsed() < std::time::Duration::from_secs(timeout_secs),
                "server/model did not become ready within {timeout_secs}s"
            );
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }

    fn spawn_idle(&self, models_dir: Option<&str>) -> Result<()> {
        let exe = std::env::current_exe().context("cannot resolve current executable")?;
        let log_path = std::env::temp_dir().join(format!("uzu-server-{}.log", self.port));
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .context("cannot open server log")?;
        let port_str = self.port.to_string();
        let mut args = vec!["server", "--port", &port_str];
        if let Some(dir) = models_dir {
            args.extend(["--models-dir", dir]);
        }
        std::process::Command::new(exe)
            .args(&args)
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()
            .context("failed to spawn server")?;
        Ok(())
    }
}

pub async fn run_complete(args: CompleteArgs) -> Result<()> {
    if args.prompt.is_none() && args.tokens.is_none() {
        bail!("provide a PROMPT argument or --tokens");
    }
    let server = ServerHandle {
        port: args.port,
    };

    // Ensure a server is listening (idle boot takes ~2s).
    if server.status().is_none() {
        eprintln!("uzu: no server on :{} — starting one…", server.port);
        server.spawn_idle(args.models_dir.as_deref())?;
        let started = std::time::Instant::now();
        while server.status().is_none() {
            anyhow::ensure!(
                started.elapsed() < std::time::Duration::from_secs(30),
                "server did not start within 30s — see {}",
                std::env::temp_dir().join(format!("uzu-server-{}.log", server.port)).display()
            );
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
    }

    // Ensure the requested model is resident.
    if let Some(model) = &args.model {
        let status = server.status().context("server status unavailable")?;
        let resident = status["model"].as_str().unwrap_or_default();
        let state = status["status"].as_str().unwrap_or_default();
        let already = state == "ready" && resident == model;
        if !already {
            eprintln!("uzu: loading {model}…");
            server.load(model)?;
        }
        server.wait_ready(Some(model), 300)?;
    } else {
        // A model must be resident either way; wait through any in-flight load.
        server.wait_ready(None, 300)?;
    }

    // Fire the raw completion.
    let mut body = json!({
        "model": args.model.clone().unwrap_or_default(),
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    });
    if let Some(prompt) = &args.prompt {
        body["prompt"] = json!(prompt);
    }
    if let Some(tokens) = &args.tokens {
        body["raw_tokens"] = json!(tokens);
    }
    if let Some(temperature) = args.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = args.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(top_k) = args.top_k {
        body["top_k"] = json!(top_k);
    }
    if let Some(seed) = args.seed {
        body["seed"] = json!(seed);
    }
    if !args.stop.is_empty() {
        body["stop"] = json!(args.stop);
    }
    // The resident model name for the wire field; server validates it anyway.
    if body["model"].as_str().is_none_or(|m| m.is_empty()) {
        let status = server.status().context("server status unavailable")?;
        body["model"] = status["model"].clone();
    }

    let client = reqwest::blocking::Client::new();
    let response = client
        .post(format!("{}/v1/completions", server.base()))
        .json(&body)
        .send()
        .context("completion request failed")?;
    let status = response.status();
    let text = response.text().context("failed to read completion response")?;
    anyhow::ensure!(status.is_success(), "completion failed: HTTP {status} — {text}");

    if args.stream {
        for line in text.lines() {
            let Some(payload) = line.strip_prefix("data:").map(str::trim_start) else {
                continue;
            };
            if payload == "[DONE]" {
                break;
            }
            if let Ok(event) = serde_json::from_str::<serde_json::Value>(payload)
                && let Some(delta) = event["choices"][0]["text"].as_str()
            {
                print!("{delta}");
            }
        }
        println!();
    } else {
        let value: serde_json::Value = serde_json::from_str(&text).context("non-JSON completion response")?;
        let choice = &value["choices"][0];
        println!("{}", choice["text"].as_str().unwrap_or_default());
        let finish = choice["finish_reason"].as_str().unwrap_or_default();
        let prompt = value["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        let completion = value["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        eprintln!("[finish={finish} prompt={prompt} tok out={completion} tok]");
    }
    Ok(())
}
