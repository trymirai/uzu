//! Records an activation trace of a single forward pass, in lalamo's layout.
//!
//! ```text
//! cargo run -p backend-uzu --example trace -- \
//!   --model <model dir> --message "Hello, how are you?" --output uzu-trace.safetensors
//!
//! lalamo trace <model dir> --input-trace-path uzu-trace.safetensors \
//!   --output-path lalamo-trace.safetensors
//! lalamo compare-traces lalamo-trace.safetensors uzu-trace.safetensors
//! ```
//!
//! Feeding uzu's trace back as `--input-trace-path` makes lalamo replay uzu's own
//! token ids instead of re-tokenizing, so template drift shows up as a metadata
//! difference rather than a numeric mismatch.

use std::{collections::HashMap, error::Error, fs::File, io::BufReader, path::PathBuf, process::ExitCode};

use backend_uzu::engine::Engine;
use hanashi::chat::hanashi::renderer::{
    RAISE_EXCEPTION_FUNCTION_NAME, STRFTIME_NOW_FUNCTION_NAME, TEMPLATE_NAME, TOJSON_FILTER_NAME, raise_exception,
    strftime_now, to_json,
};
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

#[cfg(backend = "metal")]
type B = backend_uzu::backends::metal::Metal;
#[cfg(all(backend = "cpu", not(backend = "metal")))]
type B = backend_uzu::backends::cpu::Cpu;

struct Args {
    model: PathBuf,
    message: String,
    output: PathBuf,
    classifier: bool,
}

const USAGE: &str = "\
usage: trace --model <dir> --message <text> --output <file> [--classifier]

  --model <dir>       Model directory holding config.json, tokenizer.json and model.safetensors
  --message <text>    User message to run the forward pass on
  --output <file>     Where to write the trace (safetensors)
  --classifier        Trace a classifier model instead of a language model";

fn parse_args() -> Result<Args, String> {
    let (mut model, mut message, mut output) = (None, None, None);
    let mut classifier = false;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().ok_or_else(|| format!("{argument} needs a value"));
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--message" => message = Some(value()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--classifier" => classifier = true,
            "-h" | "--help" => return Err(USAGE.to_owned()),
            other => return Err(format!("unexpected argument {other}\n\n{USAGE}")),
        }
    }

    Ok(Args {
        model: model.ok_or_else(|| format!("--model is required\n\n{USAGE}"))?,
        message: message.ok_or_else(|| format!("--message is required\n\n{USAGE}"))?,
        output: output.ok_or_else(|| format!("--output is required\n\n{USAGE}"))?,
        classifier,
    })
}

// Mirrors lalamo's ChatCodec.render_request so both sides tokenize the same text.
fn render_request(
    codec_config: &Value,
    message: &str,
) -> Result<(String, Value), Box<dyn Error>> {
    let template = codec_config["prompt_template"].as_str().ok_or("config.json has no prompt_template")?;
    let system_role_name = codec_config["system_role_name"].as_str().unwrap_or("system");
    let user_role_name = codec_config["user_role_name"].as_str().unwrap_or("user");
    let bos_token = codec_config["bos_token"].as_str();
    let eos_token = codec_config["eos_token"].as_str();

    let mut messages = Vec::new();
    if let Some(default_system_prompt) = codec_config["default_system_prompt"].as_str() {
        messages.push(json!({ "role": system_role_name, "content": default_system_prompt }));
    }
    messages.push(json!({ "role": user_role_name, "content": message }));

    let request = json!({
        "add_generation_prompt": true,
        "messages": messages,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "enable_thinking": true,
    });

    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_function(STRFTIME_NOW_FUNCTION_NAME, strftime_now);
    environment.add_function(RAISE_EXCEPTION_FUNCTION_NAME, raise_exception);
    environment.add_filter(TOJSON_FILTER_NAME, to_json);
    environment.add_template(TEMPLATE_NAME, template)?;

    let rendered = environment.get_template(TEMPLATE_NAME)?.render(context!(
        messages => messages,
        add_generation_prompt => true,
        bos_token => bos_token,
        eos_token => eos_token,
        enable_thinking => true,
    ))?;

    Ok((rendered, request))
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let config: Value = serde_json::from_reader(BufReader::new(File::open(args.model.join("config.json"))?))?;
    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|error| format!("failed to load tokenizer: {error}"))?;

    // The token codec sits at the top level of config.json, not inside the model config.
    let codec_config = config.get("token_codec_config").ok_or("config.json has no token_codec_config")?;
    let (rendered_request, request) = render_request(codec_config, &args.message)?;

    // add_special_tokens = false: the template already emits them, as on lalamo's side.
    let encoding = tokenizer
        .encode(rendered_request.as_str(), false)
        .map_err(|error| format!("failed to tokenize prompt: {error}"))?;
    let token_ids = encoding.get_ids().iter().map(|token_id| *token_id as u64).collect::<Vec<u64>>();
    if token_ids.is_empty() {
        return Err("prompt tokenized to zero tokens".into());
    }
    println!("Prompt tokenized to {} tokens", token_ids.len());

    let engine = Engine::<B>::new()?;
    let recorder = if args.classifier {
        engine.load_classifier_model(&args.model)?.record_trace(&token_ids)?
    } else {
        engine.load_language_model(&args.model)?.record_trace(&token_ids)?
    };

    let metadata = HashMap::from([
        ("add_special_tokens".to_owned(), "false".to_owned()),
        ("prompt_template".to_owned(), codec_config["prompt_template"].as_str().unwrap_or_default().to_owned()),
        ("rendered_request".to_owned(), rendered_request),
        ("request".to_owned(), serde_json::to_string(&request)?),
        ("tokens".to_owned(), serde_json::to_string(encoding.get_tokens())?),
    ]);

    if let Some(parent) = args.output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    recorder.write(&args.output, Some(metadata))?;

    println!("Recorded {} arrays to {}", recorder.len(), args.output.display());

    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        },
    };

    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        },
    }
}
