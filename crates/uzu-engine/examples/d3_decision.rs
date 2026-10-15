use std::{
    error::Error,
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Instant,
};

use serde::Deserialize;
use serde_json::{Value, json};
use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::{Engine, d3_decision_model::D3DecisionInferenceError},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    token_ids: Vec<u32>,
    #[serde(default)]
    return_hidden: bool,
}

fn write_json(
    output: &mut impl Write,
    value: &Value,
) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(&mut *output, value)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_path = PathBuf::from(args.next().ok_or("usage: d3_decision MODEL_DIR")?);
    if args.next().is_some() {
        return Err("usage: d3_decision MODEL_DIR".into());
    }
    struct Run {
        model_path: PathBuf,
    }
    impl BackendSelection for Run {
        type Output = ();
        type Error = Box<dyn Error>;

        fn select<B: Backend>(self) -> Result<(), Self::Error> {
            if B::NAME != "metal" {
                return Err("the d3_decision example requires the Metal backend".into());
            }
            run::<B>(self.model_path)
        }
    }
    select_backend(
        Run {
            model_path,
        },
        "Metal backend is unavailable".into(),
    )
}

fn run<B: Backend>(model_path: PathBuf) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let engine = Engine::<B>::new().map_err(|error| error.to_string())?;
    let model = engine.load_d3_decision_model(&model_path)?;
    let stdout = io::stdout();
    let mut output = stdout.lock();
    write_json(
        &mut output,
        &json!({
            "type": "ready", "backend": B::NAME, "device": model.device_name(),
            "load_seconds": start.elapsed().as_secs_f64(),
            "model_dim": model.model_dim(), "num_codes": model.num_codes(), "vocab_size": model.vocab_size(),
        }),
    )?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let request = match serde_json::from_str::<Request>(&line) {
            Ok(request) => request,
            Err(error) => {
                let id = serde_json::from_str::<Value>(&line)
                    .ok()
                    .and_then(|value| value.get("id").and_then(Value::as_str).map(str::to_owned));
                write_json(
                    &mut output,
                    &json!({"id": id, "error": {"code": "invalid_request", "message": error.to_string()}}),
                )?;
                continue;
            },
        };
        match model.infer(&request.token_ids, request.return_hidden) {
            Ok(result) => {
                let mut response = json!({
                    "id": request.id, "logits": result.logits, "forward_ms": result.forward_ms,
                    "input_tokens": result.input_tokens, "peak_memory_bytes": result.peak_memory_bytes,
                });
                if let Some(hidden) = result.hidden {
                    response["hidden"] = json!(hidden);
                }
                write_json(&mut output, &response)?;
            },
            Err(error) => {
                let code = match &error {
                    D3DecisionInferenceError::Input(_) => "invalid_input",
                    D3DecisionInferenceError::Backend(_) | D3DecisionInferenceError::Encode(_) => "backend_error",
                    D3DecisionInferenceError::NonfiniteOutput => "nonfinite_output",
                };
                write_json(
                    &mut output,
                    &json!({"id": request.id, "error": {"code": code, "message": error.to_string()}}),
                )?;
            },
        }
    }
    Ok(())
}
