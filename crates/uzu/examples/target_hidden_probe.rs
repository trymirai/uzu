use std::path::PathBuf;

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    parameter::{SamplingMethod, SamplingPolicy},
    types::Input,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("UZU_MODEL_PATH").map_err(|_| "UZU_MODEL_PATH environment variable is not set.")?;
    let prompt = std::env::args().nth(1).unwrap_or_else(|| String::from("Tell me one short fact about London."));

    let run_config = RunConfig::default().tokens_limit(1).sampling_policy(SamplingPolicy::Custom {
        value: SamplingMethod::Greedy,
    });

    let mut session = ChatSession::new(PathBuf::from(model_path), DecodingConfig::default())?;
    let (output, snapshot) = session.run_capture_target_hidden(Input::Text(prompt), run_config, 5)?;

    println!("text={}", output.text.original);
    println!("layer_count={}", snapshot.layers.len());
    for layer in &*snapshot.layers {
        println!(
            "layer={} active_rows={} model_dim={} outputs_len={}",
            layer.layer_index,
            layer.active_row_count,
            layer.model_dim,
            layer.outputs.len()
        );
    }

    Ok(())
}
