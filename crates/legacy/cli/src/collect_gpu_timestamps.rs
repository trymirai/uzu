use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::mpsc,
};

use anyhow::{Error, Result, anyhow};
use uzu_engine::{
    backends::{
        BackendSelection,
        common::{Backend, CommandBufferTimestamps},
        select_backend,
    },
    engine::{Engine, language_model::stream::SamplingMethod},
};

struct CollectGpuTimestamps {
    model_path: PathBuf,
    prompt: String,
    tokens: usize,
}

impl BackendSelection for CollectGpuTimestamps {
    type Output = Vec<CommandBufferTimestamps>;
    type Error = Error;

    fn select<B: Backend>(self) -> Result<Vec<CommandBufferTimestamps>> {
        let engine = Engine::<B>::new().map_err(|error| anyhow!("{error}"))?;
        let model = engine.load_language_model(&self.model_path).map_err(|error| anyhow!("{error}"))?;
        let input = model
            .tokenizer()
            .encode(self.prompt.as_str(), false)
            .map_err(Error::msg)?
            .get_ids()
            .iter()
            .map(|&token| u64::from(token))
            .collect::<Vec<_>>();
        let mut state =
            model.create_empty_state(model.recommended_context_length(), 0).map_err(|error| anyhow!("{error}"))?;
        let (sender, receiver) = mpsc::channel();
        let mut options = model.default_stream_options();
        options.sampling_method = SamplingMethod::Greedy;
        options.timestamps = Some(sender);
        for token in model.stream(&input, &mut state, options).map_err(|error| anyhow!("{error}"))?.take(self.tokens) {
            token.map_err(|error| anyhow!("{error}"))?;
        }
        Ok(receiver.iter().collect())
    }
}

pub fn run(
    model_path: PathBuf,
    output_path: PathBuf,
    prompt: String,
    tokens: usize,
) -> Result<()> {
    let command_buffers = select_backend(
        CollectGpuTimestamps {
            model_path,
            prompt,
            tokens,
        },
        anyhow!("Unable to open any backend"),
    )?;
    let mut file = BufWriter::new(File::create(&output_path)?);
    writeln!(file, "command_buffer,name,start_us,end_us")?;
    for (command_buffer, spans) in command_buffers.iter().enumerate() {
        let Some(origin) = spans.first().map(|span| span.start) else {
            continue;
        };
        for span in spans {
            writeln!(
                file,
                "{command_buffer},\"{}\",{:.3},{:.3}",
                span.name.replace('"', "\"\""),
                span.start.duration_since(origin).as_secs_f64() * 1e6,
                span.end.duration_since(origin).as_secs_f64() * 1e6,
            )?;
        }
    }
    file.flush()?;
    println!(
        "Wrote {} spans from {} command buffers to {}",
        command_buffers.iter().map(|spans| spans.len()).sum::<usize>(),
        command_buffers.len(),
        output_path.display()
    );
    Ok(())
}
