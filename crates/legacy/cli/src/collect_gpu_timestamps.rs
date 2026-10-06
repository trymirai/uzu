use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::mpsc,
    time::Instant,
};

use anyhow::{Error, Result, anyhow};
use uzu_engine::{
    backends::{
        BackendSelection,
        common::{Backend, TimestampSampleEntry},
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
    type Output = Vec<Box<[(TimestampSampleEntry, Instant)]>>;
    type Error = Error;

    fn select<B: Backend>(self) -> Result<Vec<Box<[(TimestampSampleEntry, Instant)]>>> {
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
    writeln!(file, "command_buffer,name,kind,timestamp_us")?;
    for (command_buffer, timestamps) in command_buffers.iter().enumerate() {
        let Some(&(_, origin)) = timestamps.first() else {
            continue;
        };
        for (entry, timestamp) in timestamps {
            let (kind, name) = match entry {
                TimestampSampleEntry::Start(name) => ("start", name),
                TimestampSampleEntry::End(name) => ("end", name),
            };
            writeln!(
                file,
                "{command_buffer},\"{}\",{kind},{:.3}",
                name.replace('"', "\"\""),
                timestamp.duration_since(origin).as_secs_f64() * 1e6,
            )?;
        }
    }
    file.flush()?;
    println!(
        "Wrote {} timestamps from {} command buffers to {}",
        command_buffers.iter().map(|timestamps| timestamps.len()).sum::<usize>(),
        command_buffers.len(),
        output_path.display()
    );
    Ok(())
}
