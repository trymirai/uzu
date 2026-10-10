use std::{
    fs::{File, create_dir_all},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::mpsc,
    time::Instant,
};

use anyhow::{Error, Result, anyhow};
use uzu_engine::{
    backends::{BackendSelection, common::Backend, select_backend},
    engine::{Engine, language_model::stream::SamplingMethod},
};

const MAX_SUFFIX_LENGTH: usize = 1024;

struct CollectGpuTimestamps {
    model_path: PathBuf,
    output_dir: PathBuf,
    prefix_step: usize,
}

impl BackendSelection for CollectGpuTimestamps {
    type Output = ();
    type Error = Error;

    fn select<B: Backend>(self) -> Result<()> {
        let engine = Engine::<B>::new().map_err(|error| anyhow!("{error}"))?;
        let model = engine.load_language_model(&self.model_path).map_err(|error| anyhow!("{error}"))?;
        let context_length =
            model.recommended_context_length().ok_or_else(|| anyhow!("Model has no context length limit"))?;
        let vocab_size = model.tokenizer().get_vocab_size(true) as u64;
        let mut state = model.create_empty_state(Some(context_length), 0).map_err(|error| anyhow!("{error}"))?;
        create_dir_all(&self.output_dir)?;
        for prefix_length in (0..=context_length as usize - MAX_SUFFIX_LENGTH).step_by(self.prefix_step) {
            let started = Instant::now();
            let output_path = self.output_dir.join(format!("prefix_{prefix_length}.csv"));
            let mut file = BufWriter::new(File::create(&output_path)?);
            writeln!(file, "suffix,name,start_us,end_us")?;
            for suffix_length in 1..=MAX_SUFFIX_LENGTH {
                state.set_context_length(prefix_length as u32);
                let (sender, receiver) = mpsc::channel();
                let mut options = model.default_stream_options();
                options.sampling_method = SamplingMethod::Greedy;
                options.timestamps = Some(sender);
                let suffix = (prefix_length..prefix_length + suffix_length)
                    .map(|position| position as u64 % vocab_size)
                    .collect::<Vec<_>>();
                drop(model.stream(&suffix, &mut state, options).map_err(|error| anyhow!("{error}"))?);
                for spans in receiver.iter() {
                    let Some(origin) = spans.first().map(|span| span.start) else {
                        continue;
                    };
                    for span in &spans {
                        writeln!(
                            file,
                            "{suffix_length},\"{}\",{:.3},{:.3}",
                            span.name.replace('"', "\"\""),
                            span.start.duration_since(origin).as_secs_f64() * 1e6,
                            span.end.duration_since(origin).as_secs_f64() * 1e6,
                        )?;
                    }
                }
            }
            file.flush()?;
            println!("Wrote {} in {:.1?}", output_path.display(), started.elapsed());
        }
        Ok(())
    }
}

pub fn run(
    model_path: PathBuf,
    output_dir: PathBuf,
    prefix_step: usize,
) -> Result<()> {
    select_backend(
        CollectGpuTimestamps {
            model_path,
            output_dir,
            prefix_step,
        },
        anyhow!("Unable to open any backend"),
    )
}
