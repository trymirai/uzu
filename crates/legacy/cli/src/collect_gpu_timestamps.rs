use std::{
    fs::{File, create_dir_all},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::mpsc,
    time::Instant,
};

use anyhow::{Error, Result, anyhow};
use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::{Engine, language_model::stream::SamplingMethod},
};

const PREFIX_LENGTHS: [usize; 17] =
    [0, 256, 512, 1023, 1024, 2048, 4096, 5120, 8192, 16384, 30720, 32768, 51200, 61440, 65536, 131072, 196608];
const MAX_SUFFIX_LENGTH: usize = 1024;
const DENSE_SUFFIX_LENGTH: usize = 64;
const MAX_TREE_SIZE: usize = 32;
const REPEATS: usize = 5;

struct CollectGpuTimestamps {
    model_path: PathBuf,
    output_dir: PathBuf,
    spans: bool,
}

impl BackendSelection for CollectGpuTimestamps {
    type Output = ();
    type Error = Error;

    async fn select<B: Backend>(self) -> Result<()> {
        let engine = Engine::<B>::new().await.map_err(|error| anyhow!("{error}"))?;
        let model = engine.load_language_model(&self.model_path).map_err(|error| anyhow!("{error}"))?;
        let context_length =
            model.recommended_context_length().ok_or_else(|| anyhow!("Model has no context length limit"))? as usize;
        let vocab_size = model.tokenizer().get_vocab_size(true) as u64;
        let mut state = model.create_empty_state(Some(context_length as u32), 0).map_err(|error| anyhow!("{error}"))?;
        let mut prefix_lengths = PREFIX_LENGTHS
            .into_iter()
            .chain([context_length - MAX_SUFFIX_LENGTH])
            .filter(|prefix_length| prefix_length + MAX_SUFFIX_LENGTH <= context_length)
            .collect::<Vec<_>>();
        prefix_lengths.dedup();
        let flat_lengths: Vec<usize> =
            (1..=DENSE_SUFFIX_LENGTH).chain((DENSE_SUFFIX_LENGTH + 8..=MAX_SUFFIX_LENGTH).step_by(8)).collect();
        let mut modes = vec![("flat", flat_lengths)];
        if model.speculation_supported() {
            modes.push(("tree", (1..=MAX_TREE_SIZE).collect()));
        } else {
            println!("Skipping tree passes: the model can't verify draft trees");
        }
        create_dir_all(&self.output_dir)?;
        let mut totals = BufWriter::new(File::create(self.output_dir.join("totals.csv"))?);
        writeln!(totals, "mode,prefix,tokens,repeat,wall_us")?;
        for prefix_length in prefix_lengths {
            let started = Instant::now();
            for (mode, lengths) in &modes {
                let tree = *mode == "tree";
                if tree && prefix_length == 0 {
                    continue;
                }
                let mut spans_file = if self.spans {
                    let path = self.output_dir.join(format!("{mode}_prefix_{prefix_length}.csv"));
                    let mut file = BufWriter::new(File::create(path)?);
                    writeln!(file, "tokens,repeat,command_buffer,name,start_us,end_us")?;
                    Some(file)
                } else {
                    None
                };
                for &length in lengths {
                    for repeat in 0..=REPEATS {
                        state.set_context_length(prefix_length as u32);
                        let (sender, receiver) = mpsc::channel();
                        let mut options = model.default_stream_options();
                        options.sampling_method = SamplingMethod::Greedy;
                        options.timestamps = self.spans.then_some(sender);
                        let begin = Instant::now();
                        if tree {
                            options.draft_tree_size = Some(length);
                            let mut stream =
                                model.stream(&[], &mut state, options).map_err(|error| anyhow!("{error}"))?;
                            stream.generate().await.map_err(|error| anyhow!("{error}"))?;
                            stream.close().await.map_err(|error| anyhow!("{error}"))?;
                        } else {
                            let suffix = (prefix_length..prefix_length + length)
                                .map(|position| position as u64 % vocab_size)
                                .collect::<Vec<_>>();
                            let mut stream =
                                model.stream(&suffix, &mut state, options).map_err(|error| anyhow!("{error}"))?;
                            stream.close().await.map_err(|error| anyhow!("{error}"))?;
                        }
                        let wall = begin.elapsed();
                        writeln!(totals, "{mode},{prefix_length},{length},{repeat},{:.3}", wall.as_secs_f64() * 1e6)?;
                        let Some(file) = &mut spans_file else {
                            continue;
                        };
                        for (command_buffer, spans) in receiver.iter().enumerate() {
                            let Some(origin) = spans.first().map(|span| span.start) else {
                                continue;
                            };
                            for span in &spans {
                                writeln!(
                                    file,
                                    "{length},{repeat},{command_buffer},\"{}\",{:.3},{:.3}",
                                    span.name.replace('"', "\"\""),
                                    span.start.duration_since(origin).as_secs_f64() * 1e6,
                                    span.end.duration_since(origin).as_secs_f64() * 1e6,
                                )?;
                            }
                        }
                    }
                }
                if let Some(mut file) = spans_file {
                    file.flush()?;
                }
            }
            totals.flush()?;
            println!("Prefix {prefix_length} done in {:.1?}", started.elapsed());
        }
        Ok(())
    }
}

pub async fn run(
    model_path: PathBuf,
    output_dir: PathBuf,
    spans: bool,
) -> Result<()> {
    select_backend(
        CollectGpuTimestamps {
            model_path,
            output_dir,
            spans,
        },
        anyhow!("Unable to open any backend"),
    )
    .await
}
