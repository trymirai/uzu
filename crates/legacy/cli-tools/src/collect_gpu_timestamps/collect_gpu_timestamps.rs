use std::{
    ffi::OsStr,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Error, Result, anyhow};
use serde_json::json;
use sysinfo::System;
use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::{
        Engine,
        language_model::stream::{SamplingMethod, Timings},
    },
};

use super::{cell::Cell, parquet_table::ParquetTable};

const PREFIX_LENGTHS: [usize; 16] =
    [256, 512, 1023, 1024, 2048, 4096, 5120, 8192, 16384, 30720, 32768, 51200, 61440, 65536, 131072, 196608];
const MAX_SUFFIX_LENGTH: usize = 1024;
const DENSE_SUFFIX_LENGTH: usize = 64;
const REPEATS: usize = 5;
const MODEL_FILES: [&str; 3] = ["config.json", "speculator/config.json", "speculator/shapes.json"];
const TOTALS_SCHEMA: &str = "message totals {
    REQUIRED INT64 prefix;
    REQUIRED INT64 tokens;
    REQUIRED INT64 repeat;
    REQUIRED INT64 stamped;
    REQUIRED DOUBLE elapsed_s;
    REQUIRED DOUBLE wall_us;
    REQUIRED DOUBLE gpu_us;
}";
const SPANS_SCHEMA: &str = "message spans {
    REQUIRED INT64 prefix;
    REQUIRED INT64 tokens;
    REQUIRED INT64 repeat;
    REQUIRED INT64 command_buffer;
    REQUIRED BYTE_ARRAY name (UTF8);
    REQUIRED DOUBLE start_us;
    REQUIRED DOUBLE end_us;
}";

struct CollectGpuTimestamps {
    model_path: PathBuf,
    output_dir: PathBuf,
    spans: bool,
}

impl CollectGpuTimestamps {
    fn write_metadata(
        &self,
        device: &str,
        context_length: usize,
        prefix_lengths: &[usize],
        suffix_lengths: &[usize],
    ) -> Result<()> {
        let system = System::new_all();
        let model_path = self.model_path.canonicalize()?;
        let run = json!({
            "uzu_version": uzu_engine::VERSION,
            "device": device,
            "os": System::long_os_version(),
            "cpu": system.cpus().first().map(|cpu| cpu.brand()),
            "memory_bytes": system.total_memory(),
            "model_path": model_path,
            "model": model_path.parent().and_then(Path::file_name).map(OsStr::to_string_lossy),
            "revision": model_path.file_name().map(OsStr::to_string_lossy),
            "context_length": context_length,
            "prefix_lengths": prefix_lengths,
            "suffix_lengths": suffix_lengths,
            "repeats": REPEATS,
            "spans": self.spans,
            "started_unix_s": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        });
        serde_json::to_writer_pretty(File::create(self.output_dir.join("run.json"))?, &run)?;
        for file in MODEL_FILES {
            let source = model_path.join(file);
            if !source.exists() {
                continue;
            }
            let target = self.output_dir.join("model").join(file);
            fs::create_dir_all(target.parent().unwrap())?;
            fs::copy(source, target)?;
        }
        let mut weights = File::open(model_path.join("model.safetensors"))?;
        let mut length = [0; 8];
        weights.read_exact(&mut length)?;
        let mut header = vec![0; u64::from_le_bytes(length) as usize];
        weights.read_exact(&mut header)?;
        fs::write(self.output_dir.join("model/weights.json"), header)?;
        Ok(())
    }
}

impl BackendSelection for CollectGpuTimestamps {
    type Output = ();
    type Error = Error;

    async fn select<B: Backend>(self) -> Result<()> {
        let engine = Engine::<B>::new().await.map_err(|error| anyhow!("{error}"))?;
        let device =
            engine.device_name().ok_or_else(|| anyhow!("collect-gpu-timestamps needs a GPU backend"))?.to_owned();
        let model = engine.load_language_model(&self.model_path).await.map_err(|error| anyhow!("{error}"))?;
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
        let suffix_lengths = (1..=DENSE_SUFFIX_LENGTH)
            .chain((DENSE_SUFFIX_LENGTH + 8..=MAX_SUFFIX_LENGTH).step_by(8))
            .collect::<Vec<_>>();
        fs::create_dir_all(self.output_dir.join("totals"))?;
        if self.spans {
            fs::create_dir_all(self.output_dir.join("spans"))?;
        }
        self.write_metadata(&device, context_length, &prefix_lengths, &suffix_lengths)?;
        let passes: &[bool] = if self.spans {
            &[false, true]
        } else {
            &[false]
        };
        let started = Instant::now();
        for repeat in 0..=REPEATS {
            let round_started = Instant::now();
            let mut totals =
                ParquetTable::new(&self.output_dir.join(format!("totals/round_{repeat}.parquet")), TOTALS_SCHEMA)?;
            let mut spans = self
                .spans
                .then(|| {
                    ParquetTable::new(&self.output_dir.join(format!("spans/round_{repeat}.parquet")), SPANS_SCHEMA)
                })
                .transpose()?;
            for &prefix_length in &prefix_lengths {
                for &suffix_length in &suffix_lengths {
                    for &stamped in passes {
                        state.set_context_length(prefix_length as u32);
                        let (sender, receiver) = mpsc::channel();
                        let mut options = model.default_stream_options();
                        options.sampling_method = SamplingMethod::Greedy;
                        options.timings = Some(Timings {
                            sender,
                            spans: stamped,
                        });
                        let suffix = (prefix_length..prefix_length + suffix_length)
                            .map(|position| position as u64 % vocab_size)
                            .collect::<Vec<_>>();
                        let begin = Instant::now();
                        let mut stream =
                            model.stream(&suffix, &mut state, options).map_err(|error| anyhow!("{error}"))?;
                        stream.close().await.map_err(|error| anyhow!("{error}"))?;
                        drop(stream);
                        let wall = begin.elapsed();
                        let timings = receiver.iter().collect::<Vec<_>>();
                        let gpu = timings
                            .iter()
                            .map(|timing| timing.gpu_time)
                            .sum::<Option<Duration>>()
                            .ok_or_else(|| anyhow!("backend reports no GPU time"))?;
                        totals.push_row([
                            Cell::Int64(prefix_length as i64),
                            Cell::Int64(suffix_length as i64),
                            Cell::Int64(repeat as i64),
                            Cell::Int64(stamped as i64),
                            Cell::Double(started.elapsed().as_secs_f64()),
                            Cell::Double(wall.as_secs_f64() * 1e6),
                            Cell::Double(gpu.as_secs_f64() * 1e6),
                        ])?;
                        let Some(spans) = &mut spans else {
                            continue;
                        };
                        for (command_buffer, timing) in timings.iter().enumerate() {
                            let Some(origin) = timing.spans.first().map(|span| span.start) else {
                                continue;
                            };
                            for span in &timing.spans {
                                spans.push_row([
                                    Cell::Int64(prefix_length as i64),
                                    Cell::Int64(suffix_length as i64),
                                    Cell::Int64(repeat as i64),
                                    Cell::Int64(command_buffer as i64),
                                    Cell::Utf8(&span.name),
                                    Cell::Double(span.start.duration_since(origin).as_secs_f64() * 1e6),
                                    Cell::Double(span.end.duration_since(origin).as_secs_f64() * 1e6),
                                ])?;
                            }
                        }
                    }
                }
            }
            totals.close()?;
            if let Some(spans) = spans {
                spans.close()?;
            }
            println!("Round {repeat} done in {:.1?}", round_started.elapsed());
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
