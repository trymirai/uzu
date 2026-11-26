//! Export of [`TimestampSpan`]s as Chrome traces, viewable in [Perfetto](https://ui.perfetto.dev). Not available in wasm.

use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        LazyLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use serde_json::json;

use crate::{
    backends::common::{TimestampSource, TimestampSpan},
    utils::timestamps::TimestampSender,
};

const CPU_TRACK: u32 = 1;
const GPU_TRACK: u32 = 2;

struct ChromeTraceWriter<W> {
    writer: W,
    origin: Instant,
}

impl<W: io::Write> ChromeTraceWriter<W> {
    fn new(
        mut writer: W,
        origin: Instant,
    ) -> io::Result<Self> {
        let track =
            |track, name| json!({"ph": "M", "name": "thread_name", "pid": 0, "tid": track, "args": {"name": name}});
        writer.write_all(format!("[{},\n{}", track(CPU_TRACK, "CPU"), track(GPU_TRACK, "GPU")).as_bytes())?;
        writer.flush()?;
        Ok(Self {
            writer,
            origin,
        })
    }

    fn write(
        &mut self,
        spans: &[TimestampSpan],
    ) -> io::Result<()> {
        let mut batch = Vec::<u8>::new();
        for span in spans {
            let track = match span.source {
                TimestampSource::Cpu => CPU_TRACK,
                TimestampSource::Gpu => GPU_TRACK,
            };

            let name = span.name.rsplit_once('/').map_or(span.name.as_str(), |(_, innermost)| innermost);

            let micros = |dur: Duration| (dur.as_nanos() as f64 + 0.25) / 1000.0;
            let event = json!({
                "ph": "X",
                "name": name,
                "pid": 0,
                "tid": track,
                "ts": micros(span.start.saturating_duration_since(self.origin)),
                "dur": micros(span.end.saturating_duration_since(span.start)),
                "args": {"path": span.name},
            });
            write!(batch, ",\n{event}")?;
        }
        self.writer.write_all(&batch)?;
        self.writer.flush()
    }

    fn finish(mut self) -> io::Result<W> {
        writeln!(self.writer, "]")?;
        self.writer.flush()?;
        Ok(self.writer)
    }
}

/// Creates a sender if `UZU_TRACE` is set to a path. Dropping it waits until the trace file is complete.
pub fn sender_from_env() -> Option<TimestampSender> {
    static ORIGIN: LazyLock<Instant> = LazyLock::new(Instant::now);
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);

    let path = PathBuf::from(std::env::var_os("UZU_TRACE")?);
    let path = numbered(&path, TRACE_COUNT.fetch_add(1, Ordering::Relaxed));
    let origin = *ORIGIN;
    let (sender, receiver) = mpsc::channel::<Box<[TimestampSpan]>>();

    let write = move || -> io::Result<()> {
        let file = std::fs::File::create(&path)?;
        let mut trace = ChromeTraceWriter::new(file, origin)?;
        for spans in receiver {
            trace.write(&spans)?;
        }
        trace.finish()?;
        Ok(())
    };
    let spawned = std::thread::Builder::new().name("uzu-trace".into()).spawn(move || {
        if let Err(error) = write() {
            eprintln!("uzu: failed to write trace: {error}");
        }
    });
    match spawned {
        Ok(writer) => Some(TimestampSender::with_consumer(sender, writer)),
        Err(error) => {
            eprintln!("uzu: failed to start the trace writer: {error}");
            None
        },
    }
}

fn numbered(
    path: &Path,
    index: usize,
) -> PathBuf {
    if index == 0 {
        return path.to_path_buf();
    }
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    match path.extension() {
        Some(extension) => path.with_file_name(format!("{stem}.{index}.{}", extension.to_string_lossy())),
        None => path.with_file_name(format!("{stem}.{index}")),
    }
}
