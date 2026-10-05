//! Per-dispatch GPU time, for tuning (`UZU_AMDGPU_PROFILE=1`). Submissions record HIP events around every
//! command; completion attributes the time between them to the command's kernel. The time a command buffer
//! spans on the GPU beyond its commands is the GPU waiting for the host inside the buffer (launch overhead,
//! a suspended process); the wall time beyond the spans is the GPU idle between buffers. The totals go to
//! stderr every 10 seconds and when the context is dropped, heaviest kernels first. Single commands of
//! `OUTLIER_MILLISECONDS` or more are counted apart: they are a suspended process (the thermal guard) caught
//! between the events around a command, not kernel time. `UZU_AMDGPU_PROFILE=spans` keeps only the command
//! buffer spans and the host times (encoding, issuing, waiting), without the per-command events, which slow
//! every command down and inflate the small ones.

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    /// `UZU_AMDGPU_PROFILE=spans`: command buffer GPU spans and host times only, no per-command events (which
    /// slow every command down and inflate the small ones).
    Spans,
    /// `UZU_AMDGPU_PROFILE=timeline`: spans plus one line per command buffer (name, times, command counts).
    Timeline,
    Commands,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("UZU_AMDGPU_PROFILE").as_deref() {
        Err(_) | Ok("0") => Mode::Off,
        Ok("spans") => Mode::Spans,
        Ok("timeline") => Mode::Timeline,
        Ok(_) => Mode::Commands,
    })
}

pub fn enabled() -> bool {
    mode() != Mode::Off
}

pub fn per_command() -> bool {
    mode() == Mode::Commands
}

pub fn timeline() -> bool {
    mode() == Mode::Timeline
}

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// One command buffer on the timeline: when it was created, submitted, waited for, and what it holds.
pub struct TimelineEntry {
    name: String,
    created: Instant,
    pub submitted: Option<Instant>,
    dispatches: usize,
    copies: usize,
    fills: usize,
}

impl TimelineEntry {
    pub(super) fn new(
        name: String,
        created: Instant,
        commands: &[super::command_buffer::Command],
    ) -> Self {
        let (mut dispatches, mut copies, mut fills) = (0, 0, 0);
        for command in commands {
            match command {
                super::command_buffer::Command::Dispatch {
                    ..
                } => dispatches += 1,
                super::command_buffer::Command::Copy {
                    ..
                } => copies += 1,
                super::command_buffer::Command::Fill {
                    ..
                } => fills += 1,
            }
        }
        Self {
            name,
            created,
            submitted: None,
            dispatches,
            copies,
            fills,
        }
    }

    pub(super) fn print(
        self,
        wait_start: Instant,
        span_milliseconds: f64,
    ) {
        let at = |instant: Instant| instant.saturating_duration_since(epoch()).as_secs_f64() * 1000.0;
        let now = Instant::now();
        eprintln!(
            "[timeline] {:>18} created {:10.2} submitted {:10.2} wait {:10.2}..{:10.2} ms, GPU span {span_milliseconds:7.2} ms, {} dispatches, {} copies, {} fills",
            self.name,
            at(self.created),
            self.submitted.map_or(0.0, at),
            at(wait_start),
            at(now),
            self.dispatches,
            self.copies,
            self.fills,
        );
    }
}

/// CPU time spent on one command buffer: encoding it, issuing its commands to HIP, waiting for its completion.
#[derive(Clone, Copy, Default)]
pub struct HostTimes {
    pub encode: Duration,
    pub issue: Duration,
    pub wait: Duration,
    /// From the end of the submission to the moment the CPU saw the command buffer complete.
    pub in_flight: Duration,
}

const OUTLIER_MILLISECONDS: f64 = 100.0;
const OUTLIERS: &str = "(outliers: single commands >= 100 ms)";

#[derive(Default)]
struct Totals {
    kernels: HashMap<&'static str, (u64, f64)>,
    command_buffers: u64,
    span_milliseconds: f64,
    host: HostTimes,
    memory_calls: HashMap<&'static str, (u64, Duration)>,
    first_record: Option<Instant>,
    last_report: Option<Instant>,
}

fn totals() -> &'static Mutex<Totals> {
    static TOTALS: OnceLock<Mutex<Totals>> = OnceLock::new();
    TOTALS.get_or_init(|| Mutex::new(Totals::default()))
}

/// Kernel times of one completed command buffer and the GPU time between its first and last command.
pub fn record(
    samples: impl IntoIterator<Item = (&'static str, f64)>,
    span_milliseconds: f64,
    host: HostTimes,
) {
    let mut totals = totals().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    for (name, milliseconds) in samples {
        let name = if milliseconds >= OUTLIER_MILLISECONDS {
            OUTLIERS
        } else {
            name
        };
        let entry = totals.kernels.entry(name).or_default();
        entry.0 += 1;
        entry.1 += milliseconds;
    }
    totals.command_buffers += 1;
    totals.span_milliseconds += span_milliseconds;
    totals.host.encode += host.encode;
    totals.host.issue += host.issue;
    totals.host.wait += host.wait;
    totals.host.in_flight += host.in_flight;
    let now = Instant::now();
    totals.first_record.get_or_insert(now);
    let due = totals.last_report.is_none_or(|last| now.duration_since(last).as_secs() >= 10);
    if due {
        totals.last_report = Some(now);
        print(&totals);
    }
}

/// A HIP allocation or release (`hipHostMalloc`, `hipFree`, ...): releases wait for all queued GPU work.
pub fn record_memory_call(
    name: &'static str,
    duration: Duration,
) {
    if !enabled() {
        return;
    }
    let mut totals = totals().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = totals.memory_calls.entry(name).or_default();
    entry.0 += 1;
    entry.1 += duration;
}

pub fn report() {
    if enabled() {
        print(&totals().lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
    }
}

fn print(totals: &Totals) {
    let busy: f64 = totals.kernels.values().map(|(_, milliseconds)| milliseconds).sum();
    if totals.command_buffers == 0 {
        return;
    }
    let milliseconds = |duration: Duration| duration.as_secs_f64() * 1000.0;
    eprintln!(
        "[amdgpu profile] host: {:.1} ms encoding, {:.1} ms issuing, {:.1} ms waiting for {} command buffers; {:.1} ms from submission to observed completion",
        milliseconds(totals.host.encode),
        milliseconds(totals.host.issue),
        milliseconds(totals.host.wait),
        totals.command_buffers,
        milliseconds(totals.host.in_flight),
    );
    if !totals.memory_calls.is_empty() {
        let mut calls: Vec<_> = totals.memory_calls.iter().collect();
        calls.sort_by_key(|(name, _)| **name);
        let calls = calls
            .into_iter()
            .map(|(name, (count, duration))| format!("{name} {count} x {:.1} ms", milliseconds(*duration)))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("[amdgpu profile] memory calls: {calls}");
    }
    let wall = totals.first_record.map_or(0.0, |first| first.elapsed().as_secs_f64() * 1000.0);
    if busy == 0.0 {
        eprintln!(
            "[amdgpu profile] {} command buffers span {:.1} ms on the GPU; {wall:.1} ms wall since the first",
            totals.command_buffers, totals.span_milliseconds,
        );
        return;
    }
    eprintln!(
        "[amdgpu profile] {busy:.1} ms GPU in {} kernels; {} command buffers span {:.1} ms ({:.1} ms waiting for the host inside them); {wall:.1} ms wall since the first",
        totals.kernels.len(),
        totals.command_buffers,
        totals.span_milliseconds,
        (totals.span_milliseconds - busy).max(0.0),
    );
    let mut rows: Vec<_> = totals.kernels.iter().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    for (name, (count, milliseconds)) in rows.into_iter().take(25) {
        eprintln!(
            "  {:5.1}% {milliseconds:10.2} ms {count:8} x {:8.3} ms  {name}",
            100.0 * milliseconds / busy,
            milliseconds / *count as f64
        );
    }
}
