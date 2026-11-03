//! Timeline profiler for pipelined ANE prefill: where each GPU lane is busy or waiting, when each
//! ANE request is ready, runs and completes, and what every handoff costs. Driven by
//! `prefill_timeline` in `prefill_breakdown_test.rs`.
//!
//! Two views of a prefill:
//! - [`analyze_trace`]: command buffers (the segments between handoffs, timed by the GPU's own
//!   completion feedback) plus the ANE requests' CPU events. Collected without timestamp
//!   instrumentation, so it shows the run as it is. This is the one to trust for pipeline structure.
//! - [`analyze_kernels`]: per-kernel counter-heap timestamps. Instrumenting changes the run, and a
//!   kernel's span does not bracket its execution (an asynchronously dispatched kernel's time can
//!   show up as a gap before its span), so it is only used for the mix of kernel classes.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    path::PathBuf,
    time::Instant,
};

use serde_json::{Value, json};

use super::class_of;
use crate::backends::{
    common::TimestampSpan,
    metal::ane::trace::{AneRequestTrace, SegmentEnd, SegmentTrace},
};

type Interval = (f64, f64);

/// Sorted, disjoint union of `intervals`.
fn merge(mut intervals: Vec<Interval>) -> Vec<Interval> {
    intervals.retain(|(start, end)| end > start);
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<Interval> = Vec::new();
    for (start, end) in intervals {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

fn total_length(intervals: &[Interval]) -> f64 {
    intervals.iter().map(|(start, end)| end - start).sum()
}

/// Total length of `interval` covered by the merged `set`.
fn overlap(
    interval: Interval,
    set: &[Interval],
) -> f64 {
    set.iter().map(|(start, end)| (end.min(interval.1) - start.max(interval.0)).max(0.0)).sum()
}

fn intersect(
    a: &[Interval],
    b: &[Interval],
) -> Vec<Interval> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        let (start, end) = (a[i].0.max(b[j].0), a[i].1.min(b[j].1));
        if end > start {
            out.push((start, end));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// The parts of `window` outside the merged `set`.
fn complement(
    set: &[Interval],
    window: Interval,
) -> Vec<Interval> {
    let mut out = Vec::new();
    let mut cursor = window.0;
    for (start, end) in set {
        if *start > cursor {
            out.push((cursor, start.min(window.1.max(cursor))));
        }
        cursor = cursor.max(*end);
    }
    if cursor < window.1 {
        out.push((cursor, window.1));
    }
    out.retain(|(start, end)| end > start);
    out
}

fn subtract(
    a: &[Interval],
    b: &[Interval],
    window: Interval,
) -> Vec<Interval> {
    intersect(a, &complement(b, window))
}

fn percentile(
    sorted: &[f64],
    p: f64,
) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// "label: n=.. sum=.. mean=.. p50=.. p90=.. max=.." for seconds, printed in microseconds.
fn print_distribution(
    label: &str,
    seconds: &[f64],
) {
    let mut sorted = seconds.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sum: f64 = sorted.iter().sum();
    println!(
        "    {label:<44} n={:<4} sum {:>8.1} ms  mean {:>7.0} us  p50 {:>7.0} us  p90 {:>7.0} us  max {:>7.0} us",
        sorted.len(),
        sum * 1e3,
        sum / sorted.len().max(1) as f64 * 1e6,
        percentile(&sorted, 0.5) * 1e6,
        percentile(&sorted, 0.9) * 1e6,
        sorted.last().copied().unwrap_or(0.0) * 1e6,
    );
}

fn chrome_event(
    name: &str,
    pid: usize,
    tid: usize,
    start: f64,
    end: f64,
    args: Value,
) -> Value {
    json!({
        "name": name, "ph": "X", "pid": pid, "tid": tid,
        "ts": start * 1e6, "dur": (end - start).max(0.0) * 1e6, "args": args,
    })
}

fn chrome_name(
    kind: &str,
    pid: usize,
    tid: usize,
    name: &str,
) -> Value {
    json!({"name": kind, "ph": "M", "pid": pid, "tid": tid, "args": {"name": name}})
}

/// A command buffer of the prefill, in seconds from the start of the first one.
struct Segment {
    encoding: usize,
    lane: usize,
    chunk: usize,
    /// Position among the chunk's command buffers: the one ending at layer `position`'s handoff.
    position: usize,
    index: usize,
    end_kind: SegmentEnd,
    start: f64,
    end: f64,
}

/// One ANE request, in seconds from the start of the first command buffer.
struct Request {
    lane: usize,
    chunk: usize,
    layer: usize,
    value: u64,
    /// The command buffer that ends with the handoff, and the one that continues after it.
    ready_segment: usize,
    /// CPU: the request was encoded.
    encoded: f64,
    /// GPU: the work before the handoff completed.
    ready: f64,
    notified: f64,
    enqueue_returned: f64,
    /// The ANE signalled the request's `done`.
    done: f64,
    completed: f64,
    relay_started: f64,
    relay_returned: f64,
    /// GPU: the command buffer after the handoff started.
    resume: f64,
    /// The ANE serves requests in order: when it could start this one.
    ane_start: f64,
}

/// Times in seconds from the first command buffer's start, and the requests joined to the command
/// buffers around them.
fn build(
    segments: &[SegmentTrace],
    traces: &[AneRequestTrace],
) -> (Vec<Segment>, Vec<Request>) {
    // GPU clock -> CPU clock: the completion handler runs shortly after the GPU finished, so the
    // smallest (handler time - GPU end) over all command buffers is the best estimate of the offset.
    let base = segments.iter().map(|segment| segment.handled).min().unwrap();
    let from_base = |instant: Instant| {
        if instant >= base {
            instant.duration_since(base).as_secs_f64()
        } else {
            -base.duration_since(instant).as_secs_f64()
        }
    };
    let offset =
        segments.iter().map(|segment| from_base(segment.handled) - segment.gpu_end).fold(f64::INFINITY, f64::min);
    let zero = segments.iter().map(|segment| segment.gpu_start + offset).fold(f64::INFINITY, f64::min);
    let gpu_time = |gpu_seconds: f64| gpu_seconds + offset - zero;
    let cpu_time = |instant: Instant| from_base(instant) - zero;

    let mut ordered: Vec<&SegmentTrace> = segments.iter().collect();
    ordered.sort_by_key(|segment| (segment.encoding, segment.index));
    let mut chunk_starts: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for segment in &ordered {
        chunk_starts.entry((segment.encoding, segment.chunk)).or_insert(segment.index);
    }
    let segments: Vec<Segment> = ordered
        .iter()
        .map(|segment| Segment {
            encoding: segment.encoding,
            lane: segment.lane,
            chunk: segment.chunk,
            position: segment.index - chunk_starts[&(segment.encoding, segment.chunk)],
            index: segment.index,
            end_kind: segment.end,
            start: gpu_time(segment.gpu_start),
            end: gpu_time(segment.gpu_end),
        })
        .collect();

    let mut requests = Vec::new();
    for trace in traces {
        let ready_segment = segments
            .iter()
            .position(|segment| segment.lane == trace.lane && segment.end_kind == SegmentEnd::Handoff(trace.value));
        let (Some(ready_segment), Some(notified), Some(enqueue_returned), Some(completed)) =
            (ready_segment, trace.notified, trace.enqueue_returned, trace.completed)
        else {
            println!(
                "  WARNING lane {} chunk {} layer {}: incomplete request trace",
                trace.lane, trace.chunk, trace.layer
            );
            continue;
        };
        let ready = &segments[ready_segment];
        let Some(resume_segment) =
            segments.iter().position(|segment| segment.encoding == ready.encoding && segment.index == ready.index + 1)
        else {
            continue;
        };
        requests.push(Request {
            lane: trace.lane,
            chunk: trace.chunk,
            layer: trace.layer,
            value: trace.value,
            ready_segment,
            encoded: cpu_time(trace.encoded),
            ready: ready.end,
            notified: cpu_time(notified),
            enqueue_returned: cpu_time(enqueue_returned),
            done: cpu_time(trace.signalled.unwrap_or(completed)),
            completed: cpu_time(completed),
            relay_started: trace.relay_started.map_or(cpu_time(completed), cpu_time),
            relay_returned: trace.relay_returned.map_or(cpu_time(completed), cpu_time),
            resume: segments[resume_segment].start,
            ane_start: 0.0,
        });
    }
    // The ANE serves requests in order: a request cannot start before the previous one is done.
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by(|a, b| requests[*a].done.total_cmp(&requests[*b].done));
    let mut previous_done = f64::NEG_INFINITY;
    for index in order {
        let request = &mut requests[index];
        request.ane_start = request.enqueue_returned.max(previous_done).min(request.done);
        previous_done = request.done;
    }
    (segments, requests)
}

/// What a command buffer's start waited for.
#[derive(Clone, Copy, Debug)]
enum Binding {
    /// The previous command buffer's ANE round trip (request index).
    Ane(usize),
    /// The previous chunk's command buffer at the same position, on the other lane (segment index).
    Order(usize),
    /// The previous command buffer of the queue (segment index).
    Queue(usize),
    Start,
}

fn binding(
    segments: &[Segment],
    requests: &[Request],
    index: usize,
) -> (Binding, f64) {
    let segment = &segments[index];
    let lane_pred = segments
        .iter()
        .enumerate()
        .filter(|(_, other)| {
            other.encoding == segment.encoding && other.lane == segment.lane && other.index < segment.index
        })
        .max_by_key(|(_, other)| other.index)
        .map(|(position, _)| position)
        .or_else(|| {
            // The first command buffer of a later encoding follows the whole previous one.
            segments
                .iter()
                .enumerate()
                .filter(|(_, other)| other.encoding < segment.encoding)
                .max_by(|a, b| a.1.end.total_cmp(&b.1.end))
                .map(|(position, _)| position)
        });
    let mut best = (Binding::Start, f64::NEG_INFINITY);
    if let Some(pred) = lane_pred {
        let request = requests.iter().position(|request| request.ready_segment == pred);
        best = match request {
            Some(request) => (Binding::Ane(request), requests[request].relay_returned),
            None => (Binding::Queue(pred), segments[pred].end),
        };
    }
    if segment.chunk > 0 {
        let order = segments.iter().position(|other| {
            other.encoding == segment.encoding && other.chunk + 1 == segment.chunk && other.position == segment.position
        });
        if let Some(order) = order {
            if segments[order].end > best.1 {
                best = (Binding::Order(order), segments[order].end);
            }
        }
    }
    best
}

/// Seconds of `interval` inside the merged `set`, as a fraction.
fn fraction(
    interval: Interval,
    set: &[Interval],
) -> f64 {
    overlap(interval, set) / (interval.1 - interval.0).max(1e-12)
}

fn pct(
    part: f64,
    whole: f64,
) -> f64 {
    100.0 * part / whole.max(1e-12)
}

pub fn analyze_trace(
    length: usize,
    wall: f64,
    segments: &[SegmentTrace],
    traces: &[AneRequestTrace],
) {
    if segments.is_empty() {
        println!("  no command buffers traced (is UZU_ANE_TRACE set, and the backend Metal?)");
        return;
    }
    let (segments, requests) = build(segments, traces);
    let window_end = segments.iter().map(|segment| segment.end).fold(0.0, f64::max);
    let window = (0.0, window_end);
    println!(
        "  window (first command buffer start -> last end) {:.1} ms ({:.0} tok/s); harness wall {:.1} ms ({:.0} tok/s)",
        window_end * 1e3,
        length as f64 / window_end,
        wall * 1e3,
        length as f64 / wall
    );
    println!("  command buffers {}, ANE requests {} (traced {})", segments.len(), requests.len(), traces.len());
    let notify_slack: Vec<f64> = requests.iter().map(|r| r.notified - r.ready).collect();
    let early = notify_slack.iter().filter(|slack| **slack < -200e-6).count();
    println!(
        "  clock sanity: {early}/{} requests were notified >200 us before their command buffer ended (GPU/CPU clock offset error)",
        requests.len()
    );
    write_trace_json(length, &segments, &requests);

    let lane_union: Vec<Vec<Interval>> = (0..2)
        .map(|lane| {
            merge(
                segments
                    .iter()
                    .filter(|segment| segment.lane == lane)
                    .map(|segment| (segment.start, segment.end))
                    .collect(),
            )
        })
        .collect();
    let gpu_any = merge(segments.iter().map(|segment| (segment.start, segment.end)).collect());
    let ane_busy = merge(requests.iter().map(|request| (request.enqueue_returned, request.done)).collect());
    let ane_running = merge(requests.iter().map(|request| (request.ane_start, request.done)).collect());
    let waiting = merge(requests.iter().map(|request| (request.ready, request.resume)).collect());

    println!("  --- utilisation of the {:.1} ms window ---", window_end * 1e3);
    for lane in 0..2 {
        let busy = total_length(&lane_union[lane]);
        let first = lane_union[lane].first().map_or(0.0, |interval| interval.0);
        let last = lane_union[lane].last().map_or(0.0, |interval| interval.1);
        let ane_wait: f64 =
            requests.iter().filter(|request| request.lane == lane).map(|request| request.resume - request.ready).sum();
        println!(
            "    GPU lane {lane}: command buffers busy {:>7.1} ms ({:>4.1}%), waiting on ANE round trips {:>7.1} ms ({:>4.1}%), other waits {:>7.1} ms ({:>4.1}%), idle before first {:>6.1} ms, after last {:>6.1} ms",
            busy * 1e3,
            pct(busy, window_end),
            ane_wait * 1e3,
            pct(ane_wait, window_end),
            ((last - first) - busy - ane_wait).max(0.0) * 1e3,
            pct((last - first) - busy - ane_wait, window_end),
            first * 1e3,
            (window_end - last) * 1e3,
        );
    }
    println!(
        "    GPU any lane busy {:>7.1} ms ({:>4.1}%)   ANE busy (enqueue returned -> done) {:>7.1} ms ({:>4.1}%)   ANE running (excl. queued) {:>7.1} ms ({:>4.1}%)",
        length_ms(&gpu_any),
        pct(total_length(&gpu_any), window_end),
        length_ms(&ane_busy),
        pct(total_length(&ane_busy), window_end),
        length_ms(&ane_running),
        pct(total_length(&ane_running), window_end),
    );

    println!("  --- where the window goes (GPU any lane x ANE) ---");
    let both = intersect(&gpu_any, &ane_busy);
    let gpu_only = subtract(&gpu_any, &ane_busy, window);
    let ane_only = subtract(&ane_busy, &gpu_any, window);
    let neither = complement(&merge([gpu_any.clone(), ane_busy.clone()].concat()), window);
    let neither_waiting = intersect(&neither, &waiting);
    for (label, set) in [
        ("GPU busy and ANE busy (overlap)", &both),
        ("GPU busy, ANE idle", &gpu_only),
        ("ANE busy, GPU idle (ANE-bound)", &ane_only),
        ("both idle", &neither),
        ("  of which a lane waits for the ANE (handoff latency)", &neither_waiting),
    ] {
        println!("    {label:<56} {:>7.1} ms ({:>4.1}%)", length_ms(set), pct(total_length(set), window_end));
    }
    let ane_idle_waiting = subtract(&waiting, &ane_busy, window);
    println!(
        "    {:<56} {:>7.1} ms ({:>4.1}%)",
        "ANE idle while a lane waits on it",
        length_ms(&ane_idle_waiting),
        pct(total_length(&ane_idle_waiting), window_end)
    );

    println!("  --- ANE request round trip (per request) ---");
    let col = |f: &dyn Fn(&Request) -> f64| requests.iter().map(f).collect::<Vec<f64>>();
    print_distribution("CPU encoded -> GPU ready (>0: CPU is ahead)", &col(&|r| r.ready - r.encoded));
    print_distribution("GPU ready -> listener fired (notify)", &col(&|r| r.notified - r.ready));
    print_distribution("listener -> enqueue returned (CPU)", &col(&|r| r.enqueue_returned - r.notified));
    print_distribution("enqueue returned -> ANE could start (queue)", &col(&|r| r.ane_start - r.enqueue_returned));
    print_distribution("ANE run (start -> event signalled)", &col(&|r| r.done - r.ane_start));
    print_distribution("event signalled -> completion handler", &col(&|r| r.completed - r.done));
    print_distribution("relay_done call", &col(&|r| r.relay_returned - r.relay_started));
    print_distribution("ANE signalled -> relay returned", &col(&|r| r.relay_returned - r.done));
    print_distribution("relay returned -> next command buffer starts", &col(&|r| r.resume - r.relay_returned));
    print_distribution("ANE signalled -> next command buffer starts", &col(&|r| r.resume - r.done));
    print_distribution("ready -> ANE start (all handoff latency)", &col(&|r| r.ane_start - r.ready));
    print_distribution("ROUND TRIP (GPU ready -> GPU resumes)", &col(&|r| r.resume - r.ready));
    let (mut run_gpu_busy, mut run_gpu_idle) = (Vec::new(), Vec::new());
    for request in &requests {
        let busy = fraction((request.ane_start, request.done), &gpu_any);
        if busy > 0.5 {
            &mut run_gpu_busy
        } else {
            &mut run_gpu_idle
        }
        .push(request.done - request.ane_start);
    }
    print_distribution("ANE run while the GPU is busy >50% of it", &run_gpu_busy);
    print_distribution("ANE run while the GPU is busy <=50% of it", &run_gpu_idle);

    // What every command buffer's start waited for, and the lane time that cost.
    println!("  --- what each command buffer waited for (gap since the previous one on its lane) ---");
    #[derive(Default)]
    struct Waits {
        count: usize,
        notify: f64,
        enqueue: f64,
        ane_queue: f64,
        ane_run: f64,
        relay: f64,
        resume: f64,
        order: f64,
    }
    let mut waits: BTreeMap<&str, Waits> = BTreeMap::new();
    let mut resume_latency: BTreeMap<usize, f64> = BTreeMap::new();
    for index in 0..segments.len() {
        let (bound, time) = binding(&segments, &requests, index);
        let entry = match bound {
            Binding::Ane(_) => waits.entry("ANE round trip"),
            Binding::Order(_) => waits.entry("cross-lane order (previous chunk)"),
            Binding::Queue(_) => waits.entry("queue (previous command buffer)"),
            Binding::Start => continue,
        };
        let wait = entry.or_default();
        wait.count += 1;
        match bound {
            Binding::Ane(request) => {
                let r = &requests[request];
                wait.notify += r.notified - r.ready;
                wait.enqueue += r.enqueue_returned - r.notified;
                wait.ane_queue += r.ane_start - r.enqueue_returned;
                wait.ane_run += r.done - r.ane_start;
                wait.relay += r.relay_returned - r.done;
            },
            Binding::Order(other) => {
                wait.order += (segments[other].end - segments[index].start.min(segments[other].end)).max(0.0)
            },
            Binding::Queue(_) | Binding::Start => {},
        }
        wait.resume += segments[index].start - time;
        resume_latency.insert(index, segments[index].start - time);
    }
    println!(
        "    {:<36} {:>5} {:>9} {:>9} {:>10} {:>9} {:>9} {:>10} {:>10}   (ms summed over command buffers)",
        "bound by", "n", "notify", "enqueue", "ANE queue", "ANE run", "relay", "order wait", "resume lat"
    );
    for (label, wait) in &waits {
        println!(
            "    {label:<36} {:>5} {:>9.1} {:>9.1} {:>10.1} {:>9.1} {:>9.1} {:>10.1} {:>10.1}",
            wait.count,
            wait.notify * 1e3,
            wait.enqueue * 1e3,
            wait.ane_queue * 1e3,
            wait.ane_run * 1e3,
            wait.relay * 1e3,
            wait.order * 1e3,
            wait.resume * 1e3
        );
    }

    println!("  --- critical path (walking back from the last command buffer) ---");
    critical_path(&segments, &requests, window_end);

    println!(
        "  --- per layer (mean over chunks, ms): command buffer ending at the layer's handoff / ANE run / round trip ---"
    );
    let positions: BTreeSet<usize> = requests.iter().map(|request| segments[request.ready_segment].position).collect();
    for position in positions {
        let in_layer: Vec<&Request> =
            requests.iter().filter(|request| segments[request.ready_segment].position == position).collect();
        let mean = |f: &dyn Fn(&Request) -> f64| in_layer.iter().map(|r| f(r)).sum::<f64>() / in_layer.len() as f64;
        println!(
            "    layer {position:>2} (n={}): GPU phase {:>6.2}  ANE run {:>6.2}  ANE queue {:>6.2}  round trip {:>6.2}  next phase starts {:>5.2} after relay",
            in_layer.len(),
            mean(&|r| segments[r.ready_segment].end - segments[r.ready_segment].start) * 1e3,
            mean(&|r| r.done - r.ane_start) * 1e3,
            mean(&|r| r.ane_start - r.enqueue_returned) * 1e3,
            mean(&|r| r.resume - r.ready) * 1e3,
            mean(&|r| r.resume - r.relay_returned) * 1e3,
        );
    }

    println!(
        "  --- GPU phase contention: layer command buffer duration / fastest of the same layer, by ANE overlap x other-lane overlap ---"
    );
    let layer_segments: Vec<usize> = (0..segments.len())
        .filter(|index| matches!(segments[*index].end_kind, SegmentEnd::Handoff(_)) && segments[*index].position > 0)
        .collect();
    let mut best_by_position: BTreeMap<usize, f64> = BTreeMap::new();
    for index in &layer_segments {
        let segment = &segments[*index];
        let entry = best_by_position.entry(segment.position).or_insert(f64::INFINITY);
        *entry = entry.min(segment.end - segment.start);
    }
    let edges = [0.05, 0.5, 0.95];
    let bucket = |fraction: f64| edges.iter().filter(|edge| fraction > **edge).count();
    let mut grid = [[(0.0, 0.0, 0usize); 4]; 4];
    for index in &layer_segments {
        let segment = &segments[*index];
        let interval = (segment.start, segment.end);
        let cell = &mut grid[bucket(fraction(interval, &ane_running))]
            [bucket(fraction(interval, &lane_union[1 - segment.lane.min(1)]))];
        let duration = segment.end - segment.start;
        cell.0 += duration;
        cell.1 += duration / best_by_position[&segment.position];
        cell.2 += 1;
    }
    println!("    {:<12} {:>26} {:>26} {:>26} {:>26}", "", "other lane <5%", "5-50%", "50-95%", ">95%");
    for (row, label) in ["ANE <5%", "ANE 5-50%", "ANE 50-95%", "ANE >95%"].iter().enumerate() {
        let cells: Vec<String> = grid[row]
            .iter()
            .map(|(sum, ratio, count)| {
                if *count == 0 {
                    "-".to_string()
                } else {
                    format!("{:.1} ms x{:.2} (n={count})", sum / *count as f64 * 1e3, ratio / *count as f64)
                }
            })
            .collect();
        println!("    {label:<12} {:>26} {:>26} {:>26} {:>26}", cells[0], cells[1], cells[2], cells[3]);
    }
}

/// Walks back from the last command buffer through whatever bound each start, summing the time of
/// each kind of step; the steps add up to the window.
fn critical_path(
    segments: &[Segment],
    requests: &[Request],
    window_end: f64,
) {
    let mut steps: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut add = |label: &'static str, seconds: f64| *steps.entry(label).or_default() += seconds;
    let mut current = (0..segments.len()).max_by(|a, b| segments[*a].end.total_cmp(&segments[*b].end)).unwrap();
    let mut visited = 0;
    loop {
        visited += 1;
        let segment = &segments[current];
        add(
            if segment.position == 0 && matches!(segment.end_kind, SegmentEnd::Handoff(_)) {
                "GPU: first command buffer of a chunk"
            } else if matches!(segment.end_kind, SegmentEnd::Handoff(_)) {
                "GPU: layer command buffers (unpack, mixer, stage, pack)"
            } else {
                "GPU: last command buffer of a chunk / logits"
            },
            segment.end - segment.start,
        );
        let (bound, time) = binding(segments, requests, current);
        match bound {
            Binding::Ane(request) => {
                let mut r = &requests[request];
                add("GPU resume latency after relay", segment.start - time);
                add("CPU: ANE signalled -> relay returned", r.relay_returned - r.done);
                loop {
                    add("ANE run", r.done - r.ane_start);
                    if r.ane_start <= r.enqueue_returned + 1e-9 {
                        break;
                    }
                    // Queued behind the request that finished when this one started.
                    let ahead = requests
                        .iter()
                        .filter(|other| (other.done - r.ane_start).abs() < 1e-9 && other.value != r.value)
                        .min_by(|a, b| a.done.total_cmp(&b.done));
                    match ahead {
                        Some(ahead) => r = ahead,
                        None => break,
                    }
                }
                add("CPU: listener fired -> enqueue returned", r.enqueue_returned - r.notified);
                add("notify latency (GPU ready -> listener)", r.notified - r.ready);
                current = r.ready_segment;
            },
            Binding::Order(other) => {
                add("GPU resume latency after cross-lane order", segment.start - time);
                current = other;
            },
            Binding::Queue(previous) => {
                add("GPU queue latency", segment.start - time);
                current = previous;
            },
            Binding::Start => {
                add("start", segment.start);
                break;
            },
        }
        if visited > segments.len() + requests.len() {
            println!("    (path did not terminate)");
            break;
        }
    }
    let mut rows: Vec<_> = steps.iter().collect();
    rows.sort_by(|a, b| b.1.total_cmp(a.1));
    let total: f64 = steps.values().sum();
    for (label, seconds) in rows {
        println!("    {:>8.1} ms {:>5.1}%  {label}", seconds * 1e3, pct(*seconds, window_end));
    }
    println!("    {:>8.1} ms (sum of steps) vs window {:.1} ms", total * 1e3, window_end * 1e3);
}

fn write_trace_json(
    length: usize,
    segments: &[Segment],
    requests: &[Request],
) {
    const GPU_PID: usize = 10;
    const ANE_PID: usize = 20;
    const CPU_PID: usize = 30;
    let mut events = Vec::new();
    for lane in 0..2 {
        events.push(chrome_name("process_name", GPU_PID + lane, 0, &format!("GPU lane {lane}")));
        events.push(chrome_name("thread_name", GPU_PID + lane, 1, "command buffers"));
        events.push(chrome_name("thread_name", GPU_PID + lane, 2, "waiting on ANE round trip"));
    }
    events.push(chrome_name("process_name", ANE_PID, 0, "ANE"));
    events.push(chrome_name("thread_name", ANE_PID, 1, "running"));
    events.push(chrome_name("thread_name", ANE_PID, 2, "queued behind another request"));
    events.push(chrome_name("process_name", CPU_PID, 0, "CPU notify / relay"));
    for lane in 0..2 {
        events.push(chrome_name("thread_name", CPU_PID, 1 + lane * 2, &format!("lane {lane} ready -> enqueue")));
        events.push(chrome_name("thread_name", CPU_PID, 2 + lane * 2, &format!("lane {lane} ANE done -> relay")));
    }
    for segment in segments {
        events.push(chrome_event(
            &format!("c{} #{}", segment.chunk, segment.position),
            GPU_PID + segment.lane,
            1,
            segment.start,
            segment.end,
            json!({"encoding": segment.encoding, "chunk": segment.chunk, "position": segment.position, "ends": format!("{:?}", segment.end_kind)}),
        ));
    }
    for request in requests {
        let label = format!("L{} c{} lane{}", request.layer, request.chunk, request.lane);
        let args = json!({
            "layer": request.layer, "chunk": request.chunk, "lane": request.lane,
            "cpu_encoded_ms_before_ready": (request.ready - request.encoded) * 1e3,
            "notify_us": (request.notified - request.ready) * 1e6,
            "enqueue_call_us": (request.enqueue_returned - request.notified) * 1e6,
            "ane_queue_ms": (request.ane_start - request.enqueue_returned) * 1e3,
            "ane_run_ms": (request.done - request.ane_start) * 1e3,
            "signalled_to_completion_us": (request.completed - request.done) * 1e6,
            "relay_call_us": (request.relay_returned - request.relay_started) * 1e6,
            "relay_to_resume_ms": (request.resume - request.relay_returned) * 1e3,
            "round_trip_ms": (request.resume - request.ready) * 1e3,
        });
        events.push(chrome_event(
            &format!("wait {label}"),
            GPU_PID + request.lane,
            2,
            request.ready,
            request.resume,
            args.clone(),
        ));
        events.push(chrome_event(&label, ANE_PID, 1, request.ane_start, request.done, args.clone()));
        if request.ane_start > request.enqueue_returned {
            events.push(chrome_event(&label, ANE_PID, 2, request.enqueue_returned, request.ane_start, args.clone()));
        }
        events.push(chrome_event(
            &label,
            CPU_PID,
            1 + request.lane * 2,
            request.notified,
            request.enqueue_returned,
            args.clone(),
        ));
        events.push(chrome_event(
            &label,
            CPU_PID,
            2 + request.lane * 2,
            request.done,
            request.relay_returned.max(request.done),
            args,
        ));
    }
    let path = PathBuf::from(format!("/tmp/uzu_prefill_timeline_{length}.json"));
    let trace = json!({"traceEvents": events, "displayTimeUnit": "ms"});
    std::fs::write(&path, serde_json::to_string(&trace).unwrap()).unwrap();
    println!("  chrome trace (command buffers, ANE, CPU): {}", path.display());
}

// ---------------------------------------------------------------------------------------------
// Kernel-level view (counter-heap timestamps)
// ---------------------------------------------------------------------------------------------

struct GpuSpan {
    lane: usize,
    /// The span's path without the encoding/lane/chunk prefix.
    path: String,
    start: f64,
    end: f64,
    leaf: bool,
}

/// "enc 2/lane 1/chunk 3/decoder/..." -> (1, "decoder/...").
fn split_prefix(name: &str) -> (usize, &str) {
    let parsed = (|| {
        let (_, rest) = name.strip_prefix("enc ")?.split_once("/lane ")?;
        let (lane, rest) = rest.split_once("/chunk ")?;
        let (_, path) = rest.split_once('/')?;
        Some((lane.parse().ok()?, path))
    })();
    parsed.unwrap_or((0, name))
}

pub fn analyze_kernels(
    length: usize,
    wall: f64,
    untimed_wall: f64,
    spans: &[TimestampSpan],
) {
    let Some(t0) = spans.iter().map(|span| span.start).min() else {
        println!("  no timestamps recorded");
        return;
    };
    let rel = |instant: Instant| {
        if instant >= t0 {
            instant.duration_since(t0).as_secs_f64()
        } else {
            -t0.duration_since(instant).as_secs_f64()
        }
    };
    let ancestors: HashSet<&str> =
        spans.iter().flat_map(|span| span.name.match_indices('/').map(|(index, _)| &span.name[..index])).collect();
    let gpu: Vec<GpuSpan> = spans
        .iter()
        .map(|span| {
            let (lane, path) = split_prefix(&span.name);
            GpuSpan {
                lane,
                path: path.to_string(),
                start: rel(span.start),
                end: rel(span.end),
                leaf: !ancestors.contains(span.name.as_str()),
            }
        })
        .collect();
    println!(
        "  kernel timestamps slow the prefill: {:.1} ms vs {:.1} ms without ({:+.0}%); spans {} ({} leaf)",
        wall * 1e3,
        untimed_wall * 1e3,
        100.0 * (wall / untimed_wall - 1.0),
        gpu.len(),
        gpu.iter().filter(|span| span.leaf).count()
    );
    let _ = length;

    let mut events = Vec::new();
    for lane in 0..2 {
        events.push(chrome_name("process_name", 10 + lane, 0, &format!("GPU lane {lane} (kernel timestamps)")));
    }
    for span in &gpu {
        let depth = span.path.matches('/').count();
        let kernel = span.path.rsplit('/').next().unwrap_or(&span.path);
        events.push(chrome_event(kernel, 10 + span.lane, depth, span.start, span.end, json!({"path": span.path})));
    }
    let path = PathBuf::from(format!("/tmp/uzu_prefill_timeline_{length}_kernels.json"));
    std::fs::write(&path, serde_json::to_string(&json!({"traceEvents": events, "displayTimeUnit": "ms"})).unwrap())
        .unwrap();
    println!("  chrome trace (kernel timestamps, lanes only): {}", path.display());

    // A kernel's execution can show up as a gap before its span (see the module docs), so the gap
    // before a span is charged to it, except the wait of a handoff (before the unpack kernel).
    let mut classes: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut attributed_total = 0.0;
    for lane in 0..2 {
        let mut leaves: Vec<&GpuSpan> = gpu.iter().filter(|span| span.leaf && span.lane == lane).collect();
        leaves.sort_by(|a, b| a.start.total_cmp(&b.start));
        let mut previous_end: Option<f64> = None;
        for span in leaves {
            let kernel = span.path.rsplit('/').next().unwrap_or(&span.path);
            let from = match previous_end {
                Some(end) if kernel != "AneUnpack" && end < span.start => end,
                _ => span.start,
            };
            let entry = classes.entry(class_of(&span.path)).or_default();
            entry.0 += span.end - from;
            entry.1 += 1;
            attributed_total += span.end - from;
            previous_end = Some(span.end);
        }
    }
    println!(
        "  --- GPU kernel classes (span + the gap before it; {:.1} ms on both lanes, spans alone {:.1} ms) ---",
        attributed_total * 1e3,
        gpu.iter().filter(|span| span.leaf).map(|span| span.end - span.start).sum::<f64>() * 1e3
    );
    println!("    {:>9} {:>6} {:>7} {:>9}  class", "total ms", "share", "calls", "mean us");
    let mut rows: Vec<_> = classes.iter().collect();
    rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    for (class, (seconds, calls)) in rows.into_iter().take(24) {
        println!(
            "    {:>9.1} {:>5.1}% {:>7} {:>9.0}  {class}",
            seconds * 1e3,
            pct(*seconds, attributed_total),
            calls,
            seconds / *calls as f64 * 1e6
        );
    }
}

fn length_ms(intervals: &[Interval]) -> f64 {
    total_length(intervals) * 1e3
}
