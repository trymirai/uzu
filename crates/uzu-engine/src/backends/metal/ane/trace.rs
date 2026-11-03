//! Opt-in timeline of the ANE requests of pipelined prefill (`UZU_ANE_TRACE=1`).
//!
//! Every request of [`super::ffn::AneFfn`] records the CPU instants of its handoffs, in the clock
//! domain of the timestamp spans the GPU returns (`Instant`), so a harness can lay both on one
//! timeline. Without the environment variable nothing is recorded and every call is a branch.

use std::{sync::OnceLock, time::Instant};

use parking_lot::Mutex;

/// Where in a request's life an instant was taken.
#[derive(Clone, Copy)]
pub enum Mark {
    /// The listener fired: the GPU reached the request's `ready` value.
    Notified,
    /// `AneProgram::enqueue` is about to be called.
    EnqueueStarted,
    /// `AneProgram::enqueue` returned.
    EnqueueReturned,
    /// A listener on the shared event saw the ANE signal the request's `done` value.
    Signalled,
    /// The request's completion handler ran.
    Completed,
    /// `relay_done` is about to be called (it tells the lane's queue the request is done).
    RelayStarted,
    /// `relay_done` returned.
    RelayReturned,
}

// Only the timeline harness (tests) reads the recorded fields.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct AneRequestTrace {
    pub lane: usize,
    pub chunk: usize,
    pub layer: usize,
    pub rows: u32,
    /// The `ready` value of the lane's event, unique per lane.
    pub value: u64,
    pub encoded: Instant,
    pub notified: Option<Instant>,
    pub enqueue_started: Option<Instant>,
    pub enqueue_returned: Option<Instant>,
    pub signalled: Option<Instant>,
    pub completed: Option<Instant>,
    pub relay_started: Option<Instant>,
    pub relay_returned: Option<Instant>,
}

impl AneRequestTrace {
    fn slot(
        &mut self,
        mark: Mark,
    ) -> &mut Option<Instant> {
        match mark {
            Mark::Notified => &mut self.notified,
            Mark::EnqueueStarted => &mut self.enqueue_started,
            Mark::EnqueueReturned => &mut self.enqueue_returned,
            Mark::Signalled => &mut self.signalled,
            Mark::Completed => &mut self.completed,
            Mark::RelayStarted => &mut self.relay_started,
            Mark::RelayReturned => &mut self.relay_returned,
        }
    }
}

/// What ended a command buffer segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentEnd {
    /// A handoff to the ANE: the request that became ready at this value of the lane's event.
    Handoff(u64),
    /// The encoding moved to the other lane (the next chunk starts there).
    ChunkSwitch,
    /// The encoding's last command buffer.
    Final,
}

/// One command buffer of an encoding, with the GPU times its completion feedback reports.
// Only the timeline harness (tests) reads the recorded fields.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SegmentTrace {
    pub encoding: usize,
    pub lane: usize,
    pub chunk: usize,
    /// Position among the encoding's command buffers.
    pub index: usize,
    pub end: SegmentEnd,
    /// When the completion handler ran (CPU clock).
    pub handled: Instant,
    /// Seconds on the GPU's (CFTimeInterval) clock.
    pub gpu_start: f64,
    pub gpu_end: f64,
}

static REQUESTS: Mutex<Vec<AneRequestTrace>> = Mutex::new(Vec::new());
static SEGMENTS: Mutex<Vec<SegmentTrace>> = Mutex::new(Vec::new());

/// Whether `UZU_ANE_TRACE` is set (read once).
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("UZU_ANE_TRACE").is_some_and(|value| value != "0"))
}

/// Starts the trace of the request that becomes ready when `lane`'s event reaches `value`.
pub fn begin(
    lane: usize,
    chunk: usize,
    layer: usize,
    rows: u32,
    value: u64,
) {
    if !enabled() {
        return;
    }
    REQUESTS.lock().push(AneRequestTrace {
        lane,
        chunk,
        layer,
        rows,
        value,
        encoded: Instant::now(),
        notified: None,
        enqueue_started: None,
        enqueue_returned: None,
        signalled: None,
        completed: None,
        relay_started: None,
        relay_returned: None,
    });
}

/// Records now as `mark` of the request of `lane` that becomes ready at `value`.
pub fn mark(
    lane: usize,
    value: u64,
    mark: Mark,
) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let mut requests = REQUESTS.lock();
    if let Some(request) = requests.iter_mut().rev().find(|request| request.lane == lane && request.value == value) {
        *request.slot(mark) = Some(now);
    }
}

/// Takes the requests traced so far, in encoding order.
#[cfg(test)]
pub fn drain() -> Vec<AneRequestTrace> {
    std::mem::take(&mut *REQUESTS.lock())
}

/// Records a completed command buffer of an encoding.
pub fn segment(segment: SegmentTrace) {
    SEGMENTS.lock().push(segment);
}

/// Takes the command buffers traced so far.
#[cfg(test)]
pub fn drain_segments() -> Vec<SegmentTrace> {
    std::mem::take(&mut *SEGMENTS.lock())
}
