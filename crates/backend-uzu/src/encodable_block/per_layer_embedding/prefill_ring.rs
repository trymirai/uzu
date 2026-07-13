use std::{
    io,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use super::{PerLayerEmbeddingError, staging::wait_until_event_or_cancelled};
use crate::{
    backends::common::{Backend, Context, DenseBuffer, SharedEvent},
    parameters::ParameterRowSource,
};

pub const PREFILL_CHUNK_SIZE: usize = 1024;
const SLOT_COUNT: usize = 2;
const PLAN_WAIT_TIMEOUT_MS: u64 = 10;

pub(crate) fn prefill_chunk_size() -> usize {
    PREFILL_CHUNK_SIZE
}

struct PrefillRingState {
    cancelled: AtomicBool,
    error: Mutex<Option<(io::ErrorKind, String)>>,
}

impl PrefillRingState {
    fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    fn fail(
        &self,
        error: io::Error,
    ) {
        let mut stored = self.error.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if stored.is_none() {
            *stored = Some((error.kind(), error.to_string()));
        }
    }

    fn error(&self) -> Option<(io::ErrorKind, String)> {
        self.error.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlanStatus {
    Free,
    Queued,
    Ready,
}

struct PlanData {
    ids: Vec<u64>,
    chunk: usize,
    status: PlanStatus,
    ticket_active: bool,
    error: Option<(io::ErrorKind, String)>,
}

struct RequestPlan {
    data: Mutex<PlanData>,
    changed: Condvar,
}

impl RequestPlan {
    fn new(chunk_size: usize) -> Self {
        Self {
            data: Mutex::new(PlanData {
                ids: Vec::with_capacity(chunk_size),
                chunk: usize::MAX,
                status: PlanStatus::Free,
                ticket_active: false,
                error: None,
            }),
            changed: Condvar::new(),
        }
    }

    fn release(
        &self,
        chunk: usize,
    ) {
        let mut data = self.data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if data.chunk != chunk {
            return;
        }
        data.ticket_active = false;
        if data.status == PlanStatus::Ready {
            data.ids.clear();
            data.status = PlanStatus::Free;
            self.changed.notify_all();
        }
    }
}

pub struct PreparedPrefillBatch<'a, B: Backend> {
    pub allocation: &'a B::DenseBuffer,
    pub indices: &'a B::DenseBuffer,
    pub ready_event: &'a B::SharedEvent,
    pub consumed_event: &'a B::SharedEvent,
    pub value: u64,
    pub batch_size: usize,
}

pub struct PrefillChunkTicket<B: Backend> {
    chunk: usize,
    slot: usize,
    batch_size: usize,
    plan: Arc<RequestPlan>,
    ready_event: B::SharedEvent,
    released: bool,
}

impl<B: Backend> PrefillChunkTicket<B> {
    pub fn complete(mut self) -> io::Result<()> {
        if self.ready_event.signaled_value() < self.chunk as u64 + 1 {
            return Err(io::Error::other("PLE prefill rows-ready event was not signaled"));
        }

        let data = self.plan.data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if data.chunk != self.chunk || data.status != PlanStatus::Ready {
            return Err(io::Error::other("PLE prefill request state was not ready"));
        }
        let result = data.error.clone().map_or(Ok(()), |(kind, message)| Err(io::Error::new(kind, message)));
        drop(data);
        self.plan.release(self.chunk);
        self.released = true;
        result
    }
}

impl<B: Backend> Drop for PrefillChunkTicket<B> {
    fn drop(&mut self) {
        if !self.released {
            self.plan.release(self.chunk);
        }
    }
}

pub struct PrefillRing<B: Backend> {
    rows: [B::DenseBuffer; SLOT_COUNT],
    indices: B::DenseBuffer,
    ready_event: B::SharedEvent,
    consumed_event: B::SharedEvent,
    state: Arc<PrefillRingState>,
    plans: [Arc<RequestPlan>; SLOT_COUNT],
    request_tx: Option<SyncSender<usize>>,
    worker: Option<JoinHandle<()>>,
    chunk_size: usize,
    next_chunk: usize,
}

impl<B: Backend> Unpin for PrefillRing<B> {}

impl<B: Backend> PrefillRing<B> {
    pub fn new(
        context: &B::Context,
        source: ParameterRowSource,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let row_bytes = source.row_bytes();
        let slot_bytes = PREFILL_CHUNK_SIZE
            .checked_mul(row_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "PLE prefill slot size overflow"))?;
        let rows = [
            context.create_buffer(slot_bytes).map_err(PerLayerEmbeddingError::BackendError)?,
            context.create_buffer(slot_bytes).map_err(PerLayerEmbeddingError::BackendError)?,
        ];
        let indices = context
            .create_buffer(PREFILL_CHUNK_SIZE * size_of::<u64>())
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let index_slice =
            unsafe { std::slice::from_raw_parts_mut(indices.cpu_ptr().as_ptr().cast::<u64>(), PREFILL_CHUNK_SIZE) };
        for (index, value) in index_slice.iter_mut().enumerate() {
            *value = index as u64;
        }
        let row_ptrs = rows.each_ref().map(|buffer| buffer.cpu_ptr().as_ptr() as usize);
        let ready_event = context.create_shared_event().map_err(PerLayerEmbeddingError::BackendError)?;
        let consumed_event = context.create_shared_event().map_err(PerLayerEmbeddingError::BackendError)?;
        let chunk_size = prefill_chunk_size();
        let state = Arc::new(PrefillRingState::new());
        let plans = std::array::from_fn(|_| Arc::new(RequestPlan::new(chunk_size)));
        let worker_plans = plans.clone();
        let worker_state = Arc::clone(&state);
        let worker_ready_event = ready_event.clone();
        let worker_consumed_event = consumed_event.clone();
        let (request_tx, request_rx) = sync_channel(SLOT_COUNT);
        let worker = thread::spawn(move || {
            worker_loop(
                request_rx,
                worker_plans,
                row_bytes,
                row_ptrs,
                &worker_ready_event,
                &worker_consumed_event,
                &worker_state,
                |_, ids, destination| source.read_rows(ids, destination),
            );
        });

        Ok(Self {
            rows,
            indices,
            ready_event,
            consumed_event,
            state,
            plans,
            request_tx: Some(request_tx),
            worker: Some(worker),
            chunk_size,
            next_chunk: 0,
        })
    }

    pub fn stage(
        &mut self,
        chunk: usize,
        input_chunk: &[u64],
    ) -> io::Result<PrefillChunkTicket<B>> {
        if chunk != self.next_chunk {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected PLE prefill chunk {}, got {chunk}", self.next_chunk),
            ));
        }
        if input_chunk.is_empty() || input_chunk.len() > self.chunk_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("PLE prefill chunk length must be in 1..={}, got {}", self.chunk_size, input_chunk.len()),
            ));
        }
        let slot = chunk % SLOT_COUNT;
        let plan = Arc::clone(&self.plans[slot]);
        let mut data = plan.data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while data.status != PlanStatus::Free {
            if self.state.cancelled.load(Ordering::Acquire) {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "PLE prefill ring was cancelled"));
            }
            (data, _) = plan
                .changed
                .wait_timeout(data, Duration::from_millis(PLAN_WAIT_TIMEOUT_MS))
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        data.ids.extend_from_slice(input_chunk);
        data.chunk = chunk;
        data.status = PlanStatus::Queued;
        data.ticket_active = true;
        data.error = None;
        drop(data);

        if self.request_tx.as_ref().is_none_or(|sender| sender.send(slot).is_err()) {
            let mut data = plan.data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            data.ids.clear();
            data.ticket_active = false;
            data.status = PlanStatus::Free;
            plan.changed.notify_all();
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "PLE prefill worker stopped"));
        }
        self.next_chunk += 1;
        Ok(PrefillChunkTicket {
            chunk,
            slot,
            batch_size: input_chunk.len(),
            plan,
            ready_event: self.ready_event.clone(),
            released: false,
        })
    }

    pub fn prepared<'a>(
        &'a self,
        ticket: &PrefillChunkTicket<B>,
    ) -> PreparedPrefillBatch<'a, B> {
        assert!(Arc::ptr_eq(&self.plans[ticket.slot], &ticket.plan), "PLE prefill ticket belongs to another ring");
        PreparedPrefillBatch {
            allocation: &self.rows[ticket.slot],
            indices: &self.indices,
            ready_event: &self.ready_event,
            consumed_event: &self.consumed_event,
            value: ticket.chunk as u64 + 1,
            batch_size: ticket.batch_size,
        }
    }
}

impl<B: Backend> Drop for PrefillRing<B> {
    fn drop(&mut self) {
        self.state.cancelled.store(true, Ordering::Release);
        self.ready_event.signal(u64::MAX);
        self.consumed_event.signal(u64::MAX);
        for plan in &self.plans {
            plan.changed.notify_all();
        }
        self.request_tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn worker_loop<E: SharedEvent, F: FnMut(usize, &[u64], &mut [u8]) -> io::Result<()>>(
    request_rx: Receiver<usize>,
    plans: [Arc<RequestPlan>; SLOT_COUNT],
    row_bytes: usize,
    row_ptrs: [usize; SLOT_COUNT],
    ready_event: &E,
    consumed_event: &E,
    state: &PrefillRingState,
    mut read_rows: F,
) {
    while let Ok(slot) = request_rx.recv() {
        let plan = &plans[slot];
        let mut data = plan.data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let chunk = data.chunk;
        if chunk >= SLOT_COUNT
            && !wait_until_event_or_cancelled(consumed_event, (chunk - SLOT_COUNT + 1) as u64, &state.cancelled)
        {
            data.error = Some((io::ErrorKind::Interrupted, "PLE prefill ring was cancelled".into()));
            finish_plan(plan, &mut data);
            return;
        }

        let destination =
            unsafe { std::slice::from_raw_parts_mut(row_ptrs[slot] as *mut u8, data.ids.len() * row_bytes) };
        let result = if let Some(error) = state.error() {
            Err(io::Error::new(error.0, error.1))
        } else {
            catch_unwind(AssertUnwindSafe(|| read_rows(chunk, &data.ids, destination)))
                .unwrap_or_else(|_| Err(io::Error::other("PLE prefill worker panicked")))
        };
        match result {
            Ok(_) => {},
            Err(error) => {
                state.fail(io::Error::new(error.kind(), error.to_string()));
                destination.fill(0);
                data.error = Some((error.kind(), error.to_string()));
            },
        }
        finish_plan(plan, &mut data);
        ready_event.signal(chunk as u64 + 1);
    }
}

fn finish_plan(
    plan: &RequestPlan,
    data: &mut PlanData,
) {
    if data.ticket_active {
        data.status = PlanStatus::Ready;
    } else {
        data.ids.clear();
        data.status = PlanStatus::Free;
    }
    plan.changed.notify_all();
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/per_layer_embedding/prefill_ring.rs"]
mod tests;
