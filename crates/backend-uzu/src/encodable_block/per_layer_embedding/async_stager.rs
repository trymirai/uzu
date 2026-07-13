use std::{
    io,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender, sync_channel},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::backends::common::{Backend, Context, DenseBuffer, SharedEvent};

const SLOT_FREE: u8 = 0;
const SLOT_PENDING: u8 = 1;
const SLOT_LOADING: u8 = 2;
const SLOT_READY: u8 = 3;
const RESERVE_WAIT_TIMEOUT_MS: u64 = 10;
const EVENT_WAIT_TIMEOUT_MS: u64 = 10;

pub(crate) fn wait_until_event_or_cancelled<E: SharedEvent>(
    event: &E,
    value: u64,
    cancelled: &AtomicBool,
) -> bool {
    while event.signaled_value() < value {
        if cancelled.load(Ordering::Acquire) {
            return false;
        }
        event.wait_until_signaled_value_timeout_ms(value, EVENT_WAIT_TIMEOUT_MS);
    }
    !cancelled.load(Ordering::Acquire)
}

pub(crate) struct StageRequest<R> {
    pub(crate) generation: u64,
    pub(crate) slot: usize,
    pub(crate) bytes: usize,
    pub(crate) wait_for_consumed: Option<u64>,
    pub(crate) request: R,
}

struct StageSlot {
    generation: AtomicU64,
    lifecycle: AtomicU8,
}

impl Default for StageSlot {
    fn default() -> Self {
        Self {
            generation: AtomicU64::new(0),
            lifecycle: AtomicU8::new(SLOT_FREE),
        }
    }
}

pub(crate) struct StageState {
    slots: Vec<StageSlot>,
    errors: Mutex<Vec<Option<io::Error>>>,
    changed: Condvar,
    cancelled: AtomicBool,
}

impl StageState {
    fn new(slot_count: usize) -> Self {
        Self {
            slots: (0..slot_count).map(|_| StageSlot::default()).collect(),
            errors: Mutex::new((0..slot_count).map(|_| None).collect()),
            changed: Condvar::new(),
            cancelled: AtomicBool::new(false),
        }
    }

    fn wait_for_free(
        &self,
        slot: usize,
    ) -> bool {
        let mut guard = self.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        while self.slots[slot].lifecycle.load(Ordering::Acquire) != SLOT_FREE {
            if self.cancelled.load(Ordering::Acquire) {
                return false;
            }
            (guard, _) = self
                .changed
                .wait_timeout(guard, Duration::from_millis(RESERVE_WAIT_TIMEOUT_MS))
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        true
    }

    fn release(
        &self,
        slot: usize,
        generation: u64,
    ) -> io::Result<()> {
        let state = &self.slots[slot];
        if state.generation.load(Ordering::Acquire) != generation {
            return Err(io::Error::other("staging slot was reused before completion"));
        }
        if state.lifecycle.compare_exchange(SLOT_READY, SLOT_FREE, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(io::Error::other("staging slot is not ready or was already released"));
        }
        self.changed.notify_all();
        Ok(())
    }

    fn store_error(
        &self,
        slot: usize,
        error: io::Error,
    ) {
        self.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner())[slot] = Some(error);
    }

    fn take_error(
        &self,
        slot: usize,
    ) -> Option<io::Error> {
        self.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner())[slot].take()
    }
}

pub(crate) struct StageTicket<B: Backend> {
    pub(crate) slot: usize,
    pub(crate) generation: u64,
    pub(crate) ready_event: B::SharedEvent,
    pub(crate) consumed_event: B::SharedEvent,
    state: Arc<StageState>,
}

impl<B: Backend> StageTicket<B> {
    pub(crate) fn complete(self) -> io::Result<()> {
        if self.ready_event.signaled_value() < self.generation {
            return Err(io::Error::other("staging ready event was not signaled"));
        }
        let error = self.state.take_error(self.slot);
        self.state.release(self.slot, self.generation)?;
        error.map_or(Ok(()), Err)
    }
}

impl<B: Backend> Drop for StageTicket<B> {
    fn drop(&mut self) {
        let state = &self.state.slots[self.slot];
        if state.generation.load(Ordering::Acquire) == self.generation
            && state.lifecycle.compare_exchange(SLOT_PENDING, SLOT_FREE, Ordering::AcqRel, Ordering::Acquire).is_ok()
        {
            self.state.changed.notify_all();
        }
    }
}

pub(crate) struct PreparedStage<'a, B: Backend> {
    pub(crate) allocation: &'a B::DenseBuffer,
    pub(crate) ready_event: &'a B::SharedEvent,
    pub(crate) consumed_event: &'a B::SharedEvent,
    pub(crate) value: u64,
}

pub(crate) struct AsyncStager<B: Backend, R: Send + 'static> {
    buffers: Vec<B::DenseBuffer>,
    ready_event: B::SharedEvent,
    consumed_event: B::SharedEvent,
    state: Arc<StageState>,
    request_tx: Option<SyncSender<StageRequest<R>>>,
    worker: Option<JoinHandle<()>>,
    slot_count: usize,
    slot_bytes: usize,
    next_generation: u64,
}

impl<B: Backend, R: Send + 'static> Unpin for AsyncStager<B, R> {}

impl<B: Backend, R: Send + 'static> AsyncStager<B, R> {
    pub(crate) fn new<L>(
        context: &B::Context,
        slot_count: usize,
        slot_bytes: usize,
        loader: L,
    ) -> Result<Self, B::Error>
    where
        L: FnMut(StageRequest<R>, &AtomicBool, &mut [u8]) -> io::Result<()> + Send + 'static,
    {
        assert!(slot_count > 0);
        let buffers = (0..slot_count).map(|_| context.create_buffer(slot_bytes)).collect::<Result<Vec<_>, _>>()?;
        let buffer_ptrs = buffers.iter().map(|buffer| buffer.cpu_ptr().as_ptr() as usize).collect::<Vec<_>>();
        let ready_event = context.create_shared_event()?;
        let consumed_event = context.create_shared_event()?;
        let state = Arc::new(StageState::new(slot_count));
        let (request_tx, request_rx) = sync_channel(slot_count);
        let worker_state = Arc::clone(&state);
        let worker_ready_event = ready_event.clone();
        let worker_consumed_event = consumed_event.clone();
        let worker = thread::spawn(move || {
            run_worker(
                request_rx,
                buffer_ptrs,
                slot_bytes,
                worker_ready_event,
                worker_consumed_event,
                worker_state,
                loader,
            );
        });

        Ok(Self {
            buffers,
            ready_event,
            consumed_event,
            state,
            request_tx: Some(request_tx),
            worker: Some(worker),
            slot_count,
            slot_bytes,
            next_generation: 0,
        })
    }

    pub(crate) fn reserve(
        &mut self,
        wait: bool,
    ) -> io::Result<StageTicket<B>> {
        let generation =
            self.next_generation.checked_add(1).ok_or_else(|| io::Error::other("staging event value overflow"))?;
        let slot = (generation as usize - 1) % self.slot_count;
        loop {
            if self.state.cancelled.load(Ordering::Acquire) {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "staging was cancelled"));
            }
            let lifecycle = &self.state.slots[slot].lifecycle;
            if lifecycle.compare_exchange(SLOT_FREE, SLOT_PENDING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                self.state.slots[slot].generation.store(generation, Ordering::Release);
                self.state.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner())[slot] = None;
                self.next_generation = generation;
                return Ok(StageTicket {
                    slot,
                    generation,
                    ready_event: self.ready_event.clone(),
                    consumed_event: self.consumed_event.clone(),
                    state: Arc::clone(&self.state),
                });
            }
            if !wait || !self.state.wait_for_free(slot) {
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "staging ring exhausted"));
            }
        }
    }

    pub(crate) fn enqueue(
        &self,
        ticket: &StageTicket<B>,
        request: R,
        bytes: usize,
        wait_for_consumed: Option<u64>,
    ) -> io::Result<()> {
        if bytes > self.slot_bytes {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "staging request exceeds slot size"));
        }
        if ticket.state.slots[ticket.slot].generation.load(Ordering::Acquire) != ticket.generation {
            return Err(io::Error::other("staging ticket generation mismatch"));
        }
        let request = StageRequest {
            generation: ticket.generation,
            slot: ticket.slot,
            bytes,
            wait_for_consumed,
            request,
        };
        self.request_tx
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "staging worker stopped"))?
            .send(request)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "staging worker stopped"))
    }

    pub(crate) fn prepared<'a>(
        &'a self,
        ticket: &'a StageTicket<B>,
    ) -> PreparedStage<'a, B> {
        assert!(Arc::ptr_eq(&self.state, &ticket.state), "staging ticket belongs to another stager");
        PreparedStage {
            allocation: &self.buffers[ticket.slot],
            ready_event: &ticket.ready_event,
            consumed_event: &ticket.consumed_event,
            value: ticket.generation,
        }
    }
}

impl<B: Backend, R: Send + 'static> Drop for AsyncStager<B, R> {
    fn drop(&mut self) {
        self.state.cancelled.store(true, Ordering::Release);
        self.ready_event.signal(u64::MAX);
        self.consumed_event.signal(u64::MAX);
        self.state.changed.notify_all();
        self.request_tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run_worker<E, R, L>(
    request_rx: Receiver<StageRequest<R>>,
    buffer_ptrs: Vec<usize>,
    slot_bytes: usize,
    ready_event: E,
    consumed_event: E,
    state: Arc<StageState>,
    mut loader: L,
) where
    E: SharedEvent,
    R: Send + 'static,
    L: FnMut(StageRequest<R>, &AtomicBool, &mut [u8]) -> io::Result<()>,
{
    while let Ok(request) = request_rx.recv() {
        if state.cancelled.load(Ordering::Acquire) {
            return;
        }
        let generation = request.generation;
        let slot_index = request.slot;
        let bytes = request.bytes;
        let slot = &state.slots[slot_index];
        slot.lifecycle.store(SLOT_LOADING, Ordering::Release);
        let destination = unsafe { std::slice::from_raw_parts_mut(buffer_ptrs[request.slot] as *mut u8, slot_bytes) };
        let result = catch_unwind(AssertUnwindSafe(|| {
            if let Some(value) = request.wait_for_consumed
                && !wait_until_event_or_cancelled(&consumed_event, value, &state.cancelled)
            {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "staging was cancelled"));
            }
            loader(request, &state.cancelled, &mut destination[..bytes])
        }))
        .unwrap_or_else(|_| Err(io::Error::other("staging worker panicked")));
        if let Err(error) = result {
            destination.fill(0);
            state.store_error(slot_index, error);
        }
        slot.lifecycle.store(SLOT_READY, Ordering::Release);
        ready_event.signal(generation);
        state.changed.notify_all();
    }
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/per_layer_embedding/async_stager.rs"]
mod tests;
