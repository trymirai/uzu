use std::{
    collections::BTreeSet,
    io,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
    },
    thread::{self, JoinHandle},
};

use crate::backends::common::{Backend, Context, DenseBuffer, SharedEvent};

const WAIT_TIMEOUT_MS: u64 = 10;

pub(crate) fn wait_until_event_or_cancelled<E: SharedEvent>(
    event: &E,
    value: u64,
    cancelled: &AtomicBool,
) -> bool {
    while event.signaled_value() < value {
        if cancelled.load(Ordering::Acquire) {
            return false;
        }
        event.wait_until_signaled_value_timeout_ms(value, WAIT_TIMEOUT_MS);
    }
    !cancelled.load(Ordering::Acquire)
}

pub(crate) struct StageRequest<R> {
    pub(crate) generation: u64,
    pub(crate) slot: usize,
    pub(crate) bytes: usize,
    pub(crate) request: R,
}

pub(crate) struct StageState {
    free_slots: SyncSender<usize>,
    errors: Box<[Mutex<Option<io::Error>>]>,
    abandoned_generations: Mutex<BTreeSet<u64>>,
    pub(crate) cancelled: AtomicBool,
}

impl StageState {
    pub(crate) fn new(slot_count: usize) -> (Self, Receiver<usize>) {
        let (free_slots, receiver) = sync_channel(slot_count);
        for slot in 0..slot_count {
            free_slots.send(slot).expect("new staging slot channel disconnected");
        }
        (
            Self {
                free_slots,
                errors: (0..slot_count).map(|_| Mutex::new(None)).collect(),
                abandoned_generations: Mutex::new(BTreeSet::new()),
                cancelled: AtomicBool::new(false),
            },
            receiver,
        )
    }
}

pub(crate) struct StageLease<B: Backend> {
    slot: Option<usize>,
    pub(crate) generation: u64,
    pub(crate) ready_event: B::SharedEvent,
    state: Arc<StageState>,
    submitted: bool,
}

impl<B: Backend> StageLease<B> {
    pub(crate) fn slot(&self) -> usize {
        self.slot.expect("staging lease already completed")
    }

    pub(crate) fn complete(mut self) -> io::Result<()> {
        if self.ready_event.signaled_value() < self.generation {
            return Err(io::Error::other("staging ready event was not signaled"));
        }
        let slot = self.slot.take().expect("staging lease already completed");
        let error = self.state.errors[slot].lock().unwrap_or_else(PoisonError::into_inner).take();
        self.state
            .free_slots
            .send(slot)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "stager was dropped"))?;
        error.map_or(Ok(()), Err)
    }
}

impl<B: Backend> Drop for StageLease<B> {
    fn drop(&mut self) {
        // Submitted slots return only after the consuming command buffer completes.
        if !self.submitted
            && let Some(slot) = self.slot.take()
        {
            self.state.abandoned_generations.lock().unwrap_or_else(PoisonError::into_inner).insert(self.generation);
            let _ = self.state.free_slots.send(slot);
        }
    }
}

pub(crate) struct StageView<'a, B: Backend> {
    pub(crate) allocation: &'a B::DenseBuffer,
    pub(crate) ready_event: &'a B::SharedEvent,
    pub(crate) value: u64,
}

pub(crate) struct AsyncStager<B: Backend, R: Send + 'static> {
    buffers: Vec<B::DenseBuffer>,
    ready_event: B::SharedEvent,
    state: Arc<StageState>,
    free_slots: Receiver<usize>,
    requests: Option<SyncSender<StageRequest<R>>>,
    worker: Option<JoinHandle<()>>,
    slot_bytes: usize,
    next_generation: u64,
    next_enqueue_generation: u64,
}

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
        let pointers = buffers.iter().map(|buffer| buffer.cpu_ptr().as_ptr() as usize).collect();
        let ready_event = context.create_shared_event()?;
        let (state, free_slots) = StageState::new(slot_count);
        let state = Arc::new(state);
        let (requests, receiver) = sync_channel(slot_count);
        let worker_state = Arc::clone(&state);
        let worker_event = ready_event.clone();
        let worker =
            thread::spawn(move || run_worker(receiver, pointers, slot_bytes, worker_event, worker_state, loader));
        Ok(Self {
            buffers,
            ready_event,
            state,
            free_slots,
            requests: Some(requests),
            worker: Some(worker),
            slot_bytes,
            next_generation: 0,
            next_enqueue_generation: 1,
        })
    }

    pub(crate) fn reserve(
        &mut self,
        wait: bool,
    ) -> io::Result<StageLease<B>> {
        if self.state.cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "staging was cancelled"));
        }
        let slot = if wait {
            self.free_slots.recv().map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "stager was dropped"))?
        } else {
            self.free_slots.try_recv().map_err(|error| match error {
                TryRecvError::Empty => io::Error::new(io::ErrorKind::WouldBlock, "all staging slots are occupied"),
                TryRecvError::Disconnected => io::Error::new(io::ErrorKind::BrokenPipe, "stager was dropped"),
            })?
        };
        let generation =
            self.next_generation.checked_add(1).ok_or_else(|| io::Error::other("staging event value overflow"))?;
        self.next_generation = generation;
        *self.state.errors[slot].lock().unwrap_or_else(PoisonError::into_inner) = None;
        Ok(StageLease {
            slot: Some(slot),
            generation,
            ready_event: self.ready_event.clone(),
            state: Arc::clone(&self.state),
            submitted: false,
        })
    }

    pub(crate) fn enqueue(
        &mut self,
        lease: &mut StageLease<B>,
        request: R,
        bytes: usize,
    ) -> io::Result<()> {
        if bytes > self.slot_bytes {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "staging request exceeds slot size"));
        }
        let mut abandoned = self.state.abandoned_generations.lock().unwrap_or_else(PoisonError::into_inner);
        while abandoned.remove(&self.next_enqueue_generation) {
            self.next_enqueue_generation += 1;
        }
        if lease.generation != self.next_enqueue_generation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "staging requests must be enqueued in generation order",
            ));
        }
        drop(abandoned);
        self.requests
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "staging worker stopped"))?
            .send(StageRequest {
                generation: lease.generation,
                slot: lease.slot(),
                bytes,
                request,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "staging worker stopped"))?;
        self.next_enqueue_generation += 1;
        lease.submitted = true;
        Ok(())
    }

    pub(crate) fn view<'a>(
        &'a self,
        lease: &'a StageLease<B>,
    ) -> StageView<'a, B> {
        assert!(Arc::ptr_eq(&self.state, &lease.state), "staging lease belongs to another stager");
        StageView {
            allocation: &self.buffers[lease.slot()],
            ready_event: &lease.ready_event,
            value: lease.generation,
        }
    }
}

impl<B: Backend, R: Send + 'static> Drop for AsyncStager<B, R> {
    fn drop(&mut self) {
        self.state.cancelled.store(true, Ordering::Release);
        self.ready_event.signal(u64::MAX);
        self.requests.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(crate) fn run_worker<E, R, L>(
    receiver: Receiver<StageRequest<R>>,
    pointers: Vec<usize>,
    slot_bytes: usize,
    ready_event: E,
    state: Arc<StageState>,
    mut loader: L,
) where
    E: SharedEvent,
    R: Send + 'static,
    L: FnMut(StageRequest<R>, &AtomicBool, &mut [u8]) -> io::Result<()>,
{
    while let Ok(request) = receiver.recv() {
        if state.cancelled.load(Ordering::Acquire) {
            return;
        }
        let (slot, generation, bytes) = (request.slot, request.generation, request.bytes);
        let destination = unsafe { std::slice::from_raw_parts_mut(pointers[slot] as *mut u8, slot_bytes) };
        let result = catch_unwind(AssertUnwindSafe(|| loader(request, &state.cancelled, &mut destination[..bytes])))
            .unwrap_or_else(|_| Err(io::Error::other("staging worker panicked")));
        if let Err(error) = result {
            destination.fill(0);
            *state.errors[slot].lock().unwrap_or_else(PoisonError::into_inner) = Some(error);
        }
        ready_event.signal(generation);
    }
}

#[cfg(test)]
#[path = "../unit/staging.rs"]
mod tests;
