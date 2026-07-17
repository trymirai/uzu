use std::{
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
    bytes: usize,
    pub(crate) request: R,
}

struct StageState {
    free_slots: SyncSender<usize>,
    errors: Box<[Mutex<Option<io::Error>>]>,
    cancelled: AtomicBool,
}

impl StageState {
    fn new(slot_count: usize) -> (Self, Receiver<usize>) {
        let (free_slots, receiver) = sync_channel(slot_count);
        for slot in 0..slot_count {
            free_slots.send(slot).expect("new staging slot channel disconnected");
        }
        (
            Self {
                free_slots,
                errors: (0..slot_count).map(|_| Mutex::new(None)).collect(),
                cancelled: AtomicBool::new(false),
            },
            receiver,
        )
    }
}

#[must_use]
pub(crate) struct Reservation {
    slot: Option<usize>,
    state: Arc<StageState>,
}

impl Reservation {
    pub(crate) fn slot(&self) -> usize {
        self.slot.expect("staging reservation already submitted")
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            let _ = self.state.free_slots.send(slot);
        }
    }
}

#[must_use]
pub(crate) struct Stage<B: Backend> {
    slot: usize,
    generation: u64,
    ready_event: B::SharedEvent,
    state: Arc<StageState>,
}

impl<B: Backend> Stage<B> {
    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn complete(self) -> io::Result<()> {
        if self.state.cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "staging was cancelled"));
        }
        if self.ready_event.signaled_value() < self.generation {
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "staging ready event was not signaled"));
        }
        let error = self.state.errors[self.slot].lock().unwrap_or_else(PoisonError::into_inner).take();
        self.state
            .free_slots
            .send(self.slot)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "stager was dropped"))?;
        error.map_or(Ok(()), Err)
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
        })
    }

    pub(crate) fn reserve(&self) -> io::Result<Reservation> {
        self.reserve_slot(true)
    }

    pub(crate) fn try_reserve(&self) -> io::Result<Reservation> {
        self.reserve_slot(false)
    }

    fn reserve_slot(
        &self,
        wait: bool,
    ) -> io::Result<Reservation> {
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
        Ok(Reservation {
            slot: Some(slot),
            state: Arc::clone(&self.state),
        })
    }

    pub(crate) fn submit(
        &mut self,
        mut reservation: Reservation,
        request: R,
        bytes: usize,
    ) -> io::Result<Stage<B>> {
        if !Arc::ptr_eq(&self.state, &reservation.state) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "staging reservation belongs to another stager"));
        }
        if bytes > self.slot_bytes {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "staging request exceeds slot size"));
        }
        let generation =
            self.next_generation.checked_add(1).ok_or_else(|| io::Error::other("staging event value overflow"))?;
        let slot = reservation.slot();
        *self.state.errors[slot].lock().unwrap_or_else(PoisonError::into_inner) = None;
        self.requests
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "staging worker stopped"))?
            .send(StageRequest {
                generation,
                slot,
                bytes,
                request,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "staging worker stopped"))?;
        self.next_generation = generation;
        reservation.slot.take();
        Ok(Stage {
            slot,
            generation,
            ready_event: self.ready_event.clone(),
            state: Arc::clone(&self.state),
        })
    }

    pub(crate) fn view<'a>(
        &'a self,
        stage: &'a Stage<B>,
    ) -> StageView<'a, B> {
        assert!(Arc::ptr_eq(&self.state, &stage.state), "stage belongs to another stager");
        StageView {
            allocation: &self.buffers[stage.slot()],
            ready_event: &stage.ready_event,
            value: stage.generation,
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

fn run_worker<E, R, L>(
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
