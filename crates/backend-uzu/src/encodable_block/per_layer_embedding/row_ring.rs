use std::{
    array, io,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
};

use super::{PerLayerEmbeddingError, staging::wait_until_event_or_cancelled};
use crate::{
    backends::common::{Allocation, Backend, Context, DenseBuffer, Encoder, SharedEvent},
    parameters::ParameterRowSource,
};

const SLOT_COUNT: usize = 3;
const SLOT_FREE: u8 = 0;
const SLOT_PENDING: u8 = 1;
const SLOT_LOADING: u8 = 2;
const SLOT_READY: u8 = 3;
const SLOT_COMPLETING: u8 = 4;

struct SlotState {
    value: AtomicU64,
    lifecycle: AtomicU8,
}

impl Default for SlotState {
    fn default() -> Self {
        Self {
            value: AtomicU64::new(0),
            lifecycle: AtomicU8::new(SLOT_FREE),
        }
    }
}

struct RowRingState {
    slots: [SlotState; SLOT_COUNT],
    errors: Mutex<[Option<io::Error>; SLOT_COUNT]>,
    cancelled: AtomicBool,
}

impl Default for RowRingState {
    fn default() -> Self {
        Self {
            slots: array::from_fn(|_| SlotState::default()),
            errors: Mutex::new(array::from_fn(|_| None)),
            cancelled: AtomicBool::new(false),
        }
    }
}

fn reserve_slot(
    state: &RowRingState,
    next_value: &mut u64,
) -> io::Result<(u64, usize)> {
    let value = next_value.checked_add(1).ok_or_else(|| io::Error::other("PLE row event value overflow"))?;
    let slot_index = (value as usize - 1) % SLOT_COUNT;
    state.slots[slot_index]
        .lifecycle
        .compare_exchange(SLOT_FREE, SLOT_PENDING, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| io::Error::new(io::ErrorKind::WouldBlock, "PLE row staging ring exhausted"))?;
    *next_value = value;
    Ok((value, slot_index))
}

pub struct RowTicket<B: Backend> {
    slot: usize,
    value: u64,
    sample_event: B::SharedEvent,
    row_event: B::SharedEvent,
    state: Arc<RowRingState>,
}

pub struct PreparedRow<'a, B: Backend> {
    pub allocation: &'a B::DenseBuffer,
    pub row_event: &'a B::SharedEvent,
    pub value: u64,
}

impl<B: Backend> RowTicket<B> {
    pub fn complete(self) -> io::Result<()> {
        if self.sample_event.signaled_value() < self.value {
            return Err(io::Error::other("PLE sample event was not signaled"));
        }
        if !wait_until_event_or_cancelled(&self.row_event, self.value, &self.state.cancelled) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "PLE row staging ring was cancelled"));
        }
        let slot = &self.state.slots[self.slot];
        if slot.value.load(Ordering::Acquire) != self.value {
            return Err(io::Error::other("PLE row ticket was reused before completion"));
        }
        if slot.lifecycle.compare_exchange(SLOT_READY, SLOT_COMPLETING, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return Err(io::Error::other("PLE row ticket is not ready or was already completed"));
        }
        let error = self.state.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner())[self.slot].take();
        slot.lifecycle.store(SLOT_FREE, Ordering::Release);
        error.map_or(Ok(()), Err)
    }
}

pub struct RowRing<B: Backend> {
    mailboxes: [B::DenseBuffer; SLOT_COUNT],
    rows: [B::DenseBuffer; SLOT_COUNT],
    sample_event: B::SharedEvent,
    row_event: B::SharedEvent,
    state: Arc<RowRingState>,
    worker: Option<JoinHandle<()>>,
    next_value: u64,
}

impl<B: Backend> Unpin for RowRing<B> {}

impl<B: Backend> RowRing<B> {
    pub fn new(
        context: &B::Context,
        source: ParameterRowSource,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let row_bytes = source.row_bytes();
        let mailboxes = [
            context.create_buffer(size_of::<u32>()).map_err(PerLayerEmbeddingError::BackendError)?,
            context.create_buffer(size_of::<u32>()).map_err(PerLayerEmbeddingError::BackendError)?,
            context.create_buffer(size_of::<u32>()).map_err(PerLayerEmbeddingError::BackendError)?,
        ];
        let rows = [
            context.create_buffer(row_bytes).map_err(PerLayerEmbeddingError::BackendError)?,
            context.create_buffer(row_bytes).map_err(PerLayerEmbeddingError::BackendError)?,
            context.create_buffer(row_bytes).map_err(PerLayerEmbeddingError::BackendError)?,
        ];
        let mailbox_ptrs = mailboxes.each_ref().map(|mailbox| mailbox.cpu_ptr().as_ptr() as usize);
        let row_ptrs = rows.each_ref().map(|row| row.cpu_ptr().as_ptr() as usize);
        let sample_event = context.create_shared_event().map_err(PerLayerEmbeddingError::BackendError)?;
        let row_event = context.create_shared_event().map_err(PerLayerEmbeddingError::BackendError)?;
        let state = Arc::new(RowRingState::default());
        let worker_state = Arc::clone(&state);
        let worker_sample_event = sample_event.clone();
        let worker_row_event = row_event.clone();
        let worker = thread::spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut value = 1_u64;
                loop {
                    if !wait_until_event_or_cancelled(&worker_sample_event, value, &worker_state.cancelled) {
                        return;
                    }

                    let slot_index = (value as usize - 1) % SLOT_COUNT;
                    let slot = &worker_state.slots[slot_index];
                    slot.lifecycle.store(SLOT_LOADING, Ordering::Release);
                    let token_id = unsafe { std::ptr::read_volatile(mailbox_ptrs[slot_index] as *const u32) };
                    let destination =
                        unsafe { std::slice::from_raw_parts_mut(row_ptrs[slot_index] as *mut u8, row_bytes) };
                    let result = source.read_rows(&[u64::from(token_id)], destination);
                    if let Err(error) = result {
                        destination.fill(0);
                        worker_state.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner())[slot_index] =
                            Some(error);
                    }
                    slot.lifecycle.store(SLOT_READY, Ordering::Release);
                    worker_row_event.signal(value);
                    value = value.checked_add(1).expect("PLE row event value overflow");
                }
            }));
            if result.is_err() {
                worker_state.cancelled.store(true, Ordering::Release);
                let mut errors = worker_state.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                for (slot_index, slot) in worker_state.slots.iter().enumerate() {
                    if matches!(slot.lifecycle.load(Ordering::Acquire), SLOT_PENDING | SLOT_LOADING) {
                        let destination =
                            unsafe { std::slice::from_raw_parts_mut(row_ptrs[slot_index] as *mut u8, row_bytes) };
                        destination.fill(0);
                        errors[slot_index] = Some(io::Error::other("PLE row worker terminated unexpectedly"));
                        slot.lifecycle.store(SLOT_READY, Ordering::Release);
                    }
                }
                drop(errors);
                worker_row_event.signal(u64::MAX);
            }
        });

        Ok(Self {
            mailboxes,
            rows,
            sample_event,
            row_event,
            state,
            worker: Some(worker),
            next_value: 0,
        })
    }

    pub fn publish_sample(
        &mut self,
        sampled_token: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> io::Result<RowTicket<B>> {
        let ticket = self.reserve()?;
        encoder.encode_copy(sampled_token, 0..size_of::<u32>(), &mut self.mailboxes[ticket.slot], ..);
        encoder.signal_event(&self.sample_event, ticket.value);
        Ok(ticket)
    }

    pub fn publish_known(
        &mut self,
        token_id: u64,
    ) -> io::Result<RowTicket<B>> {
        let ticket = self.reserve()?;
        unsafe { (self.mailboxes[ticket.slot].cpu_ptr().as_ptr() as *mut u32).write(token_id as u32) };
        self.sample_event.signal(ticket.value);
        Ok(ticket)
    }

    fn reserve(&mut self) -> io::Result<RowTicket<B>> {
        let (value, slot_index) = reserve_slot(&self.state, &mut self.next_value)?;
        let slot = &self.state.slots[slot_index];
        slot.value.store(value, Ordering::Release);
        self.state.errors.lock().unwrap_or_else(|poisoned| poisoned.into_inner())[slot_index] = None;
        Ok(RowTicket {
            slot: slot_index,
            value,
            sample_event: self.sample_event.clone(),
            row_event: self.row_event.clone(),
            state: Arc::clone(&self.state),
        })
    }

    pub fn prepared<'a>(
        &'a self,
        ticket: &'a RowTicket<B>,
    ) -> PreparedRow<'a, B> {
        assert!(Arc::ptr_eq(&self.state, &ticket.state), "PLE row ticket belongs to another stream");
        PreparedRow {
            allocation: &self.rows[ticket.slot],
            row_event: &ticket.row_event,
            value: ticket.value,
        }
    }
}

impl<B: Backend> Drop for RowRing<B> {
    fn drop(&mut self) {
        self.state.cancelled.store(true, Ordering::Release);
        self.sample_event.signal(u64::MAX);
        self.row_event.signal(u64::MAX);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/per_layer_embedding/row_ring.rs"]
mod tests;
