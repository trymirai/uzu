use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

use proc_macros::uzu_test;

use super::*;

#[derive(Clone, Default)]
struct TestEvent(Arc<AtomicU64>);

impl SharedEvent for TestEvent {
    fn signaled_value(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    fn signal(
        &self,
        value: u64,
    ) {
        self.0.fetch_max(value, Ordering::Release);
    }
}

struct Harness {
    plans: [Arc<RequestPlan>; SLOT_COUNT],
    tx: Option<SyncSender<usize>>,
    ready: TestEvent,
    consumed: TestEvent,
    worker: Option<JoinHandle<Vec<Vec<u64>>>>,
    slots: [Vec<u8>; SLOT_COUNT],
    chunk_size: usize,
}

impl Harness {
    fn new(
        chunk_size: usize,
        fail_at: Option<usize>,
    ) -> Self {
        let mut slots = [vec![0_u8; chunk_size * 8], vec![0_u8; chunk_size * 8]];
        let row_ptrs = slots.each_mut().map(|slot| slot.as_mut_ptr() as usize);
        let plans = std::array::from_fn(|_| Arc::new(RequestPlan::new(chunk_size)));
        let worker_plans = plans.clone();
        let ready = TestEvent::default();
        let worker_ready = ready.clone();
        let consumed = TestEvent::default();
        let worker_consumed = consumed.clone();
        let state = Arc::new(PrefillRingState::new());
        let worker_state = Arc::clone(&state);
        let (tx, rx) = sync_channel(SLOT_COUNT);
        let worker = thread::spawn(move || {
            let reads = Arc::new(Mutex::new(Vec::new()));
            let worker_reads = Arc::clone(&reads);
            worker_loop(
                rx,
                worker_plans,
                8,
                row_ptrs,
                &worker_ready,
                &worker_consumed,
                &worker_state,
                |chunk, ids, destination| {
                    if fail_at == Some(chunk) {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
                    }
                    worker_reads.lock().unwrap().push(ids.to_vec());
                    for (id, row) in ids.iter().zip(destination.chunks_exact_mut(8)) {
                        row.copy_from_slice(&id.to_le_bytes());
                    }
                    Ok(())
                },
            );
            drop(worker_reads);
            Arc::try_unwrap(reads).unwrap().into_inner().unwrap()
        });
        Self {
            plans,
            tx: Some(tx),
            ready,
            consumed,
            worker: Some(worker),
            slots,
            chunk_size,
        }
    }

    fn stage(
        &self,
        chunk: usize,
        ids: &[u64],
    ) {
        let slot = chunk % SLOT_COUNT;
        let plan = &self.plans[slot];
        let mut data = plan.data.lock().unwrap();
        assert_eq!(data.status, PlanStatus::Free);
        assert!(data.ids.capacity() >= self.chunk_size);
        data.ids.extend_from_slice(ids);
        data.chunk = chunk;
        data.status = PlanStatus::Queued;
        data.ticket_active = true;
        data.error = None;
        drop(data);
        self.tx.as_ref().unwrap().send(slot).unwrap();
    }

    fn complete(
        &self,
        chunk: usize,
    ) -> io::Result<()> {
        let plan = &self.plans[chunk % SLOT_COUNT];
        assert!(self.ready.wait_until_signaled_value_timeout_ms(chunk as u64 + 1, 1_000));
        let data = plan.data.lock().unwrap();
        let result = data.error.clone().map_or(Ok(()), |error| Err(io::Error::new(error.0, error.1)));
        drop(data);
        plan.release(chunk);
        self.consumed.signal(chunk as u64 + 1);
        result
    }

    fn request(
        &self,
        chunk: usize,
        ids: &[u64],
    ) -> io::Result<()> {
        self.stage(chunk, ids);
        self.complete(chunk)
    }

    fn finish(mut self) -> Vec<Vec<u64>> {
        self.tx.take();
        self.worker.take().unwrap().join().unwrap()
    }
}

#[uzu_test]
fn next_chunk_loads_before_current_chunk_is_consumed() {
    let harness = Harness::new(2, None);
    harness.stage(0, &[1, 2]);
    harness.stage(1, &[3, 4]);

    assert!(harness.ready.wait_until_signaled_value_timeout_ms(2, 1_000));
    assert_eq!(harness.consumed.signaled_value(), 0);
    assert_eq!(harness.slots[1], [3_u64.to_le_bytes(), 4_u64.to_le_bytes()].concat());

    harness.complete(0).unwrap();
    harness.complete(1).unwrap();
    harness.finish();
}

#[uzu_test]
fn bounded_plans_preserve_checksums_partial_chunks_and_slot_reuse() {
    for request_count in [2, 20, 200] {
        let chunk_size = 7;
        let harness = Harness::new(chunk_size, None);
        let mut expected_slot_checksums = [0_u64; SLOT_COUNT];
        for chunk in 0..request_count {
            let len = if chunk + 1 == request_count {
                3
            } else {
                chunk_size
            };
            let ids = (0..len).map(|index| (chunk * 100 + index) as u64).collect::<Vec<_>>();
            harness.request(chunk, &ids).unwrap();
            expected_slot_checksums[chunk % SLOT_COUNT] = ids.iter().sum();
            let checksum = harness.slots[chunk % SLOT_COUNT][..ids.len() * 8]
                .chunks_exact(8)
                .map(|row| u64::from_le_bytes(row.try_into().unwrap()))
                .sum::<u64>();
            assert_eq!(checksum, expected_slot_checksums[chunk % SLOT_COUNT]);
        }
        assert_eq!(harness.plans.len(), SLOT_COUNT);
        assert!(harness.plans.iter().all(|plan| plan.data.lock().unwrap().ids.capacity() == chunk_size));
        let reads = harness.finish();
        assert_eq!(reads.len(), request_count);
    }
}

#[uzu_test]
fn failure_zeros_signals_and_keeps_worker_alive() {
    let harness = Harness::new(4, Some(1));
    harness.request(0, &[1, 2, 3, 4]).unwrap();
    let error = harness.request(1, &[5, 6, 7, 8]).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    assert!(harness.slots[1].iter().all(|byte| *byte == 0));
    assert!(harness.request(2, &[9, 10]).is_err());
    assert_eq!(harness.ready.signaled_value(), 3);
    assert!(harness.slots[0][..16].iter().all(|byte| *byte == 0));
    harness.finish();
}

#[uzu_test]
fn cancellation_releases_worker_waiting_for_consumption() {
    let mut slots = [vec![0_u8; 16], vec![0_u8; 16]];
    let row_ptrs = slots.each_mut().map(|slot| slot.as_mut_ptr() as usize);
    let plans = std::array::from_fn(|_| Arc::new(RequestPlan::new(2)));
    let worker_plans = plans.clone();
    let ready = TestEvent::default();
    let consumed = TestEvent::default();
    let state = Arc::new(PrefillRingState::new());
    let worker_state = Arc::clone(&state);
    let (tx, rx) = sync_channel(SLOT_COUNT);
    let worker = thread::spawn(move || {
        worker_loop(rx, worker_plans, 8, row_ptrs, &ready, &consumed, &worker_state, |_, _, _| Ok(()));
    });
    {
        let mut data = plans[0].data.lock().unwrap();
        data.ids.extend_from_slice(&[5, 6]);
        data.chunk = 2;
        data.status = PlanStatus::Queued;
    }
    tx.send(0).unwrap();
    thread::sleep(Duration::from_millis(PLAN_WAIT_TIMEOUT_MS * 2));
    state.cancelled.store(true, Ordering::Release);
    drop(tx);
    worker.join().unwrap();
}
